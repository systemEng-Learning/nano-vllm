import torch
from torch import nn

from nanovllm.utils.context import get_context
from nanovllm.kvcache.base import BaseKVCache
from nanovllm.layers.flash_attn_backend import (
    BaseFlashAttentionBackend,
    FlashAttentionRegistry,
    ensure_builtin_backends_registered,
)


class Attention(nn.Module):
    """
    Attention module with pluggable KV cache and flash attention backends.
    
    This module supports:
    - Different KV cache implementations (FP16, INT8, INT4, etc.)
    - Different flash attention backends (standard, quantized-aware, custom)
    
    The separation allows you to:
    - Use INT8 cache with dequantization + standard flash-attn
    - OR use INT8 cache with custom INT8-aware flash-attn kernel
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
        cache_backend: BaseKVCache | None = None,
        attn_backend: BaseFlashAttentionBackend | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.cache_backend = cache_backend
        
        # Flash attention backend - defaults to "default" implementation
        if attn_backend is None:
            ensure_builtin_backends_registered()
            attn_backend = FlashAttentionRegistry.get("default")()
        self.attn_backend = attn_backend
        
        # Cache tensors (will be set by model runner)
        self.k_cache = self.v_cache = torch.tensor([])
        self.additional_cache_tensors = ()  # For quantized caches (scales, etc.)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        def get_attention_cache_tensors():
            if self.attn_backend.supports_quantized_cache_inputs:
                return k_cache, v_cache, self.additional_cache_tensors
            if self.cache_backend is None or not self.cache_backend.needs_dequantization():
                return k_cache, v_cache, self.additional_cache_tensors
            try:
                attn_k, attn_v = self.cache_backend.retrieve(
                    k_cache,
                    v_cache,
                    *self.additional_cache_tensors,
                )
            except NotImplementedError as exc:
                raise NotImplementedError(
                    f"{self.cache_backend.name} does not provide a dequantized KV cache path "
                    "for attention. A fused attention kernel is required for this cache format."
                ) from exc
            return attn_k, attn_v, ()
        
        # Store KV pairs in cache using the cache backend
        if k_cache.numel() and v_cache.numel():
            if self.cache_backend is not None:
                # Store supports additional tensors for quantized caches
                self.cache_backend.store(
                    k, v, k_cache, v_cache, context.slot_mapping,
                    *self.additional_cache_tensors
                )
            else:
                raise RuntimeError(
                    "No KV cache backend configured. This should not happen - "
                    "ensure DefaultKVCache is imported and registered."
                )
        
        # Perform attention using pluggable backend
        if context.is_prefill:
            # For prefill, use keys/values or cache depending on prefix caching
            if context.block_tables is not None:  # prefix cache
                k, v, attention_cache_tensors = get_attention_cache_tensors()
            else:
                attention_cache_tensors = self.additional_cache_tensors
            
            o = self.attn_backend.prefill(
                q, k, v,
                scale=self.scale,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                block_table=context.block_tables,
                *attention_cache_tensors,
            )
        else:  # decode
            attn_k_cache, attn_v_cache, attention_cache_tensors = get_attention_cache_tensors()
            # Pass all cache tensors to backend (quantized backends need scales, etc.)
            o = self.attn_backend.decode(
                q,
                attn_k_cache,
                attn_v_cache,
                scale=self.scale,
                cache_seqlens=context.context_lens,
                block_table=context.block_tables,
                *attention_cache_tensors,
            )
        
        return o
