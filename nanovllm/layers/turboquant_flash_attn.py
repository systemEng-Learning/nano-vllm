"""TurboQuant-aware flash attention backend.

This backend forwards the packed TurboQuant KV cache directly to the imported
flash-attention kernels. We assume those kernels understand the packed cache
format and perform any required dequantization internally.
"""
from typing import Optional

import torch
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from nanovllm.layers.flash_attn_backend import (
    BaseFlashAttentionBackend,
    FlashAttentionRegistry,
)


@FlashAttentionRegistry.register("turboquant")
class TurboQuantFlashAttention(BaseFlashAttentionBackend):
    """Flash-attention backend for a fused TurboQuant-aware kernel."""

    def prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float,
        max_seqlen_q: Optional[int],
        cu_seqlens_q: Optional[torch.Tensor],
        max_seqlen_k: Optional[int],
        cu_seqlens_k: Optional[torch.Tensor],
        block_table: Optional[torch.Tensor] = None,
        *additional_cache_tensors,
    ) -> torch.Tensor:
        return flash_attn_varlen_func(
            q,
            k,
            v,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=max_seqlen_k,
            cu_seqlens_k=cu_seqlens_k,
            softmax_scale=scale,
            causal=True,
            block_table=block_table,
            *additional_cache_tensors,
        )

    def decode(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        scale: float,
        cache_seqlens: Optional[torch.Tensor],
        block_table: Optional[torch.Tensor],
        *additional_cache_tensors,
    ) -> torch.Tensor:
        if q.ndim == 2:
            q = q.unsqueeze(1)
        elif q.ndim == 3 and q.shape[1] != 1:
            q = q.unsqueeze(1)

        return flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            softmax_scale=scale,
            causal=True,
            *additional_cache_tensors,
        )
