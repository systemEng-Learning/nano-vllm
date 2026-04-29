"""Default attention backend using FlashInfer."""

from nanovllm.layers.flash_attn_backend import FlashAttentionRegistry
from nanovllm.layers.flashinfer_flash_attn import FlashInferAttention


@FlashAttentionRegistry.register("default")
class DefaultFlashAttention(FlashInferAttention):
    """Default FlashInfer-backed attention for FP16/BF16 KV cache."""
