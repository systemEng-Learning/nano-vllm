"""INT8 attention backend routed through FlashInfer."""

from nanovllm.layers.flash_attn_backend import FlashAttentionRegistry
from nanovllm.layers.flashinfer_flash_attn import FlashInferAttention


@FlashAttentionRegistry.register("int8")
class Int8FlashAttention(FlashInferAttention):
    """FlashInfer-backed attention for INT8 KV cache after dequantization."""
