"""INT8 KV cache (per-token symmetric absmax) for nano-vllm."""
import torch
import triton
import triton.language as tl
from typing import Tuple
from nanovllm.kvcache.base import BaseKVCache, KVCacheRegistry


@triton.jit
def store_kvcache_int8_kernel(
    key_ptr, key_stride, value_ptr, value_stride,
    k_cache_ptr, v_cache_ptr, k_scale_ptr, v_scale_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return

    offsets = tl.arange(0, D)
    key = tl.load(key_ptr + idx * key_stride + offsets).to(tl.float32)
    value = tl.load(value_ptr + idx * value_stride + offsets).to(tl.float32)

    # scale = max(|x|) / 127; guard against all-zero tokens
    k_scale = tl.max(tl.abs(key), axis=0) / 127.0
    v_scale = tl.max(tl.abs(value), axis=0) / 127.0
    k_scale = tl.where(k_scale == 0.0, 1.0, k_scale)
    v_scale = tl.where(v_scale == 0.0, 1.0, v_scale)

    k_quant = tl.clamp(tl.math.llrint(key / k_scale), -128, 127).to(tl.int8)
    v_quant = tl.clamp(tl.math.llrint(value / v_scale), -128, 127).to(tl.int8)

    cache_offsets = slot * D + offsets
    tl.store(k_cache_ptr + cache_offsets, k_quant)
    tl.store(v_cache_ptr + cache_offsets, v_quant)
    tl.store(k_scale_ptr + slot, k_scale)
    tl.store(v_scale_ptr + slot, v_scale)


@triton.jit
def load_kvcache_int8_kernel(
    k_cache_ptr, v_cache_ptr, k_scale_ptr, v_scale_ptr,
    key_out_ptr, value_out_ptr,
    num_slots: tl.constexpr,
    D: tl.constexpr,
):
    slot = tl.program_id(0)
    if slot >= num_slots:
        return

    offsets = tl.arange(0, D)
    cache_offsets = slot * D + offsets
    key = tl.load(k_cache_ptr + cache_offsets).to(tl.float32) * tl.load(k_scale_ptr + slot)
    value = tl.load(v_cache_ptr + cache_offsets).to(tl.float32) * tl.load(v_scale_ptr + slot)
    tl.store(key_out_ptr + cache_offsets, key)
    tl.store(value_out_ptr + cache_offsets, value)


@KVCacheRegistry.register("int8")
class Int8KVCache(BaseKVCache):
    """Per-token symmetric absmax INT8 KV cache."""

    def allocate(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (k_cache, v_cache, k_scales, v_scales)."""
        cache_shape = (self.num_blocks, self.block_size, self.num_heads, self.head_dim)
        num_slots = self.num_blocks * self.block_size
        k_cache = torch.zeros(cache_shape, dtype=torch.int8, device=self.device)
        v_cache = torch.zeros(cache_shape, dtype=torch.int8, device=self.device)
        k_scales = torch.ones(num_slots, dtype=torch.float32, device=self.device)
        v_scales = torch.ones(num_slots, dtype=torch.float32, device=self.device)
        return k_cache, v_cache, k_scales, v_scales

    def store(self, key, value, k_cache, v_cache, slot_mapping, *additional_tensors):
        if len(additional_tensors) < 2:
            raise ValueError("store() requires k_scales and v_scales in additional_tensors")
        k_scales, v_scales = additional_tensors[0], additional_tensors[1]

        N, num_heads, head_dim = key.shape
        D = num_heads * head_dim

        assert key.stride(-1) == 1 and value.stride(-1) == 1
        assert k_cache.stride(1) == D and v_cache.stride(1) == D
        assert slot_mapping.numel() == N

        store_kvcache_int8_kernel[(N,)](
            key, key.stride(0), value, value.stride(0),
            k_cache.view(-1), v_cache.view(-1),
            k_scales, v_scales, slot_mapping,
            D,
        )

    def retrieve(self, k_cache, v_cache, *additional_tensors) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(additional_tensors) < 2:
            raise ValueError("retrieve() requires k_scales and v_scales in additional_tensors")
        k_scales, v_scales = additional_tensors[0], additional_tensors[1]

        num_slots = k_cache.shape[0] * k_cache.shape[1]
        D = k_cache.shape[2] * k_cache.shape[3]
        k_out = torch.empty(num_slots * D, dtype=torch.float32, device=k_cache.device)
        v_out = torch.empty(num_slots * D, dtype=torch.float32, device=v_cache.device)

        load_kvcache_int8_kernel[(num_slots,)](
            k_cache.view(-1), v_cache.view(-1),
            k_scales, v_scales, k_out, v_out,
            num_slots, D,
        )
        return k_out.view(k_cache.shape).to(self.dtype), v_out.view(k_cache.shape).to(self.dtype)

    def needs_dequantization(self) -> bool:
        return True

    def get_cache_block_size_bytes(self) -> int:
        int8_bytes = 2 * self.block_size * self.num_heads * self.head_dim
        scale_bytes = 2 * self.block_size * 4  # float32 scales
        return int8_bytes + scale_bytes
