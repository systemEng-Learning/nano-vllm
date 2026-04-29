"""TurboQuant-style 4-bit KV cache for nano-vllm.

Based on: turboquant-pytorch/lloyd_max.py (Zandieh et al.) and vllm
"""
import math
from functools import lru_cache
from typing import Tuple

import torch
import triton
import triton.language as tl

from nanovllm.kvcache.base import BaseKVCache, KVCacheRegistry


def _trapz(f, a: float, b: float, n: int = 200) -> float:
    h = (b - a) / n
    result = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        result += f(a + i * h)
    return result * h


def _gaussian_pdf(x: float, sigma2: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi * sigma2)) * math.exp(
        -(x * x) / (2.0 * sigma2)
    )


@lru_cache(maxsize=16)
def _lloyd_max_centroids(head_dim: int, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    n_levels = 2**bits
    sigma2 = 1.0 / head_dim
    sigma = math.sqrt(sigma2)

    def pdf(x: float) -> float:
        return _gaussian_pdf(x, sigma2)

    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]
    for _ in range(200):
        boundaries = [
            (centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)
        ]
        edges = [lo * 3.0] + boundaries + [hi * 3.0]
        new_centroids = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            num = _trapz(lambda x: x * pdf(x), a, b)
            den = _trapz(pdf, a, b)
            new_centroids.append(num / den if den > 1e-15 else centroids[i])
        if max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels)) < 1e-10:
            centroids = new_centroids
            break
        centroids = new_centroids

    boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
    return (
        torch.tensor(centroids, dtype=torch.float32),
        torch.tensor(boundaries, dtype=torch.float32),
    )


@lru_cache(maxsize=16)
def _hadamard_matrix(head_dim: int) -> torch.Tensor:
    if head_dim < 1 or (head_dim & (head_dim - 1)) != 0:
        raise ValueError(
            "TurboQuantKVCache requires a power-of-two head_dim for Hadamard rotation"
        )

    matrix = torch.tensor([[1.0]], dtype=torch.float32)
    base = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float32)
    dim = 1
    while dim < head_dim:
        matrix = torch.kron(matrix, base)
        dim *= 2
    return matrix / math.sqrt(head_dim)


@triton.jit
def store_kvcache_turboquant_kernel(
    key_ptr,
    key_token_stride,
    key_head_stride,
    value_ptr,
    value_token_stride,
    value_head_stride,
    k_cache_ptr,
    v_cache_ptr,
    k_norms_ptr,
    v_scales_ptr,
    v_zeros_ptr,
    slot_mapping_ptr,
    rotation_ptr,
    boundaries_ptr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    key_packed_bytes: tl.constexpr,
    value_packed_bytes: tl.constexpr,
    num_boundaries: tl.constexpr,
):
    pid = tl.program_id(0)
    token_idx = pid // num_heads
    head_idx = pid % num_heads
    slot = tl.load(slot_mapping_ptr + token_idx)
    if slot == -1:
        return

    offsets = tl.arange(0, head_dim)
    key = tl.load(
        key_ptr + token_idx * key_token_stride + head_idx * key_head_stride + offsets
    ).to(tl.float32)
    value = tl.load(
        value_ptr + token_idx * value_token_stride + head_idx * value_head_stride + offsets
    ).to(tl.float32)

    key_norm = tl.sqrt(tl.sum(key * key, axis=0))
    safe_key_norm = tl.where(key_norm > 0.0, key_norm, 1.0)
    key_unit = key / safe_key_norm

    row_offsets = tl.arange(0, head_dim)[:, None]
    col_offsets = tl.arange(0, head_dim)[None, :]
    rotation = tl.load(rotation_ptr + row_offsets * head_dim + col_offsets)
    rotated_key = tl.sum(rotation * key_unit[None, :], axis=1)

    key_indices = tl.zeros([head_dim], dtype=tl.int32)
    for i in range(num_boundaries):
        boundary = tl.load(boundaries_ptr + i)
        key_indices += (rotated_key > boundary).to(tl.int32)

    key_cache_base = (slot * num_heads + head_idx) * key_packed_bytes
    for i in range(key_packed_bytes):
        lo = key_indices[2 * i]
        hi = 0
        if 2 * i + 1 < head_dim:
            hi = key_indices[2 * i + 1]
        packed = lo | (hi << 4)
        tl.store(k_cache_ptr + key_cache_base + i, packed.to(tl.uint8))

    value_min = tl.min(value, axis=0)
    value_max = tl.max(value, axis=0)
    value_scale = (value_max - value_min) / 15.0
    value_scale = tl.where(value_scale > 0.0, value_scale, 1.0)
    value_indices = tl.clamp(
        tl.math.llrint((value - value_min) / value_scale),
        0,
        15,
    ).to(tl.int32)

    value_cache_base = (slot * num_heads + head_idx) * value_packed_bytes
    for i in range(value_packed_bytes):
        lo = value_indices[2 * i]
        hi = 0
        if 2 * i + 1 < head_dim:
            hi = value_indices[2 * i + 1]
        packed = lo | (hi << 4)
        tl.store(v_cache_ptr + value_cache_base + i, packed.to(tl.uint8))

    meta_idx = slot * num_heads + head_idx
    tl.store(k_norms_ptr + meta_idx, safe_key_norm.to(tl.float16))
    tl.store(v_scales_ptr + meta_idx, value_scale.to(tl.float16))
    tl.store(v_zeros_ptr + meta_idx, value_min.to(tl.float16))


@triton.jit
def dequantize_kvcache_turboquant_kernel(
    k_cache_ptr,
    v_cache_ptr,
    block_table_ptr,
    k_norms_ptr,
    v_scales_ptr,
    v_zeros_ptr,
    k_centroids_ptr,
    k_out_ptr,
    v_out_ptr,
    k_cache_stride_block,
    k_cache_stride_pos,
    k_cache_stride_head,
    v_cache_stride_block,
    v_cache_stride_pos,
    v_cache_stride_head,
    block_table_stride,
    meta_stride_block,
    meta_stride_pos,
    meta_stride_head,
    out_stride_token,
    out_stride_head,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    num_heads: tl.constexpr,
    key_packed_bytes: tl.constexpr,
    value_packed_bytes: tl.constexpr,
    centroid_count: tl.constexpr,
    block_d: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    page_idx = token_idx // block_size
    page_off = token_idx % block_size
    block_num = tl.load(block_table_ptr + page_idx * block_table_stride).to(tl.int64)

    k_base = (
        block_num * k_cache_stride_block
        + page_off.to(tl.int64) * k_cache_stride_pos
        + tl.cast(head_idx, tl.int64) * k_cache_stride_head
    )
    v_base = (
        block_num * v_cache_stride_block
        + page_off.to(tl.int64) * v_cache_stride_pos
        + tl.cast(head_idx, tl.int64) * v_cache_stride_head
    )
    meta_base = (
        block_num * meta_stride_block
        + page_off.to(tl.int64) * meta_stride_pos
        + tl.cast(head_idx, tl.int64) * meta_stride_head
    )
    out_base = token_idx * out_stride_token + head_idx * out_stride_head

    d_offs = tl.arange(0, block_d)
    d_mask = d_offs < head_dim
    byte_idx = d_offs // 2
    bit_shift = (d_offs % 2) * 4

    k_packed = tl.load(
        k_cache_ptr + k_base + byte_idx,
        mask=d_mask & (byte_idx < key_packed_bytes),
        other=0,
    ).to(tl.int32)
    k_idx = ((k_packed >> bit_shift) & 0xF).to(tl.int32)
    k_idx = tl.where(d_mask, k_idx, 0)
    centroid_mask = d_mask & (k_idx < centroid_count)
    k_vals = tl.load(k_centroids_ptr + k_idx, mask=centroid_mask, other=0.0).to(tl.float32)
    k_norm = tl.load(k_norms_ptr + meta_base).to(tl.float32)
    tl.store(k_out_ptr + out_base + d_offs, (k_vals * k_norm).to(k_out_ptr.dtype.element_ty), mask=d_mask)

    v_packed = tl.load(
        v_cache_ptr + v_base + byte_idx,
        mask=d_mask & (byte_idx < value_packed_bytes),
        other=0,
    ).to(tl.int32)
    v_idx = ((v_packed >> bit_shift) & 0xF).to(tl.float32)
    v_scale = tl.load(v_scales_ptr + meta_base).to(tl.float32)
    v_zero = tl.load(v_zeros_ptr + meta_base).to(tl.float32)
    v_vals = v_idx * v_scale + v_zero
    tl.store(v_out_ptr + out_base + d_offs, v_vals.to(v_out_ptr.dtype.element_ty), mask=d_mask)


def dequantize_kvcache_turboquant(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seq_len: int,
    block_table_row: torch.Tensor,
    k_norms: torch.Tensor,
    v_scales: torch.Tensor,
    v_zeros: torch.Tensor,
    k_centroids: torch.Tensor,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if seq_len <= 0:
        raise ValueError("TurboQuant dequantization expects a positive sequence length.")

    head_dim = k_norms.shape[-1] if k_norms.ndim > 3 else k_cache.shape[-1] * 2
    num_heads = k_cache.shape[2]
    block_size = k_cache.shape[1]
    block_d = triton.next_power_of_2(head_dim)
    block_table_row = block_table_row.to(device=k_cache.device, dtype=torch.int32).contiguous()
    dense_k = torch.empty((seq_len, num_heads, head_dim), dtype=out_dtype, device=k_cache.device)
    dense_v = torch.empty_like(dense_k)

    grid = (seq_len, num_heads)
    dequantize_kvcache_turboquant_kernel[grid](
        k_cache,
        v_cache,
        block_table_row,
        k_norms,
        v_scales,
        v_zeros,
        k_centroids,
        dense_k,
        dense_v,
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        block_table_row.stride(0),
        k_norms.stride(0),
        k_norms.stride(1),
        k_norms.stride(2),
        dense_k.stride(0),
        dense_k.stride(1),
        head_dim=head_dim,
        block_size=block_size,
        num_heads=num_heads,
        key_packed_bytes=k_cache.shape[-1],
        value_packed_bytes=v_cache.shape[-1],
        centroid_count=k_centroids.numel(),
        block_d=block_d,
    )
    return dense_k, dense_v


@KVCacheRegistry.register("turboquant")
class TurboQuantKVCache(BaseKVCache):
    """
    TurboQuant-style cache with 4-bit packed storage.

    Keys are L2-normalized, Hadamard-rotated, then Lloyd-Max quantized.
    Values use per-vector affine 4-bit quantization.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        key_bits: int = 4,
        value_bits: int = 4,
    ):
        super().__init__(
            num_blocks=num_blocks,
            block_size=block_size,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )
        if key_bits != 4 or value_bits != 4:
            raise ValueError(
                "TurboQuantKVCache currently supports only the 4-bit T4-oriented layout"
            )
        if head_dim < 1 or (head_dim & (head_dim - 1)) != 0:
            raise ValueError(
                "TurboQuantKVCache requires a power-of-two head_dim for Hadamard rotation"
            )

        self.key_bits = key_bits
        self.value_bits = value_bits
        self.key_packed_bytes = math.ceil(head_dim * key_bits / 8)
        self.value_packed_bytes = math.ceil(head_dim * value_bits / 8)

        centroids, boundaries = _lloyd_max_centroids(head_dim, key_bits)
        rotation = _hadamard_matrix(head_dim)
        self._k_centroids = centroids.to(device=self.device)
        self._k_boundaries = boundaries.to(device=self.device)
        self._rotation = rotation.to(device=self.device)

    def allocate(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cache_prefix = (self.num_blocks, self.block_size, self.num_heads)
        k_cache = torch.zeros(
            cache_prefix + (self.key_packed_bytes,),
            dtype=torch.uint8,
            device=self.device,
        )
        v_cache = torch.zeros(
            cache_prefix + (self.value_packed_bytes,),
            dtype=torch.uint8,
            device=self.device,
        )
        k_norms = torch.ones(cache_prefix, dtype=torch.float16, device=self.device)
        v_scales = torch.ones(cache_prefix, dtype=torch.float16, device=self.device)
        v_zeros = torch.zeros(cache_prefix, dtype=torch.float16, device=self.device)
        k_centroids = self._k_centroids.to(dtype=torch.float16).clone()
        rotation = self._rotation.to(dtype=torch.float16).clone()
        return k_cache, v_cache, k_norms, v_scales, v_zeros, k_centroids, rotation

    def store(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        *additional_tensors,
    ):
        if len(additional_tensors) < 5:
            raise ValueError(
                "store() requires k_norms, v_scales, v_zeros, k_centroids, rotation in additional_tensors"
            )
        k_norms, v_scales, v_zeros, _k_centroids, rotation = additional_tensors[:5]

        num_tokens, num_heads, head_dim = key.shape
        assert num_heads == self.num_heads and head_dim == self.head_dim
        assert value.shape == key.shape
        assert key.stride(-1) == 1 and value.stride(-1) == 1
        assert key.stride(1) == head_dim and value.stride(1) == head_dim
        assert slot_mapping.numel() == num_tokens
        assert k_cache.shape[-1] == self.key_packed_bytes
        assert v_cache.shape[-1] == self.value_packed_bytes

        store_kvcache_turboquant_kernel[(num_tokens * num_heads,)](
            key,
            key.stride(0),
            key.stride(1),
            value,
            value.stride(0),
            value.stride(1),
            k_cache.view(-1),
            v_cache.view(-1),
            k_norms.view(-1),
            v_scales.view(-1),
            v_zeros.view(-1),
            slot_mapping,
            rotation.view(-1),
            self._k_boundaries,
            self.num_heads,
            self.head_dim,
            self.key_packed_bytes,
            self.value_packed_bytes,
            self._k_boundaries.numel(),
        )

    def retrieve(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *additional_tensors,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "TurboQuantKVCache stores packed tensors only. "
            "Use the TurboQuant attention backend to consume the cache directly."
        )

    def needs_dequantization(self) -> bool:
        return True

    def get_cache_block_size_bytes(self) -> int:
        quant_bytes = self.block_size * self.num_heads * (
            self.key_packed_bytes + self.value_packed_bytes
        )
        metadata_bytes = self.block_size * self.num_heads * 6
        return quant_bytes + metadata_bytes
