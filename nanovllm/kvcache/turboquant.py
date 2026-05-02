"""TurboQuant-style packed KV cache for nano-vllm.

Based on: turboquant-pytorch/lloyd_max.py (Zandieh et al.) and vllm
"""
import math
from functools import lru_cache
from typing import Tuple

import torch
import triton
import triton.language as tl

from nanovllm.kvcache.base import BaseKVCache, KVCacheRegistry
from nanovllm.kvcache.turboquant_config import (
    TurboQuantConfig,
    packed_size_bytes,
    turboquant_config_for_preset,
)


# Lloyd-Max tables for a unit-variance normal distribution. Runtime tables are
# scaled by 1 / sqrt(head_dim), matching the normalized key component variance.
_LLOYD_MAX_UNIT_TABLES: dict[int, tuple[tuple[float, ...], tuple[float, ...]]] = {
    3: (
        (
            -2.1503495389,
            -1.3431800019,
            -0.7556454989,
            -0.2449835925,
            0.2449835925,
            0.7556454989,
            1.3431800019,
            2.1503495389,
        ),
        (
            -1.7467647704,
            -1.0494127504,
            -0.5003145457,
            0.0,
            0.5003145457,
            1.0494127504,
            1.7467647704,
        ),
    ),
    4: (
        (
            -2.7309222188,
            -2.0684471187,
            -1.6178817596,
            -1.2562575697,
            -0.9424482433,
            -0.6568799185,
            -0.3881377909,
            -0.1284276607,
            0.1284276607,
            0.3881377909,
            0.6568799185,
            0.9424482433,
            1.2562575697,
            1.6178817596,
            2.0684471187,
            2.7309222188,
        ),
        (
            -2.3996846687,
            -1.8431644391,
            -1.4370696646,
            -1.0993529065,
            -0.7996640809,
            -0.5225088547,
            -0.2582827258,
            0.0,
            0.2582827258,
            0.5225088547,
            0.7996640809,
            1.0993529065,
            1.4370696646,
            1.8431644391,
            2.3996846687,
        ),
    ),
}


def _static_lloyd_max_centroids(head_dim: int, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    if bits not in _LLOYD_MAX_UNIT_TABLES:
        raise ValueError(f"Unsupported TurboQuant key bit-width: {bits}")
    centroids, boundaries = _LLOYD_MAX_UNIT_TABLES[bits]
    scale = 1.0 / math.sqrt(head_dim)
    return (
        torch.tensor(centroids, dtype=torch.float32) * scale,
        torch.tensor(boundaries, dtype=torch.float32) * scale,
    )


def _bits_from_centroid_count(centroid_count: int) -> int:
    if centroid_count == 8:
        return 3
    if centroid_count == 16:
        return 4
    raise ValueError(
        "TurboQuant key centroids must contain 8 or 16 levels, "
        f"got {centroid_count}."
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
def _load_packed_indices(
    cache_ptr,
    base,
    d_offs,
    d_mask,
    bits: tl.constexpr,
    packed_bytes: tl.constexpr,
):
    if bits == 4:
        byte_idx = d_offs // 2
        bit_shift = (d_offs % 2) * 4
        packed = tl.load(
            cache_ptr + base + byte_idx,
            mask=d_mask & (byte_idx < packed_bytes),
            other=0,
        ).to(tl.int32)
        return ((packed >> bit_shift) & 0xF).to(tl.int32)

    group_idx = d_offs // 8
    lane = d_offs % 8
    byte_base = group_idx * 3
    b0 = tl.load(
        cache_ptr + base + byte_base,
        mask=d_mask & (byte_base < packed_bytes),
        other=0,
    ).to(tl.int32)
    b1 = tl.load(
        cache_ptr + base + byte_base + 1,
        mask=d_mask & ((byte_base + 1) < packed_bytes),
        other=0,
    ).to(tl.int32)
    b2 = tl.load(
        cache_ptr + base + byte_base + 2,
        mask=d_mask & ((byte_base + 2) < packed_bytes),
        other=0,
    ).to(tl.int32)

    idx0 = b0 & 0x7
    idx1 = (b0 >> 3) & 0x7
    idx2 = ((b0 >> 6) & 0x3) | ((b1 & 0x1) << 2)
    idx3 = (b1 >> 1) & 0x7
    idx4 = (b1 >> 4) & 0x7
    idx5 = ((b1 >> 7) & 0x1) | ((b2 & 0x3) << 1)
    idx6 = (b2 >> 2) & 0x7
    idx7 = (b2 >> 5) & 0x7

    idx = tl.where(lane == 0, idx0, idx1)
    idx = tl.where(lane == 2, idx2, idx)
    idx = tl.where(lane == 3, idx3, idx)
    idx = tl.where(lane == 4, idx4, idx)
    idx = tl.where(lane == 5, idx5, idx)
    idx = tl.where(lane == 6, idx6, idx)
    idx = tl.where(lane == 7, idx7, idx)
    return idx.to(tl.int32)


@triton.jit
def store_kvcache_turboquant_kernel(
    rotated_key_ptr,
    rotated_key_token_stride,
    rotated_key_head_stride,
    key_norm_ptr,
    key_norm_token_stride,
    key_norm_head_stride,
    value_ptr,
    value_token_stride,
    value_head_stride,
    k_cache_ptr,
    v_cache_ptr,
    k_norms_ptr,
    v_scales_ptr,
    v_zeros_ptr,
    slot_mapping_ptr,
    boundaries_ptr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    key_bits: tl.constexpr,
    value_bits: tl.constexpr,
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
    rotated_key = tl.load(
        rotated_key_ptr
        + token_idx * rotated_key_token_stride
        + head_idx * rotated_key_head_stride
        + offsets
    ).to(tl.float32)
    value = tl.load(
        value_ptr + token_idx * value_token_stride + head_idx * value_head_stride + offsets
    ).to(tl.float32)
    key_norm = tl.load(
        key_norm_ptr
        + token_idx * key_norm_token_stride
        + head_idx * key_norm_head_stride
    ).to(tl.float32)
    safe_key_norm = tl.where(key_norm > 0.0, key_norm, 1.0)

    key_indices = tl.zeros((head_dim,), dtype=tl.int32)
    for i in tl.static_range(num_boundaries):
        boundary = tl.load(boundaries_ptr + i)
        key_indices += (rotated_key > boundary).to(tl.int32)

    key_cache_base = (slot * num_heads + head_idx) * key_packed_bytes
    if key_bits == 4:
        key_pairs = tl.reshape(key_indices, (head_dim // 2, 2))
        key_lo, key_hi = tl.split(key_pairs)
        key_packed = key_lo | (key_hi << 4)
        key_pair_offs = tl.arange(0, head_dim // 2)
        tl.store(k_cache_ptr + key_cache_base + key_pair_offs, key_packed.to(tl.uint8))
    else:
        key_groups = tl.reshape(key_indices, (head_dim // 8, 8))
        k0, k1, k2, k3, k4, k5, k6, k7 = tl.split(key_groups)
        key_group_offs = tl.arange(0, head_dim // 8) * 3
        key_b0 = k0 | (k1 << 3) | ((k2 & 0x3) << 6)
        key_b1 = ((k2 >> 2) & 0x1) | (k3 << 1) | (k4 << 4) | ((k5 & 0x1) << 7)
        key_b2 = ((k5 >> 1) & 0x3) | (k6 << 2) | (k7 << 5)
        tl.store(k_cache_ptr + key_cache_base + key_group_offs, key_b0.to(tl.uint8))
        tl.store(k_cache_ptr + key_cache_base + key_group_offs + 1, key_b1.to(tl.uint8))
        tl.store(k_cache_ptr + key_cache_base + key_group_offs + 2, key_b2.to(tl.uint8))

    value_min = tl.min(value, axis=0)
    value_max = tl.max(value, axis=0)
    value_levels_max = (1 << value_bits) - 1
    value_scale = (value_max - value_min) / value_levels_max
    value_scale = tl.where(value_scale > 0.0, value_scale, 1.0)
    value_scaled = (value - value_min) / value_scale
    # Triton compatibility: avoid tl.math.llrint (missing on some builds).
    value_rounded = tl.where(
        value_scaled >= 0.0,
        tl.floor(value_scaled + 0.5),
        -tl.floor(-value_scaled + 0.5),
    )
    value_indices = tl.clamp(value_rounded, 0, value_levels_max).to(tl.int32)

    value_cache_base = (slot * num_heads + head_idx) * value_packed_bytes
    if value_bits == 4:
        value_pairs = tl.reshape(value_indices, (head_dim // 2, 2))
        value_lo, value_hi = tl.split(value_pairs)
        value_packed = value_lo | (value_hi << 4)
        value_pair_offs = tl.arange(0, head_dim // 2)
        tl.store(v_cache_ptr + value_cache_base + value_pair_offs, value_packed.to(tl.uint8))
    else:
        value_groups = tl.reshape(value_indices, (head_dim // 8, 8))
        v0, v1, v2, v3, v4, v5, v6, v7 = tl.split(value_groups)
        value_group_offs = tl.arange(0, head_dim // 8) * 3
        value_b0 = v0 | (v1 << 3) | ((v2 & 0x3) << 6)
        value_b1 = ((v2 >> 2) & 0x1) | (v3 << 1) | (v4 << 4) | ((v5 & 0x1) << 7)
        value_b2 = ((v5 >> 1) & 0x3) | (v6 << 2) | (v7 << 5)
        tl.store(v_cache_ptr + value_cache_base + value_group_offs, value_b0.to(tl.uint8))
        tl.store(v_cache_ptr + value_cache_base + value_group_offs + 1, value_b1.to(tl.uint8))
        tl.store(v_cache_ptr + value_cache_base + value_group_offs + 2, value_b2.to(tl.uint8))

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
    key_bits: tl.constexpr,
    value_bits: tl.constexpr,
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

    k_idx = _load_packed_indices(
        k_cache_ptr,
        k_base,
        d_offs,
        d_mask,
        key_bits,
        key_packed_bytes,
    )
    k_idx = tl.where(d_mask, k_idx, 0)
    centroid_mask = d_mask & (k_idx < centroid_count)
    k_vals = tl.load(k_centroids_ptr + k_idx, mask=centroid_mask, other=0.0).to(
        tl.float32
    )
    k_norm = tl.load(k_norms_ptr + meta_base).to(tl.float32)
    tl.store(
        k_out_ptr + out_base + d_offs,
        (k_vals * k_norm).to(k_out_ptr.dtype.element_ty),
        mask=d_mask,
    )

    v_idx = _load_packed_indices(
        v_cache_ptr,
        v_base,
        d_offs,
        d_mask,
        value_bits,
        value_packed_bytes,
    ).to(tl.float32)
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
    head_dim: int | None = None,
    key_bits: int | None = None,
    value_bits: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if seq_len <= 0:
        raise ValueError("TurboQuant dequantization expects a positive sequence length.")

    if key_bits is None:
        key_bits = _bits_from_centroid_count(k_centroids.numel())
    if head_dim is None:
        head_dim = (k_cache.shape[-1] * 8) // key_bits
    if value_bits is None:
        value_bits = (v_cache.shape[-1] * 8) // head_dim
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
        key_bits=key_bits,
        value_bits=value_bits,
        key_packed_bytes=k_cache.shape[-1],
        value_packed_bytes=v_cache.shape[-1],
        centroid_count=k_centroids.numel(),
        block_d=block_d,
    )
    return dense_k, dense_v


@triton.jit
def dequantize_kvcache_turboquant_batched_kernel(
    k_cache_ptr,
    v_cache_ptr,
    block_table_ptr,
    seq_lens_ptr,
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
    block_table_stride_batch,
    block_table_stride_block,
    meta_stride_block,
    meta_stride_pos,
    meta_stride_head,
    out_stride_batch,
    out_stride_token,
    out_stride_head,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    num_heads: tl.constexpr,
    key_bits: tl.constexpr,
    value_bits: tl.constexpr,
    key_packed_bytes: tl.constexpr,
    value_packed_bytes: tl.constexpr,
    centroid_count: tl.constexpr,
    block_d: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)

    seq_len = tl.load(seq_lens_ptr + batch_idx).to(tl.int32)
    if token_idx >= seq_len:
        return

    page_idx = token_idx // block_size
    page_off = token_idx % block_size
    block_num = tl.load(
        block_table_ptr
        + batch_idx * block_table_stride_batch
        + page_idx * block_table_stride_block
    ).to(tl.int64)

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
    out_base = (
        batch_idx * out_stride_batch
        + token_idx * out_stride_token
        + head_idx * out_stride_head
    )

    d_offs = tl.arange(0, block_d)
    d_mask = d_offs < head_dim

    k_idx = _load_packed_indices(
        k_cache_ptr,
        k_base,
        d_offs,
        d_mask,
        key_bits,
        key_packed_bytes,
    )
    k_idx = tl.where(d_mask, k_idx, 0)
    centroid_mask = d_mask & (k_idx < centroid_count)
    k_vals = tl.load(k_centroids_ptr + k_idx, mask=centroid_mask, other=0.0).to(tl.float32)
    k_norm = tl.load(k_norms_ptr + meta_base).to(tl.float32)
    tl.store(
        k_out_ptr + out_base + d_offs,
        (k_vals * k_norm).to(k_out_ptr.dtype.element_ty),
        mask=d_mask,
    )

    v_idx = _load_packed_indices(
        v_cache_ptr,
        v_base,
        d_offs,
        d_mask,
        value_bits,
        value_packed_bytes,
    ).to(tl.float32)
    v_scale = tl.load(v_scales_ptr + meta_base).to(tl.float32)
    v_zero = tl.load(v_zeros_ptr + meta_base).to(tl.float32)
    v_vals = v_idx * v_scale + v_zero
    tl.store(v_out_ptr + out_base + d_offs, v_vals.to(v_out_ptr.dtype.element_ty), mask=d_mask)


def dequantize_kvcache_turboquant_batched(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    k_norms: torch.Tensor,
    v_scales: torch.Tensor,
    v_zeros: torch.Tensor,
    k_centroids: torch.Tensor,
    out_dtype: torch.dtype,
    max_seq_len: int | None = None,
    out_k: torch.Tensor | None = None,
    out_v: torch.Tensor | None = None,
    head_dim: int | None = None,
    key_bits: int | None = None,
    value_bits: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if block_table.ndim != 2:
        raise ValueError("block_table must have shape [batch_size, max_num_blocks].")
    if seq_lens.ndim != 1:
        raise ValueError("seq_lens must have shape [batch_size].")
    if block_table.shape[0] != seq_lens.shape[0]:
        raise ValueError("block_table and seq_lens batch size mismatch.")

    batch_size = block_table.shape[0]
    if batch_size == 0:
        num_heads = k_cache.shape[2]
        if key_bits is None:
            key_bits = _bits_from_centroid_count(k_centroids.numel())
        if head_dim is None:
            head_dim = (k_cache.shape[-1] * 8) // key_bits
        empty = torch.empty((0, 0, num_heads, head_dim), dtype=out_dtype, device=k_cache.device)
        return empty, empty

    seq_lens = seq_lens.to(device=k_cache.device, dtype=torch.int32).contiguous()
    if max_seq_len is None:
        max_seq_len = int(seq_lens.max().item())
    if max_seq_len <= 0:
        num_heads = k_cache.shape[2]
        if key_bits is None:
            key_bits = _bits_from_centroid_count(k_centroids.numel())
        if head_dim is None:
            head_dim = (k_cache.shape[-1] * 8) // key_bits
        empty = torch.empty(
            (batch_size, 0, num_heads, head_dim),
            dtype=out_dtype,
            device=k_cache.device,
        )
        return empty, empty

    if key_bits is None:
        key_bits = _bits_from_centroid_count(k_centroids.numel())
    if head_dim is None:
        head_dim = (k_cache.shape[-1] * 8) // key_bits
    if value_bits is None:
        value_bits = (v_cache.shape[-1] * 8) // head_dim
    num_heads = k_cache.shape[2]
    block_size = k_cache.shape[1]
    block_d = triton.next_power_of_2(head_dim)
    block_table = block_table.to(device=k_cache.device, dtype=torch.int32).contiguous()

    if out_k is None or out_v is None:
        dense_k = torch.empty(
            (batch_size, max_seq_len, num_heads, head_dim),
            dtype=out_dtype,
            device=k_cache.device,
        )
        dense_v = torch.empty_like(dense_k)
    else:
        if (
            out_k.device != k_cache.device
            or out_v.device != k_cache.device
            or out_k.dtype != out_dtype
            or out_v.dtype != out_dtype
            or out_k.shape[0] < batch_size
            or out_v.shape[0] < batch_size
            or out_k.shape[1] < max_seq_len
            or out_v.shape[1] < max_seq_len
            or out_k.shape[2] != num_heads
            or out_v.shape[2] != num_heads
            or out_k.shape[3] != head_dim
            or out_v.shape[3] != head_dim
        ):
            raise ValueError(
                "Provided output buffers are incompatible with requested dequant shape."
            )
        dense_k = out_k[:batch_size, :max_seq_len]
        dense_v = out_v[:batch_size, :max_seq_len]

    grid = (max_seq_len, num_heads, batch_size)
    dequantize_kvcache_turboquant_batched_kernel[grid](
        k_cache,
        v_cache,
        block_table,
        seq_lens,
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
        block_table.stride(0),
        block_table.stride(1),
        k_norms.stride(0),
        k_norms.stride(1),
        k_norms.stride(2),
        dense_k.stride(0),
        dense_k.stride(1),
        dense_k.stride(2),
        head_dim=head_dim,
        block_size=block_size,
        num_heads=num_heads,
        key_bits=key_bits,
        value_bits=value_bits,
        key_packed_bytes=k_cache.shape[-1],
        value_packed_bytes=v_cache.shape[-1],
        centroid_count=k_centroids.numel(),
        block_d=block_d,
    )
    return dense_k, dense_v


@KVCacheRegistry.register("turboquant")
class TurboQuantKVCache(BaseKVCache):
    """
    TurboQuant-style cache with 3/4-bit packed storage.

    Keys are L2-normalized, Hadamard-rotated, then Lloyd-Max quantized.
    Values use per-vector affine quantization.
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
        turboquant_config: TurboQuantConfig | None = None,
    ):
        super().__init__(
            num_blocks=num_blocks,
            block_size=block_size,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )
        self.turboquant_config = turboquant_config or TurboQuantConfig(
            preset=f"turboquant_k{key_bits}v{value_bits}",
            key_bits=key_bits,
            value_bits=value_bits,
        )
        key_bits = self.turboquant_config.key_bits
        value_bits = self.turboquant_config.value_bits
        if key_bits not in (3, 4) or value_bits not in (3, 4):
            raise ValueError("TurboQuantKVCache supports only 3-bit and 4-bit K/V modes.")
        if head_dim < 1 or (head_dim & (head_dim - 1)) != 0:
            raise ValueError(
                "TurboQuantKVCache requires a power-of-two head_dim for Hadamard rotation"
            )

        self.key_bits = key_bits
        self.value_bits = value_bits
        self.key_packed_bytes = packed_size_bytes(head_dim, key_bits)
        self.value_packed_bytes = packed_size_bytes(head_dim, value_bits)

        centroids, boundaries = _static_lloyd_max_centroids(head_dim, key_bits)
        rotation = _hadamard_matrix(head_dim)
        self._k_centroids = centroids.to(device=self.device)
        self._k_boundaries = boundaries.to(device=self.device)
        self._rotation = rotation.to(device=self.device)

    def allocate(
        self,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
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
                "store() requires k_norms, v_scales, v_zeros, k_centroids, "
                "rotation in additional_tensors"
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

        # Use batched GEMM for Hadamard rotation instead of loading a D x D
        # matrix per token/head inside the Triton store kernel.
        key_fp32 = key.to(torch.float32)
        key_norm_input = torch.linalg.norm(key_fp32, dim=-1)
        safe_norm = torch.clamp_min(key_norm_input, 1e-12).unsqueeze(-1)
        key_unit = key_fp32 / safe_norm
        rotation_fp32 = rotation.to(device=key.device, dtype=torch.float32)
        rotated_key = torch.matmul(key_unit, rotation_fp32.t()).contiguous()
        key_norm_input = key_norm_input.contiguous()

        store_kvcache_turboquant_kernel[(num_tokens * num_heads,)](
            rotated_key,
            rotated_key.stride(0),
            rotated_key.stride(1),
            key_norm_input,
            key_norm_input.stride(0),
            key_norm_input.stride(1),
            value,
            value.stride(0),
            value.stride(1),
            k_cache.view(-1),
            v_cache.view(-1),
            k_norms.view(-1),
            v_scales.view(-1),
            v_zeros.view(-1),
            slot_mapping,
            self._k_boundaries,
            self.num_heads,
            self.head_dim,
            self.key_bits,
            self.value_bits,
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


@KVCacheRegistry.register("turboquant_k4v4")
class TurboQuantK4V4KVCache(TurboQuantKVCache):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault(
            "turboquant_config", turboquant_config_for_preset("turboquant_k4v4")
        )
        super().__init__(*args, **kwargs)


@KVCacheRegistry.register("turboquant_k3v4")
class TurboQuantK3V4KVCache(TurboQuantKVCache):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault(
            "turboquant_config", turboquant_config_for_preset("turboquant_k3v4")
        )
        super().__init__(*args, **kwargs)


@KVCacheRegistry.register("turboquant_k3v3")
class TurboQuantK3V3KVCache(TurboQuantKVCache):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault(
            "turboquant_config", turboquant_config_for_preset("turboquant_k3v3")
        )
        super().__init__(*args, **kwargs)
