"""TurboQuant attention backend.

This backend follows the same high-level split as upstream vLLM:
- first-chunk prefill uses normal FlashInfer attention on raw Q/K/V
- decode reads the packed TurboQuant cache directly
- continuation prefills can use a decode-style path for small chunks
- larger prefix-cache prefills fall back to dense masked attention
"""

from typing import Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from nanovllm.kvcache.turboquant import (
    dequantize_kvcache_turboquant,
    dequantize_kvcache_turboquant_batched,
)
from nanovllm.layers.flash_attn_backend import (
    BaseFlashAttentionBackend,
    FlashAttentionRegistry,
)
from nanovllm.layers.attn_utils import normalize_decode_query
from nanovllm.layers.flashinfer_flash_attn import FlashInferAttention


_CONTINUATION_DECODE_THRESHOLD = 256


@triton.jit
def _continuation_decode_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    q_stride_token,
    q_stride_head,
    k_stride_token,
    k_stride_head,
    v_stride_token,
    v_stride_head,
    out_stride_token,
    out_stride_head,
    cached_len,
    seq_len,
    kv_group_size,
    softmax_scale,
    head_dim: tl.constexpr,
    block_d: tl.constexpr,
    block_k: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    kv_head = head_idx // kv_group_size
    visible_len = cached_len + token_idx + 1

    d_offs = tl.arange(0, block_d)
    d_mask = d_offs < head_dim
    q_base = token_idx * q_stride_token + head_idx * q_stride_head
    q = tl.load(q_ptr + q_base + d_offs, mask=d_mask, other=0.0).to(tl.float32)

    m_prev = -float("inf")
    l_prev = 0.0
    acc = tl.zeros([block_d], dtype=tl.float32)

    for start_n in tl.range(0, seq_len, block_k):
        kv_idx = start_n + tl.arange(0, block_k)
        kv_mask = kv_idx < visible_len

        k_ptrs = k_ptr + kv_idx[:, None] * k_stride_token + kv_head * k_stride_head + d_offs[None, :]
        k = tl.load(k_ptrs, mask=kv_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        scores = tl.sum(q[None, :] * k, axis=1) * softmax_scale
        scores = tl.where(kv_mask, scores, -float("inf"))

        m_curr = tl.max(scores, axis=0)
        m_next = tl.maximum(m_prev, m_curr)
        alpha = tl.exp(m_prev - m_next)
        probs = tl.exp(scores - m_next)

        v_ptrs = v_ptr + kv_idx[:, None] * v_stride_token + kv_head * v_stride_head + d_offs[None, :]
        v = tl.load(v_ptrs, mask=kv_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        acc = acc * alpha + tl.sum(probs[:, None] * v, axis=0)
        l_prev = l_prev * alpha + tl.sum(probs, axis=0)
        m_prev = m_next

    out = acc / l_prev
    out_base = token_idx * out_stride_token + head_idx * out_stride_head
    tl.store(out_ptr + out_base + d_offs, out.to(out_ptr.dtype.element_ty), mask=d_mask)


@triton.jit
def _packed_continuation_decode_attention_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_norms_ptr,
    v_scales_ptr,
    v_zeros_ptr,
    k_centroids_ptr,
    block_table_ptr,
    out_ptr,
    q_stride_token,
    q_stride_head,
    k_cache_stride_block,
    k_cache_stride_pos,
    k_cache_stride_head,
    v_cache_stride_block,
    v_cache_stride_pos,
    v_cache_stride_head,
    meta_stride_block,
    meta_stride_pos,
    meta_stride_head,
    block_table_stride,
    out_stride_token,
    out_stride_head,
    cached_len,
    seq_len,
    kv_group_size,
    softmax_scale,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    key_packed_bytes: tl.constexpr,
    value_packed_bytes: tl.constexpr,
    centroid_count: tl.constexpr,
    block_d: tl.constexpr,
    block_k: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    kv_head = head_idx // kv_group_size
    visible_len = cached_len + token_idx + 1

    d_offs = tl.arange(0, block_d)
    d_mask = d_offs < head_dim
    q_base = token_idx * q_stride_token + head_idx * q_stride_head
    q = tl.load(q_ptr + q_base + d_offs, mask=d_mask, other=0.0).to(tl.float32)

    byte_idx = d_offs // 2
    bit_shift = (d_offs % 2) * 4

    m_prev = -float("inf")
    l_prev = 0.0
    acc = tl.zeros([block_d], dtype=tl.float32)

    for start_n in tl.range(0, seq_len, block_k):
        kv_idx = start_n + tl.arange(0, block_k)
        kv_mask = kv_idx < visible_len

        page_idx = kv_idx // block_size
        page_off = kv_idx % block_size
        block_num = tl.load(block_table_ptr + page_idx * block_table_stride, mask=kv_mask, other=0).to(tl.int64)

        k_base = (
            block_num * k_cache_stride_block
            + page_off.to(tl.int64) * k_cache_stride_pos
            + tl.cast(kv_head, tl.int64) * k_cache_stride_head
        )
        meta_base = (
            block_num * meta_stride_block
            + page_off.to(tl.int64) * meta_stride_pos
            + tl.cast(kv_head, tl.int64) * meta_stride_head
        )

        k_packed = tl.load(
            k_cache_ptr + k_base[:, None] + byte_idx[None, :],
            mask=kv_mask[:, None] & d_mask[None, :] & (byte_idx[None, :] < key_packed_bytes),
            other=0,
        ).to(tl.int32)
        k_idx = ((k_packed >> bit_shift[None, :]) & 0xF).to(tl.int32)
        k_idx = tl.where(d_mask[None, :], k_idx, 0)
        centroid_mask = kv_mask[:, None] & d_mask[None, :] & (k_idx < centroid_count)
        k_vals = tl.load(k_centroids_ptr + k_idx, mask=centroid_mask, other=0.0).to(tl.float32)
        k_norm = tl.load(k_norms_ptr + meta_base, mask=kv_mask, other=1.0).to(tl.float32)
        k = k_vals * k_norm[:, None]

        scores = tl.sum(q[None, :] * k, axis=1) * softmax_scale
        scores = tl.where(kv_mask, scores, -float("inf"))

        m_curr = tl.max(scores, axis=0)
        m_next = tl.maximum(m_prev, m_curr)
        alpha = tl.exp(m_prev - m_next)
        probs = tl.exp(scores - m_next)

        v_base = (
            block_num * v_cache_stride_block
            + page_off.to(tl.int64) * v_cache_stride_pos
            + tl.cast(kv_head, tl.int64) * v_cache_stride_head
        )
        v_packed = tl.load(
            v_cache_ptr + v_base[:, None] + byte_idx[None, :],
            mask=kv_mask[:, None] & d_mask[None, :] & (byte_idx[None, :] < value_packed_bytes),
            other=0,
        ).to(tl.int32)
        v_idx = ((v_packed >> bit_shift[None, :]) & 0xF).to(tl.float32)
        v_scale = tl.load(v_scales_ptr + meta_base, mask=kv_mask, other=1.0).to(tl.float32)
        v_zero = tl.load(v_zeros_ptr + meta_base, mask=kv_mask, other=0.0).to(tl.float32)
        v = v_idx * v_scale[:, None] + v_zero[:, None]

        acc = acc * alpha + tl.sum(probs[:, None] * v, axis=0)
        l_prev = l_prev * alpha + tl.sum(probs, axis=0)
        m_prev = m_next

    out = acc / l_prev
    out_base = token_idx * out_stride_token + head_idx * out_stride_head
    tl.store(out_ptr + out_base + d_offs, out.to(out_ptr.dtype.element_ty), mask=d_mask)


@FlashAttentionRegistry.register("turboquant")
class TurboQuantFlashAttention(BaseFlashAttentionBackend):
    """Attention backend for packed TurboQuant KV cache."""

    def __init__(self):
        self._dense_prefill = FlashInferAttention()
        self._decode_workspace: dict[tuple[int, torch.dtype, int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self._prefill_arange_cache: dict[tuple[int, torch.dtype], torch.Tensor] = {}
        self._prefill_mask_cache: dict[tuple[int, torch.device], torch.Tensor] = {}

    @property
    def supports_quantized_cache_inputs(self) -> bool:
        return True

    def _validate_quant_tensors(self, additional_cache_tensors) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if len(additional_cache_tensors) < 5:
            raise ValueError(
                "TurboQuant attention requires k_norms, v_scales, v_zeros, "
                "k_centroids, and rotation in additional_cache_tensors."
            )
        k_norms, v_scales, v_zeros, k_centroids, rotation = additional_cache_tensors[:5]
        return k_norms, v_scales, v_zeros, k_centroids, rotation

    def _dequantize_sequence(
        self,
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
        dense_k_rot, dense_v = dequantize_kvcache_turboquant(
            k_cache,
            v_cache,
            seq_len,
            block_table_row,
            k_norms,
            v_scales,
            v_zeros,
            k_centroids,
            out_dtype,
        )
        return dense_k_rot, dense_v

    def _rotate_query_for_turboquant(
        self,
        q: torch.Tensor,
        rotation: torch.Tensor,
    ) -> torch.Tensor:
        # Keys in cache are stored in rotated space (k_rot = k @ R^T for row vectors).
        # Instead of reconstructing unrotated keys (k = k_rot @ R), rotate the query once
        # and compute attention in rotated space: q·k == (q @ R)·k_rot.
        return torch.matmul(q, rotation.to(device=q.device, dtype=q.dtype))

    def _get_decode_workspace(
        self,
        batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (device.index or 0, dtype, num_kv_heads, head_dim)
        cached = self._decode_workspace.get(key)
        if (
            cached is None
            or cached[0].shape[0] < batch_size
            or cached[0].shape[1] < max_seq_len
        ):
            shape = (batch_size, max_seq_len, num_kv_heads, head_dim)
            k_buf = torch.empty(shape, dtype=dtype, device=device)
            v_buf = torch.empty_like(k_buf)
            self._decode_workspace[key] = (k_buf, v_buf)
        return self._decode_workspace[key]

    def _get_arange_cache(
        self,
        size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (device.index or 0, dtype)
        cached = self._prefill_arange_cache.get(key)
        if cached is None or cached.shape[0] <= size:
            cached = torch.arange(size + 1, device=device, dtype=dtype)
            self._prefill_arange_cache[key] = cached
        return cached

    def _get_prefill_mask_buffer(
        self,
        q_len: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        key = (device.index or 0, device)
        cached = self._prefill_mask_cache.get(key)
        if cached is None or cached.shape[0] < q_len or cached.shape[1] < seq_len:
            cached = torch.empty((q_len, seq_len), dtype=torch.bool, device=device)
            self._prefill_mask_cache[key] = cached
        return cached[:q_len, :seq_len]

    def _run_decode_attention(
        self,
        q_rot: torch.Tensor,
        dense_k_rot: torch.Tensor,
        dense_v: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        q_t = q_rot.unsqueeze(0).unsqueeze(2)
        k_t = dense_k_rot.transpose(0, 1).unsqueeze(0)
        v_t = dense_v.transpose(0, 1).unsqueeze(0)
        out = F.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            scale=scale,
            enable_gqa=(dense_k_rot.shape[1] < q_rot.shape[0]),
        )
        return out.squeeze(0).squeeze(1)

    def _run_prefill_attention(
        self,
        q_rot: torch.Tensor,
        dense_k_rot: torch.Tensor,
        dense_v: torch.Tensor,
        cached_len: int,
        scale: float,
        q_positions: Optional[torch.Tensor] = None,
        k_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q_len = q_rot.shape[0]
        seq_len = dense_k_rot.shape[0]
        q_t = q_rot.transpose(0, 1).unsqueeze(0)
        k_t = dense_k_rot.transpose(0, 1).unsqueeze(0)
        v_t = dense_v.transpose(0, 1).unsqueeze(0)
        if q_positions is None or k_positions is None:
            q_pos = torch.arange(q_len, device=q_rot.device).unsqueeze(1) + cached_len
            k_pos = torch.arange(seq_len, device=q_rot.device).unsqueeze(0)
            mask = k_pos <= q_pos
        else:
            q_pos = q_positions[:q_len] + cached_len
            k_pos = k_positions[:seq_len]
            mask = self._get_prefill_mask_buffer(q_len, seq_len, q_rot.device)
            torch.le(k_pos.unsqueeze(0), q_pos.unsqueeze(1), out=mask)
        out = F.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            attn_mask=mask,
            scale=scale,
            enable_gqa=(dense_k_rot.shape[1] < q_rot.shape[1]),
        )
        return out.squeeze(0).transpose(0, 1)

    def _run_small_continuation_prefill_packed(
        self,
        q_rot: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table_row: torch.Tensor,
        k_norms: torch.Tensor,
        v_scales: torch.Tensor,
        v_zeros: torch.Tensor,
        k_centroids: torch.Tensor,
        cached_len: int,
        seq_len: int,
        scale: float,
    ) -> torch.Tensor:
        output = torch.empty_like(q_rot)
        block_d = triton.next_power_of_2(q_rot.shape[-1])
        block_k = 32
        kv_group_size = q_rot.shape[1] // k_cache.shape[2]
        _packed_continuation_decode_attention_kernel[(q_rot.shape[0], q_rot.shape[1])](
            q_rot,
            k_cache,
            v_cache,
            k_norms,
            v_scales,
            v_zeros,
            k_centroids,
            block_table_row,
            output,
            q_rot.stride(0),
            q_rot.stride(1),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            k_norms.stride(0),
            k_norms.stride(1),
            k_norms.stride(2),
            block_table_row.stride(0),
            output.stride(0),
            output.stride(1),
            cached_len,
            seq_len,
            kv_group_size,
            scale,
            head_dim=q_rot.shape[-1],
            block_size=k_cache.shape[1],
            key_packed_bytes=k_cache.shape[-1],
            value_packed_bytes=v_cache.shape[-1],
            centroid_count=k_centroids.numel(),
            block_d=block_d,
            block_k=block_k,
        )
        return output

    def _run_small_continuation_prefill(
        self,
        q_rot: torch.Tensor,
        dense_k_rot: torch.Tensor,
        dense_v: torch.Tensor,
        cached_len: int,
        scale: float,
    ) -> torch.Tensor:
        output = torch.empty_like(q_rot)
        block_d = triton.next_power_of_2(q_rot.shape[-1])
        block_k = 32
        kv_group_size = q_rot.shape[1] // dense_k_rot.shape[1]
        grid = (q_rot.shape[0], q_rot.shape[1])
        _continuation_decode_attention_kernel[grid](
            q_rot,
            dense_k_rot,
            dense_v,
            output,
            q_rot.stride(0),
            q_rot.stride(1),
            dense_k_rot.stride(0),
            dense_k_rot.stride(1),
            dense_v.stride(0),
            dense_v.stride(1),
            output.stride(0),
            output.stride(1),
            cached_len,
            dense_k_rot.shape[0],
            kv_group_size,
            scale,
            head_dim=q_rot.shape[-1],
            block_d=block_d,
            block_k=block_k,
        )
        return output

    def _prefill_with_prefix_cache(
        self,
        q: torch.Tensor,
        scale: float,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        block_table: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        additional_cache_tensors,
    ) -> torch.Tensor:
        k_norms, v_scales, v_zeros, k_centroids, rotation = self._validate_quant_tensors(
            additional_cache_tensors
        )
        output = torch.empty_like(q)
        q_starts = cu_seqlens_q.to(torch.int64).tolist()
        k_starts = cu_seqlens_k.to(torch.int64).tolist()
        max_q_len = 0
        max_seq_len = 0
        for req_idx in range(block_table.shape[0]):
            q_len = q_starts[req_idx + 1] - q_starts[req_idx]
            seq_len = k_starts[req_idx + 1] - k_starts[req_idx]
            if q_len > max_q_len:
                max_q_len = q_len
            if seq_len > max_seq_len:
                max_seq_len = seq_len
        q_positions = self._get_arange_cache(max_q_len, q.device, torch.int64)
        k_positions = self._get_arange_cache(max_seq_len, q.device, torch.int64)

        for req_idx in range(block_table.shape[0]):
            q_start = q_starts[req_idx]
            q_end = q_starts[req_idx + 1]
            q_len = q_end - q_start
            seq_len = k_starts[req_idx + 1] - k_starts[req_idx]
            if q_len <= 0 or seq_len <= 0:
                continue
            cached_len = seq_len - q_len
            q_req = q[q_start:q_end]
            q_req_rot = self._rotate_query_for_turboquant(q_req, rotation)
            if 0 < cached_len and q_len <= _CONTINUATION_DECODE_THRESHOLD:
                output[q_start:q_end] = self._run_small_continuation_prefill_packed(
                    q_req_rot,
                    k_cache,
                    v_cache,
                    block_table[req_idx],
                    k_norms,
                    v_scales,
                    v_zeros,
                    k_centroids,
                    cached_len,
                    seq_len,
                    scale,
                )
            else:
                dense_k_rot, dense_v = self._dequantize_sequence(
                    k_cache,
                    v_cache,
                    seq_len,
                    block_table[req_idx],
                    k_norms,
                    v_scales,
                    v_zeros,
                    k_centroids,
                    q.dtype,
                )
                if q_len <= _CONTINUATION_DECODE_THRESHOLD:
                    output[q_start:q_end] = self._run_small_continuation_prefill(
                        q_req_rot,
                        dense_k_rot,
                        dense_v,
                        cached_len,
                        scale,
                    )
                    continue
                output[q_start:q_end] = self._run_prefill_attention(
                    q_req_rot,
                    dense_k_rot,
                    dense_v,
                    cached_len,
                    scale,
                    q_positions=q_positions,
                    k_positions=k_positions,
                )
        return output

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
        if (
            cu_seqlens_q is None
            or cu_seqlens_k is None
            or max_seqlen_q is None
            or max_seqlen_k is None
        ):
            raise ValueError("TurboQuant prefill requires varlen metadata.")

        if block_table is None:
            return self._dense_prefill.prefill(
                q,
                k,
                v,
                scale=scale,
                max_seqlen_q=max_seqlen_q,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_k=max_seqlen_k,
                cu_seqlens_k=cu_seqlens_k,
            )

        return self._prefill_with_prefix_cache(
            q,
            scale,
            cu_seqlens_q,
            cu_seqlens_k,
            block_table,
            k,
            v,
            additional_cache_tensors,
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
        if cache_seqlens is None or block_table is None:
            raise ValueError(
                "TurboQuant decode requires cache sequence lengths and block tables."
            )

        k_norms, v_scales, v_zeros, k_centroids, rotation = self._validate_quant_tensors(
            additional_cache_tensors
        )
        q = normalize_decode_query(q)
        batch_size = q.shape[0]
        if batch_size == 0:
            return q.unsqueeze(1)

        seq_lens = cache_seqlens.to(device=q.device, dtype=torch.int32).contiguous()
        max_seq_len = int(seq_lens.max().item())
        if max_seq_len <= 0:
            return torch.zeros_like(q).unsqueeze(1)

        num_kv_heads = k_cache.shape[2]
        head_dim = q.shape[-1]
        k_buf, v_buf = self._get_decode_workspace(
            batch_size,
            max_seq_len,
            num_kv_heads,
            head_dim,
            q.dtype,
            q.device,
        )
        dense_k_rot, dense_v = dequantize_kvcache_turboquant_batched(
            k_cache,
            v_cache,
            block_table,
            seq_lens,
            k_norms,
            v_scales,
            v_zeros,
            k_centroids,
            q.dtype,
            max_seq_len=max_seq_len,
            out_k=k_buf,
            out_v=v_buf,
        )
        q_rot = self._rotate_query_for_turboquant(q, rotation)

        # Decode query length is always 1. We mask KV positions > seq_len for each request.
        q_t = q_rot.unsqueeze(2)
        k_t = dense_k_rot.transpose(1, 2)
        v_t = dense_v.transpose(1, 2)
        kv_pos = torch.arange(max_seq_len, device=q.device, dtype=seq_lens.dtype).unsqueeze(0)
        attn_mask = kv_pos < seq_lens.unsqueeze(1)
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)

        out = F.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            attn_mask=attn_mask,
            scale=scale,
            enable_gqa=(num_kv_heads < q.shape[1]),
        )
        return out.squeeze(2).unsqueeze(1)
