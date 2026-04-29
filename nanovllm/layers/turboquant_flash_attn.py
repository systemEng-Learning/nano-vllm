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

from nanovllm.kvcache.turboquant import dequantize_kvcache_turboquant
from nanovllm.layers.flash_attn_backend import (
    BaseFlashAttentionBackend,
    FlashAttentionRegistry,
)
from nanovllm.layers.attn_utils import normalize_decode_query
from nanovllm.layers.flashinfer_flash_attn import FlashInferAttention


_CONTINUATION_DECODE_THRESHOLD = 64


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


@FlashAttentionRegistry.register("turboquant")
class TurboQuantFlashAttention(BaseFlashAttentionBackend):
    """Attention backend for packed TurboQuant KV cache."""

    def __init__(self):
        self._dense_prefill = FlashInferAttention()

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
        rotation: torch.Tensor,
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
        dense_k = torch.matmul(dense_k_rot.to(torch.float32), rotation.to(torch.float32)).to(
            out_dtype
        )
        return dense_k, dense_v

    def _run_decode_attention(
        self,
        q: torch.Tensor,
        dense_k: torch.Tensor,
        dense_v: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        q_t = q.unsqueeze(0).unsqueeze(2)
        k_t = dense_k.transpose(0, 1).unsqueeze(0)
        v_t = dense_v.transpose(0, 1).unsqueeze(0)
        out = F.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            scale=scale,
            enable_gqa=(dense_k.shape[1] < q.shape[0]),
        )
        return out.squeeze(0).squeeze(1).transpose(0, 1)

    def _run_prefill_attention(
        self,
        q: torch.Tensor,
        dense_k: torch.Tensor,
        dense_v: torch.Tensor,
        cached_len: int,
        scale: float,
    ) -> torch.Tensor:
        q_len = q.shape[0]
        seq_len = dense_k.shape[0]
        q_t = q.transpose(0, 1).unsqueeze(0)
        k_t = dense_k.transpose(0, 1).unsqueeze(0)
        v_t = dense_v.transpose(0, 1).unsqueeze(0)
        q_pos = torch.arange(q_len, device=q.device).unsqueeze(1) + cached_len
        k_pos = torch.arange(seq_len, device=q.device).unsqueeze(0)
        mask = k_pos <= q_pos
        out = F.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            attn_mask=mask,
            scale=scale,
            enable_gqa=(dense_k.shape[1] < q.shape[1]),
        )
        return out.squeeze(0).transpose(0, 1)

    def _run_small_continuation_prefill(
        self,
        q: torch.Tensor,
        dense_k: torch.Tensor,
        dense_v: torch.Tensor,
        cached_len: int,
        scale: float,
    ) -> torch.Tensor:
        output = torch.empty_like(q)
        block_d = triton.next_power_of_2(q.shape[-1])
        block_k = 32
        kv_group_size = q.shape[1] // dense_k.shape[1]
        grid = (q.shape[0], q.shape[1])
        _continuation_decode_attention_kernel[grid](
            q,
            dense_k,
            dense_v,
            output,
            q.stride(0),
            q.stride(1),
            dense_k.stride(0),
            dense_k.stride(1),
            dense_v.stride(0),
            dense_v.stride(1),
            output.stride(0),
            output.stride(1),
            cached_len,
            dense_k.shape[0],
            kv_group_size,
            scale,
            head_dim=q.shape[-1],
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

        for req_idx in range(block_table.shape[0]):
            q_start = q_starts[req_idx]
            q_end = q_starts[req_idx + 1]
            q_len = q_end - q_start
            seq_len = k_starts[req_idx + 1] - k_starts[req_idx]
            if q_len <= 0 or seq_len <= 0:
                continue
            cached_len = seq_len - q_len
            dense_k, dense_v = self._dequantize_sequence(
                k_cache,
                v_cache,
                seq_len,
                block_table[req_idx],
                k_norms,
                v_scales,
                v_zeros,
                k_centroids,
                rotation,
                q.dtype,
            )
            q_req = q[q_start:q_end]
            if q_len <= _CONTINUATION_DECODE_THRESHOLD:
                output[q_start:q_end] = self._run_small_continuation_prefill(
                    q_req,
                    dense_k,
                    dense_v,
                    cached_len,
                    scale,
                )
            else:
                output[q_start:q_end] = self._run_prefill_attention(
                    q_req,
                    dense_k,
                    dense_v,
                    cached_len,
                    scale,
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
        output = torch.empty_like(q)

        for req_idx in range(batch_size):
            seq_len = int(cache_seqlens[req_idx].item())
            dense_k, dense_v = self._dequantize_sequence(
                k_cache,
                v_cache,
                seq_len,
                block_table[req_idx],
                k_norms,
                v_scales,
                v_zeros,
                k_centroids,
                rotation,
                q.dtype,
            )
            output[req_idx] = self._run_decode_attention(
                q[req_idx],
                dense_k,
                dense_v,
                scale,
            )
        return output.unsqueeze(1)
