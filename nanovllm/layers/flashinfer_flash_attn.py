"""FlashInfer-based attention backends for nano-vllm."""

from typing import Optional

import torch

from nanovllm.layers.flash_attn_backend import BaseFlashAttentionBackend
from nanovllm.layers.attn_utils import normalize_decode_query


_WORKSPACE_BYTES = 128 * 1024 * 1024


def _load_flashinfer():
    try:
        import flashinfer
    except ImportError as exc:
        raise ImportError(
            "FlashInfer attention backend requires `flashinfer-python` to be installed."
        ) from exc
    return flashinfer


def _validate_flashinfer_cache_dtype(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
) -> None:
    supported = (torch.float16, torch.bfloat16)
    if k_cache.dtype not in supported or v_cache.dtype not in supported:
        raise NotImplementedError(
            "FlashInfer backend currently expects FP16/BF16 KV cache tensors."
        )


def _build_page_metadata(
    block_table: torch.Tensor,
    seqlens: torch.Tensor,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_lens = seqlens.to(device=block_table.device, dtype=torch.int32).contiguous()
    num_pages = torch.div(seq_lens + page_size - 1, page_size, rounding_mode="floor")

    paged_kv_indptr = torch.zeros(
        seq_lens.numel() + 1,
        dtype=torch.int32,
        device=block_table.device,
    )
    paged_kv_indptr[1:] = torch.cumsum(num_pages, dim=0)

    if int(num_pages.sum().item()) > 0:
        max_pages = block_table.shape[1]
        page_positions = torch.arange(
            max_pages,
            dtype=torch.int32,
            device=block_table.device,
        ).unsqueeze(0)
        page_mask = page_positions < num_pages.unsqueeze(1)
        paged_kv_indices = block_table.to(torch.int32).masked_select(page_mask).contiguous()
    else:
        paged_kv_indices = torch.empty(0, dtype=torch.int32, device=block_table.device)

    paged_kv_last_page_len = torch.where(
        num_pages > 0,
        torch.remainder(seq_lens - 1, page_size) + 1,
        torch.ones_like(seq_lens),
    ).contiguous()

    return seq_lens, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len


class FlashInferAttention(BaseFlashAttentionBackend):
    """FlashInfer implementation for dense prefill and paged KV-cache decode."""

    def __init__(self):
        self._workspace_buffers: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
        self._ragged_prefill_wrappers: dict[tuple[int, torch.device], object] = {}
        self._paged_prefill_wrappers: dict[tuple[int, torch.device], object] = {}
        self._decode_wrappers: dict[tuple[int, torch.device, bool], object] = {}

    def _get_workspace(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (device, dtype)
        workspace = self._workspace_buffers.get(key)
        if workspace is None:
            workspace = torch.empty(
                _WORKSPACE_BYTES,
                dtype=torch.uint8,
                device=device,
            )
            self._workspace_buffers[key] = workspace
        return workspace

    def _get_ragged_prefill_wrapper(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ):
        key = (device.index or 0, device)
        wrapper = self._ragged_prefill_wrappers.get(key)
        if wrapper is None:
            flashinfer = _load_flashinfer()
            wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
                self._get_workspace(device, dtype),
                "NHD",
            )
            self._ragged_prefill_wrappers[key] = wrapper
        return wrapper

    def _get_paged_prefill_wrapper(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ):
        key = (device.index or 0, device)
        wrapper = self._paged_prefill_wrappers.get(key)
        if wrapper is None:
            flashinfer = _load_flashinfer()
            wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
                self._get_workspace(device, dtype),
                "NHD",
            )
            self._paged_prefill_wrappers[key] = wrapper
        return wrapper

    def _get_decode_wrapper(
        self,
        device: torch.device,
        dtype: torch.dtype,
        use_tensor_cores: bool,
    ):
        key = (device.index or 0, device, use_tensor_cores)
        wrapper = self._decode_wrappers.get(key)
        if wrapper is None:
            flashinfer = _load_flashinfer()
            wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
                self._get_workspace(device, dtype),
                "NHD",
                use_tensor_cores=use_tensor_cores,
            )
            self._decode_wrappers[key] = wrapper
        return wrapper

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
            raise ValueError("FlashInferAttention prefill requires varlen metadata.")

        if additional_cache_tensors:
            raise NotImplementedError(
                "FlashInferAttention does not yet support custom quantized prefix-cache metadata."
            )

        if block_table is None:
            wrapper = self._get_ragged_prefill_wrapper(q.device, q.dtype)
            wrapper.plan(
                cu_seqlens_q.to(dtype=torch.int32).contiguous(),
                cu_seqlens_k.to(dtype=torch.int32).contiguous(),
                q.shape[1],
                k.shape[1],
                q.shape[2],
                causal=True,
                sm_scale=scale,
                q_data_type=q.dtype,
                kv_data_type=k.dtype,
            )
            return wrapper.run(q, k, v)

        _validate_flashinfer_cache_dtype(k, v)
        seq_lens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
        page_size = k.shape[1]
        seq_lens, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len = (
            _build_page_metadata(block_table, seq_lens, page_size)
        )
        wrapper = self._get_paged_prefill_wrapper(q.device, q.dtype)
        wrapper.plan(
            cu_seqlens_q.to(dtype=torch.int32).contiguous(),
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            q.shape[1],
            k.shape[2],
            q.shape[2],
            page_size,
            causal=True,
            sm_scale=scale,
            q_data_type=q.dtype,
            kv_data_type=k.dtype,
            seq_lens=seq_lens,
            seq_lens_q=(cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(torch.int32).contiguous(),
            block_tables=block_table.to(torch.int32).contiguous(),
            max_token_per_sequence=max_seqlen_q,
            max_sequence_kv=max_seqlen_k,
        )
        return wrapper.run(q, (k, v))

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
                "FlashInferAttention decode requires cache sequence lengths and block tables."
            )
        if additional_cache_tensors:
            raise NotImplementedError(
                "FlashInferAttention does not yet support custom quantized decode metadata."
            )

        _validate_flashinfer_cache_dtype(k_cache, v_cache)
        q = normalize_decode_query(q)
        page_size = k_cache.shape[1]
        seq_lens, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len = (
            _build_page_metadata(block_table, cache_seqlens, page_size)
        )
        batch_size = q.shape[0]
        use_tensor_cores = q.shape[1] != k_cache.shape[2]
        wrapper = self._get_decode_wrapper(q.device, q.dtype, use_tensor_cores)
        wrapper.plan(
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            q.shape[1],
            k_cache.shape[2],
            q.shape[2],
            page_size,
            q_data_type=q.dtype,
            data_type=k_cache.dtype,
            sm_scale=scale,
            block_tables=block_table.to(torch.int32).contiguous(),
            seq_lens=seq_lens,
        )
        out = wrapper.run(q, (k_cache, v_cache))
        if out.shape[0] != batch_size:
            raise RuntimeError(
                "FlashInfer decode returned an unexpected batch dimension."
            )
        return out.unsqueeze(1)
