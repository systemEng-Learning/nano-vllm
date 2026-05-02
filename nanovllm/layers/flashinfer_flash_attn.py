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

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        # CUDA graph capture cannot do host syncs or create value-dependent
        # shapes. Use the fixed block-table capacity; FlashInfer reads the
        # active prefix from paged_kv_indptr.
        paged_kv_indices = block_table.to(torch.int32).contiguous().reshape(-1)
    elif int(num_pages.sum().item()) > 0:
        max_pages = block_table.shape[1]
        page_positions = torch.arange(
            max_pages,
            dtype=torch.int32,
            device=block_table.device,
        ).unsqueeze(0)
        page_mask = page_positions < num_pages.unsqueeze(1)
        paged_kv_indices = (
            block_table.to(torch.int32).masked_select(page_mask).contiguous()
        )
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
        self._decode_graph_wrappers: dict[
            tuple[int, torch.device, bool, int, int],
            object,
        ] = {}
        self._decode_graph_buffers: dict[
            tuple[int, torch.device, bool, int, int],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}
        self._decode_graph_params: dict[int, tuple] = {}
        self._capture_batch_size: int | None = None
        self._capture_max_num_blocks: int | None = None

    @property
    def supports_cudagraph_capture(self) -> bool:
        return True

    def begin_cudagraph_capture(self, batch_size: int, max_num_blocks: int) -> None:
        self._capture_batch_size = batch_size
        self._capture_max_num_blocks = max_num_blocks

    def end_cudagraph_capture(self) -> None:
        self._capture_batch_size = None
        self._capture_max_num_blocks = None

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

    def _get_decode_graph_wrapper(
        self,
        device: torch.device,
        dtype: torch.dtype,
        use_tensor_cores: bool,
        batch_size: int,
        max_num_blocks: int,
    ):
        key = (device.index or 0, device, use_tensor_cores, batch_size, max_num_blocks)
        wrapper = self._decode_graph_wrappers.get(key)
        if wrapper is None:
            flashinfer = _load_flashinfer()
            paged_kv_indptr = torch.empty(
                batch_size + 1,
                dtype=torch.int32,
                device=device,
            )
            paged_kv_indices = torch.empty(
                batch_size * max_num_blocks,
                dtype=torch.int32,
                device=device,
            )
            paged_kv_last_page_len = torch.empty(
                batch_size,
                dtype=torch.int32,
                device=device,
            )
            wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
                self._get_workspace(device, dtype),
                "NHD",
                use_cuda_graph=True,
                use_tensor_cores=use_tensor_cores,
                paged_kv_indptr_buffer=paged_kv_indptr,
                paged_kv_indices_buffer=paged_kv_indices,
                paged_kv_last_page_len_buffer=paged_kv_last_page_len,
            )
            self._decode_graph_wrappers[key] = wrapper
            self._decode_graph_buffers[key] = (
                paged_kv_indptr,
                paged_kv_indices,
                paged_kv_last_page_len,
            )
        return wrapper

    def _plan_decode_wrapper(
        self,
        wrapper,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        page_size: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
        scale: float,
        disable_split_kv: bool = False,
    ) -> None:
        seq_lens, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len = (
            _build_page_metadata(block_table, cache_seqlens, page_size)
        )
        wrapper.plan(
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            q_data_type=q_dtype,
            data_type=kv_dtype,
            sm_scale=scale,
            block_tables=block_table.to(torch.int32).contiguous(),
            seq_lens=seq_lens,
            disable_split_kv=disable_split_kv,
        )

    def prepare_cudagraph_replay(
        self,
        batch_size: int,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
    ) -> None:
        params = self._decode_graph_params.get(batch_size)
        if params is None:
            raise RuntimeError(
                f"FlashInfer CUDA graph metadata was not captured for batch {batch_size}."
            )
        (
            device,
            dtype,
            use_tensor_cores,
            max_num_blocks,
            page_size,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            kv_dtype,
            scale,
        ) = params
        wrapper = self._get_decode_graph_wrapper(
            device,
            dtype,
            use_tensor_cores,
            batch_size,
            max_num_blocks,
        )
        self._plan_decode_wrapper(
            wrapper,
            block_table[:batch_size],
            cache_seqlens[:batch_size],
            page_size,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            dtype,
            kv_dtype,
            scale,
            disable_split_kv=True,
        )

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
        cu_q = cu_seqlens_q.to(dtype=torch.int32).contiguous()
        # FlashInfer API/behavior differs across versions. Some builds fail in the
        # extended planner path for paged prefill; fall back to the minimal plan call.
        try:
            wrapper.plan(
                cu_q,
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
        except UnboundLocalError:
            wrapper.plan(
                cu_q,
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
        batch_size = q.shape[0]
        use_tensor_cores = q.shape[1] != k_cache.shape[2]
        use_graph_wrapper = self._capture_batch_size == batch_size
        if use_graph_wrapper:
            if self._capture_max_num_blocks is None:
                raise RuntimeError("FlashInfer CUDA graph capture was not initialized.")
            wrapper = self._get_decode_graph_wrapper(
                q.device,
                q.dtype,
                use_tensor_cores,
                batch_size,
                self._capture_max_num_blocks,
            )
            self._decode_graph_params[batch_size] = (
                q.device,
                q.dtype,
                use_tensor_cores,
                self._capture_max_num_blocks,
                page_size,
                q.shape[1],
                k_cache.shape[2],
                q.shape[2],
                k_cache.dtype,
                scale,
            )
            if not torch.cuda.is_current_stream_capturing():
                self._plan_decode_wrapper(
                    wrapper,
                    block_table,
                    cache_seqlens,
                    page_size,
                    q.shape[1],
                    k_cache.shape[2],
                    q.shape[2],
                    q.dtype,
                    k_cache.dtype,
                    scale,
                    disable_split_kv=True,
                )
        else:
            wrapper = self._get_decode_wrapper(q.device, q.dtype, use_tensor_cores)
            self._plan_decode_wrapper(
                wrapper,
                block_table,
                cache_seqlens,
                page_size,
                q.shape[1],
                k_cache.shape[2],
                q.shape[2],
                q.dtype,
                k_cache.dtype,
                scale,
            )
        out = wrapper.run(q, (k_cache, v_cache))
        if out.shape[0] != batch_size:
            raise RuntimeError(
                "FlashInfer decode returned an unexpected batch dimension."
            )
        return out.unsqueeze(1)
