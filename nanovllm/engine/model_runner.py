import pickle
import os
import warnings
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.registry import ModelRegistry
from nanovllm.kvcache import KVCacheRegistry
from nanovllm.kvcache.turboquant_config import resolve_turboquant_config
from nanovllm.layers.flash_attn_backend import (
    FlashAttentionRegistry,
    ensure_builtin_backends_registered,
)
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    @staticmethod
    def _cudagraph_batch_sizes(max_num_seqs: int) -> list[int]:
        raw_limit = os.environ.get("NANOVLLM_CUDAGRAPH_MAX_BS", "8")
        try:
            graph_limit = int(raw_limit)
        except ValueError:
            graph_limit = 8
        graph_limit = max(1, min(max_num_seqs, graph_limit))
        sizes = []
        size = 1
        while size < graph_limit:
            sizes.append(size)
            size *= 2
        sizes.append(graph_limit)
        return sorted(set(sizes))

    @staticmethod
    def _normalize_torch_dtype(raw_dtype) -> torch.dtype:
        if isinstance(raw_dtype, torch.dtype):
            return raw_dtype
        if raw_dtype is None:
            return torch.float16
        if isinstance(raw_dtype, str):
            key = raw_dtype.replace("torch.", "").lower()
            mapping = {
                "float16": torch.float16,
                "half": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
                "float": torch.float32,
            }
            if key in mapping:
                return mapping[key]
        raise ValueError(f"Unsupported model dtype: {raw_dtype!r}")

    def _resolve_model_dtype(self, raw_dtype) -> torch.dtype:
        dtype = self._normalize_torch_dtype(raw_dtype)
        major, minor = torch.cuda.get_device_capability(self.rank)
        if dtype == torch.bfloat16 and major < 8:
            warnings.warn(
                "Model config requests bfloat16, but this GPU does not support native BF16 "
                f"(compute capability {major}.{minor}). Falling back to float16."
            )
            return torch.float16
        return dtype

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        raw_model_dtype = getattr(hf_config, "dtype", None)
        if raw_model_dtype is None:
            raw_model_dtype = getattr(hf_config, "torch_dtype", None)
        self.model_dtype = self._resolve_model_dtype(raw_model_dtype)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.model_dtype)
        torch.set_default_device("cuda")
        
        # Create model from registry
        self.model = ModelRegistry.create_model(hf_config, config.model_architecture)
        load_model(self.model, config.model)
        
        # Create KV cache backend. TurboQuant exposes several presets that share
        # one cache/backend implementation.
        self.turboquant_config = resolve_turboquant_config(config.kvcache_type)
        cache_backend_name = "turboquant" if self.turboquant_config is not None else config.kvcache_type
        self.cache_backend_class = KVCacheRegistry.get_cache_class(cache_backend_name)
        self.cache_backend_kwargs = (
            {"turboquant_config": self.turboquant_config}
            if self.turboquant_config is not None
            else {}
        )
        self.attn_backend = self.create_attn_backend(cache_backend_name)
        if not self.enforce_eager and not self.attn_backend.supports_cudagraph_capture:
            warnings.warn(
                f"{self.attn_backend.name} does not support CUDA graph capture in "
                "this integration. Falling back to eager decode."
            )
            self.enforce_eager = True
        
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def create_attn_backend(self, backend_name: str):
        ensure_builtin_backends_registered()
        try:
            backend_cls = FlashAttentionRegistry.get(backend_name)
        except KeyError:
            backend_name = "default"
            backend_cls = FlashAttentionRegistry.get("default")
        if backend_name == "turboquant":
            return backend_cls(self.turboquant_config)
        return backend_cls()

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        
        # Create a temporary cache backend to calculate block size
        temp_cache = self.cache_backend_class(
            num_blocks=1,
            block_size=self.block_size,
            num_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=self.model_dtype,
            **self.cache_backend_kwargs,
        )
        block_bytes = temp_cache.get_cache_block_size_bytes()
        
        # Count number of layers that need cache (for correct memory calculation)
        num_cache_layers = hf_config.num_hidden_layers
        
        # Calculate number of blocks accounting for all layers
        # Each layer gets its own cache, so divide by num_layers
        config.num_kvcache_blocks = int(
            total * config.gpu_memory_utilization - used - peak + current
        ) // (num_cache_layers * block_bytes)
        assert config.num_kvcache_blocks > 0, \
            f"Not enough GPU memory for KV cache. Available: {total * config.gpu_memory_utilization - used - peak + current} bytes, needed per block: {block_bytes * num_cache_layers} bytes"
        
        # Create cache backends for each layer and allocate tensors
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                # Create cache backend for this layer
                cache_backend = self.cache_backend_class(
                    num_blocks=config.num_kvcache_blocks,
                    block_size=self.block_size,
                    num_heads=num_kv_heads,
                    head_dim=head_dim,
                    dtype=self.model_dtype,
                    **self.cache_backend_kwargs,
                )
                
                # Let cache backend allocate its tensors
                # Handle both simple (k, v) and quantized (k, v, scales, ...) caches
                cache_tensors = cache_backend.allocate()
                k_cache, v_cache = cache_tensors[0], cache_tensors[1]
                additional_tensors = cache_tensors[2:] if len(cache_tensors) > 2 else ()
                
                # Assign to module
                module.k_cache = k_cache
                module.v_cache = v_cache
                module.additional_cache_tensors = additional_tensors
                module.cache_backend = cache_backend
                module.attn_backend = self.attn_backend
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if (
            is_prefill
            or self.enforce_eager
            or not hasattr(self, "graphs")
            or input_ids.size(0) > self.graph_bs[-1]
        ):
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph_bs = next(x for x in self.graph_bs if x >= bs)
            graph = self.graphs[graph_bs]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"].zero_()
            graph_vars["block_tables"][
                :bs,
                :context.block_tables.size(1),
            ] = context.block_tables
            self.attn_backend.prepare_cudagraph_replay(
                graph_bs,
                graph_vars["context_lens"][:graph_bs],
                graph_vars["block_tables"][:graph_bs],
            )
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        self.graph_bs = self._cudagraph_batch_sizes(self.config.max_num_seqs)
        max_bs = self.graph_bs[-1]
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            context_lens.zero_()
            context_lens[:bs].fill_(1)
            block_tables.zero_()
            self.attn_backend.begin_cudagraph_capture(bs, max_num_blocks)
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )
            try:
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
                with torch.cuda.graph(graph, self.graph_pool):
                    outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            finally:
                self.attn_backend.end_cudagraph_capture()
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
