#!/usr/bin/env python3
"""Compare TurboQuant KV-cache performance in nano-vLLM and upstream vLLM.

The script runs each engine/configuration in a fresh subprocess so CUDA/NCCL
state and KV-cache allocations do not leak between candidates.

Timed phases use each package's public LLM.generate API with token-id prompts
and explicit CUDA synchronization before and after the call.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any

RESULT_PREFIX = "RESULT_JSON:"


@dataclass
class PhaseMetrics:
    wall_time_s: float = 0.0
    model_time_s: float | None = None
    prompt_tokens: int = 0
    cached_tokens: int | None = None
    compute_prompt_tokens: int | None = None
    generated_tokens: int = 0
    decode_tokens: int | None = None
    prefill_time_s: float | None = None
    decode_time_s: float | None = None

    @property
    def cache_hit_ratio(self) -> float | None:
        if self.cached_tokens is None:
            return None
        if self.prompt_tokens == 0:
            return 0.0
        return self.cached_tokens / self.prompt_tokens

    @property
    def output_tps(self) -> float:
        if self.wall_time_s <= 0:
            return 0.0
        return self.generated_tokens / self.wall_time_s

    @property
    def prompt_tps(self) -> float:
        if self.wall_time_s <= 0:
            return 0.0
        return self.prompt_tokens / self.wall_time_s

    @property
    def prefill_tps(self) -> float | None:
        if self.prefill_time_s is None or self.compute_prompt_tokens is None:
            return None
        if self.prefill_time_s <= 0:
            return 0.0
        return self.compute_prompt_tokens / self.prefill_time_s

    @property
    def decode_tps(self) -> float | None:
        if self.decode_time_s is None:
            return None
        if self.decode_time_s <= 0:
            return 0.0
        tokens = self.decode_tokens if self.decode_tokens is not None else self.generated_tokens
        return tokens / self.decode_time_s

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["cache_hit_ratio"] = self.cache_hit_ratio
        data["output_tps"] = self.output_tps
        data["prompt_tps"] = self.prompt_tps
        data["prefill_tps"] = self.prefill_tps
        data["decode_tps"] = self.decode_tps
        return data


@dataclass
class CandidateResult:
    ok: bool
    engine: str
    config: str
    phases: dict[str, dict[str, Any]] = field(default_factory=dict)
    workload_fingerprint: str | None = None
    peak_memory_allocated_gb: float | None = None
    peak_memory_reserved_gb: float | None = None
    error: str | None = None
    traceback: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"),
        help="Local HF model path or model id. nano-vLLM requires a local directory.",
    )
    parser.add_argument(
        "--engines",
        type=str,
        default="nanovllm,vllm",
        help="Comma-separated engines: nanovllm,vllm.",
    )
    parser.add_argument(
        "--nano-backends",
        type=str,
        default="default,turboquant",
        help="Comma-separated nano-vLLM kvcache_type values.",
    )
    parser.add_argument(
        "--vllm-kv-cache-dtypes",
        type=str,
        default="auto,turboquant_4bit_nc",
        help=(
            "Comma-separated vLLM kv_cache_dtype values. Current vLLM TurboQuant "
            "presets include turboquant_4bit_nc, turboquant_k8v4, "
            "turboquant_k3v4_nc, and turboquant_3bit_nc."
        ),
    )
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--prefix-len", type=int, default=512)
    parser.add_argument("--suffix-len", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--seed-max-tokens", type=int, default=8)
    parser.add_argument("--warmup-tokens", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--skip-cold", action="store_true", help="Only run seeded prefix-cache-hit phase.")
    parser.add_argument("--json-out", type=str, default=None, help="Optional path for machine-readable results.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--engine", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--config-name", type=str, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def get_vocab_size(model: str) -> int:
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model)
        vocab_size = getattr(cfg, "vocab_size", None)
        if isinstance(vocab_size, int) and vocab_size > 8:
            return vocab_size
    except Exception:
        pass
    return 32000


def token_id(base: int, offset: int, vocab_size: int) -> int:
    usable = max(vocab_size - 1, 1)
    return 1 + ((base + offset) % usable)


def make_shared_prefix(prefix_len: int, vocab_size: int) -> list[int]:
    return [token_id(100, i, vocab_size) for i in range(prefix_len)]


def make_prompts(
    num_prompts: int,
    prefix_len: int,
    suffix_len: int,
    vocab_size: int,
    *,
    shared_prefix: list[int] | None,
) -> list[list[int]]:
    prompts: list[list[int]] = []
    for i in range(num_prompts):
        if shared_prefix is None:
            prefix = [token_id(300 + i * 997, j, vocab_size) for j in range(prefix_len)]
        else:
            prefix = shared_prefix
        suffix = [token_id(20_000 + i * 1009, j, vocab_size) for j in range(suffix_len)]
        prompts.append(prefix + suffix)
    return prompts


def workload_fingerprint(
    args: argparse.Namespace,
    seed_prompt: list[list[int]],
    hit_prompts: list[list[int]],
    cold_prompts: list[list[int]],
) -> str:
    payload = {
        "model": args.model,
        "num_prompts": args.num_prompts,
        "prefix_len": args.prefix_len,
        "suffix_len": args.suffix_len,
        "max_tokens": args.max_tokens,
        "seed_max_tokens": args.seed_max_tokens,
        "warmup_tokens": args.warmup_tokens,
        "temperature": args.temperature,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "tensor_parallel_size": args.tensor_parallel_size,
        "seed_prompt": seed_prompt,
        "hit_prompts": hit_prompts,
        "cold_prompts": [] if args.skip_cold else cold_prompts,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def maybe_cuda_peak_memory() -> tuple[float | None, float | None]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None, None
        allocated = torch.cuda.max_memory_allocated() / (1024**3)
        reserved = torch.cuda.max_memory_reserved() / (1024**3)
        return allocated, reserved
    except Exception:
        return None, None


def cuda_synchronize() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def filter_supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def count_nanovllm_generated_tokens(outputs: Any) -> int:
    total = 0
    for output in outputs:
        if isinstance(output, dict):
            token_ids = output.get("token_ids") or []
        else:
            token_ids = getattr(output, "token_ids", None) or []
        total += len(token_ids)
    return total


def run_nanovllm_generate(llm: Any, prompts: list[list[int]], sampling_params: Any) -> PhaseMetrics:
    cuda_synchronize()
    wall_start = perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    cuda_synchronize()
    wall_time = perf_counter() - wall_start
    generated_tokens = count_nanovllm_generated_tokens(outputs)
    return PhaseMetrics(
        wall_time_s=wall_time,
        prompt_tokens=sum(len(p) for p in prompts),
        generated_tokens=generated_tokens,
        decode_tokens=generated_tokens,
    )


def run_nanovllm_worker(args: argparse.Namespace) -> CandidateResult:
    import torch
    from nanovllm import LLM, SamplingParams

    vocab_size = get_vocab_size(args.model)
    shared_prefix = make_shared_prefix(args.prefix_len, vocab_size)
    seed_prompt = [shared_prefix]
    hit_prompts = make_prompts(
        args.num_prompts,
        args.prefix_len,
        args.suffix_len,
        vocab_size,
        shared_prefix=shared_prefix,
    )
    cold_prompts = make_prompts(
        args.num_prompts,
        args.prefix_len,
        args.suffix_len,
        vocab_size,
        shared_prefix=None,
    )
    fingerprint = workload_fingerprint(args, seed_prompt, hit_prompts, cold_prompts)

    llm = LLM(
        args.model,
        kvcache_type=args.config_name,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    try:
        warmup_sp = SamplingParams(temperature=args.temperature, max_tokens=args.warmup_tokens, ignore_eos=True)
        seed_sp = SamplingParams(temperature=args.temperature, max_tokens=args.seed_max_tokens, ignore_eos=True)
        bench_sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, ignore_eos=True)

        _ = run_nanovllm_generate(llm, [[1, 2, 3, 4]], warmup_sp)
        _ = run_nanovllm_generate(llm, seed_prompt, seed_sp)

        phases = {
            "cache_hit": run_nanovllm_generate(llm, hit_prompts, bench_sp).to_json(),
        }
        if not args.skip_cold:
            phases["cold"] = run_nanovllm_generate(llm, cold_prompts, bench_sp).to_json()

        peak_allocated, peak_reserved = maybe_cuda_peak_memory()
        return CandidateResult(
            ok=True,
            engine="nanovllm",
            config=args.config_name,
            phases=phases,
            workload_fingerprint=fingerprint,
            peak_memory_allocated_gb=peak_allocated,
            peak_memory_reserved_gb=peak_reserved,
        )
    finally:
        llm.exit()
        del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def count_vllm_generated_tokens(outputs: Any) -> int:
    total = 0
    for request_output in outputs:
        choices = getattr(request_output, "outputs", None) or []
        if choices:
            token_ids = getattr(choices[0], "token_ids", None) or []
            total += len(token_ids)
    return total


def run_vllm_generate(llm: Any, prompts: list[list[int]], sampling_params: Any) -> PhaseMetrics:
    request_prompts = [{"prompt_token_ids": prompt} for prompt in prompts]
    cuda_synchronize()
    wall_start = perf_counter()
    outputs = llm.generate(request_prompts, sampling_params, use_tqdm=False)
    cuda_synchronize()
    wall_time = perf_counter() - wall_start
    generated_tokens = count_vllm_generated_tokens(outputs)
    return PhaseMetrics(
        wall_time_s=wall_time,
        prompt_tokens=sum(len(p) for p in prompts),
        generated_tokens=generated_tokens,
        decode_tokens=generated_tokens,
    )


def run_vllm_worker(args: argparse.Namespace) -> CandidateResult:
    from vllm import LLM, SamplingParams

    vocab_size = get_vocab_size(args.model)
    shared_prefix = make_shared_prefix(args.prefix_len, vocab_size)
    seed_prompt = [shared_prefix]
    hit_prompts = make_prompts(
        args.num_prompts,
        args.prefix_len,
        args.suffix_len,
        vocab_size,
        shared_prefix=shared_prefix,
    )
    cold_prompts = make_prompts(
        args.num_prompts,
        args.prefix_len,
        args.suffix_len,
        vocab_size,
        shared_prefix=None,
    )
    fingerprint = workload_fingerprint(args, seed_prompt, hit_prompts, cold_prompts)

    llm_kwargs = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "enforce_eager": args.enforce_eager,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enable_prefix_caching": True,
        "kv_cache_dtype": args.config_name,
        "disable_log_stats": True,
    }
    llm = LLM(**filter_supported_kwargs(LLM, llm_kwargs))
    try:
        warmup_sp = SamplingParams(temperature=args.temperature, max_tokens=args.warmup_tokens, ignore_eos=True)
        seed_sp = SamplingParams(temperature=args.temperature, max_tokens=args.seed_max_tokens, ignore_eos=True)
        bench_sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, ignore_eos=True)

        _ = run_vllm_generate(llm, [[1, 2, 3, 4]], warmup_sp)
        _ = run_vllm_generate(llm, seed_prompt, seed_sp)

        phases = {
            "cache_hit": run_vllm_generate(llm, hit_prompts, bench_sp).to_json(),
        }
        if not args.skip_cold:
            phases["cold"] = run_vllm_generate(llm, cold_prompts, bench_sp).to_json()

        peak_allocated, peak_reserved = maybe_cuda_peak_memory()
        return CandidateResult(
            ok=True,
            engine="vllm",
            config=args.config_name,
            phases=phases,
            workload_fingerprint=fingerprint,
            peak_memory_allocated_gb=peak_allocated,
            peak_memory_reserved_gb=peak_reserved,
        )
    finally:
        shutdown = getattr(llm, "shutdown", None)
        if callable(shutdown):
            shutdown()
        del llm


def run_worker(args: argparse.Namespace) -> CandidateResult:
    try:
        if args.engine == "nanovllm":
            return run_nanovllm_worker(args)
        if args.engine == "vllm":
            return run_vllm_worker(args)
        raise ValueError(f"Unknown worker engine: {args.engine}")
    except Exception as exc:
        return CandidateResult(
            ok=False,
            engine=args.engine or "unknown",
            config=args.config_name or "unknown",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )


def print_worker_result(result: CandidateResult) -> None:
    print(f"{RESULT_PREFIX} {json.dumps(asdict(result), sort_keys=True)}", flush=True)


def child_args(base_args: argparse.Namespace, engine: str, config: str) -> list[str]:
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--worker",
        "--engine",
        engine,
        "--config-name",
        config,
        "--model",
        base_args.model,
        "--num-prompts",
        str(base_args.num_prompts),
        "--prefix-len",
        str(base_args.prefix_len),
        "--suffix-len",
        str(base_args.suffix_len),
        "--max-tokens",
        str(base_args.max_tokens),
        "--seed-max-tokens",
        str(base_args.seed_max_tokens),
        "--warmup-tokens",
        str(base_args.warmup_tokens),
        "--temperature",
        str(base_args.temperature),
        "--max-model-len",
        str(base_args.max_model_len),
        "--max-num-batched-tokens",
        str(base_args.max_num_batched_tokens),
        "--max-num-seqs",
        str(base_args.max_num_seqs),
        "--gpu-memory-utilization",
        str(base_args.gpu_memory_utilization),
        "--tensor-parallel-size",
        str(base_args.tensor_parallel_size),
    ]
    if base_args.enforce_eager:
        cmd.append("--enforce-eager")
    if base_args.skip_cold:
        cmd.append("--skip-cold")
    return cmd


def run_candidate(args: argparse.Namespace, engine: str, config: str) -> CandidateResult:
    print(f"\n=== Running {engine}:{config} ===", flush=True)
    proc = subprocess.run(
        child_args(args, engine, config),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    result_line = None
    passthrough_lines: list[str] = []
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            result_line = line[len(RESULT_PREFIX) :].strip()
        else:
            passthrough_lines.append(line)
    if passthrough_lines:
        print("\n".join(passthrough_lines[-40:]), flush=True)
    if result_line is None:
        return CandidateResult(
            ok=False,
            engine=engine,
            config=config,
            error=f"Worker exited with code {proc.returncode} and did not emit {RESULT_PREFIX}",
            traceback=proc.stdout[-4000:],
        )
    data = json.loads(result_line)
    return CandidateResult(**data)


def format_float(value: Any, digits: int = 3, na: str = "n/a") -> str:
    if value is None:
        return na
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return na


def print_summary(results: list[CandidateResult]) -> None:
    print("\n=== Summary ===")
    failures = [r for r in results if not r.ok]
    successes = [r for r in results if r.ok]

    if failures:
        print("\nSkipped/failed candidates:")
        for r in failures:
            print(f"- {r.engine}:{r.config}: {r.error}")

    fingerprints = sorted({r.workload_fingerprint for r in successes if r.workload_fingerprint})
    if len(fingerprints) == 1:
        print(f"\nWorkload fingerprint: {fingerprints[0]} (all successful candidates match)")
    elif len(fingerprints) > 1:
        print(f"\nWarning: workload fingerprints differ: {', '.join(fingerprints)}")

    phases = sorted({phase for r in successes for phase in r.phases})
    for phase in phases:
        rows = []
        for r in successes:
            metrics = r.phases.get(phase)
            if metrics is None:
                continue
            rows.append((metrics.get("wall_time_s", float("inf")), r, metrics))
        rows.sort(key=lambda item: item[0])
        print(f"\nPhase: {phase} (lower wall time is better)")
        print("rank  engine      config                 wall_s  out_tok/s  prompt_tok/s  hit_ratio  peak_GB")
        for rank, (_, r, m) in enumerate(rows, start=1):
            peak = r.peak_memory_reserved_gb
            print(
                f"{rank:<5} "
                f"{r.engine:<11} "
                f"{r.config:<22} "
                f"{format_float(m.get('wall_time_s')):>7}  "
                f"{format_float(m.get('output_tps'), 2):>9}  "
                f"{format_float(m.get('prompt_tps'), 2):>12}  "
                f"{format_float(m.get('cache_hit_ratio')):>9}  "
                f"{format_float(peak, 2):>7}"
            )

        baseline = next((m for _, r, m in rows if r.engine == "nanovllm" and r.config == "default"), None)
        if baseline is not None and baseline.get("wall_time_s", 0) > 0:
            base_time = baseline["wall_time_s"]
            print("Speed vs nanovllm:default:")
            for _, r, m in rows:
                speedup = base_time / m["wall_time_s"] if m.get("wall_time_s", 0) > 0 else 0.0
                print(f"- {r.engine}:{r.config}: {speedup:.2f}x")


def main() -> None:
    args = parse_args()
    if args.worker:
        print_worker_result(run_worker(args))
        return

    if args.prefix_len < 256:
        print("Warning: prefix_len < 256 may reduce/disable block-level prefix cache hits in nano-vLLM.")

    engines = csv_list(args.engines)
    candidates: list[tuple[str, str]] = []
    if "nanovllm" in engines:
        candidates.extend(("nanovllm", cfg) for cfg in csv_list(args.nano_backends))
    if "vllm" in engines:
        candidates.extend(("vllm", cfg) for cfg in csv_list(args.vllm_kv_cache_dtypes))
    unknown = sorted(set(engines) - {"nanovllm", "vllm"})
    if unknown:
        raise ValueError(f"Unknown engines: {unknown}")
    if not candidates:
        raise ValueError("No benchmark candidates selected.")

    print("Benchmark workload:")
    print(
        f"- model={args.model}\n"
        f"- num_prompts={args.num_prompts}, prefix_len={args.prefix_len}, "
        f"suffix_len={args.suffix_len}, max_tokens={args.max_tokens}\n"
        f"- timing=public generate() API with CUDA synchronization\n"
        f"- candidates={', '.join(f'{e}:{c}' for e, c in candidates)}"
    )

    results = [run_candidate(args, engine, config) for engine, config in candidates]
    print_summary(results)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2, sort_keys=True)
        print(f"\nWrote JSON results to {args.json_out}")


if __name__ == "__main__":
    main()
