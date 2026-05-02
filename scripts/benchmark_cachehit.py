#!/usr/bin/env python3
"""Benchmark nano-vllm backends with explicit prefix-cache-hit and decode phases."""

import argparse
import os
from dataclasses import dataclass
from time import perf_counter

import torch

from nanovllm import LLM, SamplingParams


@dataclass
class RunMetrics:
    prefill_total_prompt_tokens: int = 0
    prefill_cached_tokens: int = 0
    prefill_compute_tokens: int = 0
    prefill_time_s: float = 0.0
    decode_tokens: int = 0
    decode_time_s: float = 0.0
    total_time_s: float = 0.0

    @property
    def prefill_hit_ratio(self) -> float:
        if self.prefill_total_prompt_tokens == 0:
            return 0.0
        return self.prefill_cached_tokens / self.prefill_total_prompt_tokens

    @property
    def prefill_tps(self) -> float:
        if self.prefill_time_s <= 0:
            return 0.0
        return self.prefill_compute_tokens / self.prefill_time_s

    @property
    def decode_tps(self) -> float:
        if self.decode_time_s <= 0:
            return 0.0
        return self.decode_tokens / self.decode_time_s


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"),
        help="Path to local HF model directory.",
    )
    parser.add_argument(
        "--backends",
        type=str,
        default="default,turboquant",
        help=(
            "Comma-separated kvcache backends to benchmark "
            "(e.g. default,turboquant_k4v4,turboquant_k3v4,int8)."
        ),
    )
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument(
        "--prefix-len",
        type=int,
        default=512,
        help="Shared prefix length for cache-hit workload. Use >= 256 and preferably multiple of 256.",
    )
    parser.add_argument("--suffix-len", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def make_shared_prefix(prefix_len: int) -> list[int]:
    return [100 + (i % 1000) for i in range(prefix_len)]


def make_prompts(
    num_prompts: int,
    prefix_len: int,
    suffix_len: int,
    *,
    shared_prefix: list[int] | None,
) -> list[list[int]]:
    prompts: list[list[int]] = []
    for i in range(num_prompts):
        if shared_prefix is not None:
            prefix = shared_prefix
        else:
            prefix = [300 + ((j + i * 131) % 9000) for j in range(prefix_len)]
        suffix = [20_000 + i * 1000 + j for j in range(suffix_len)]
        prompts.append(prefix + suffix)
    return prompts


def run_requests(
    llm: LLM,
    prompts: list[list[int]],
    sampling_params: SamplingParams,
) -> RunMetrics:
    for prompt in prompts:
        llm.add_request(prompt, sampling_params)

    m = RunMetrics()
    while not llm.is_finished():
        seqs, is_prefill = llm.scheduler.schedule()
        if is_prefill:
            m.prefill_total_prompt_tokens += sum(len(seq) for seq in seqs)
            m.prefill_cached_tokens += sum(seq.num_cached_tokens for seq in seqs)

        t0 = perf_counter()
        token_ids = llm.model_runner.call("run", seqs, is_prefill)
        dt = perf_counter() - t0
        m.total_time_s += dt
        llm.scheduler.postprocess(seqs, token_ids)

        if is_prefill:
            m.prefill_compute_tokens += sum(len(seq) - seq.num_cached_tokens for seq in seqs)
            m.prefill_time_s += dt
        else:
            m.decode_tokens += len(seqs)
            m.decode_time_s += dt
    return m


def print_metrics(title: str, metrics: RunMetrics) -> None:
    print(f"\n[{title}]")
    print(f"total_time_s={metrics.total_time_s:.4f}")
    print(
        "prefill: "
        f"prompt_tokens={metrics.prefill_total_prompt_tokens}, "
        f"cached_tokens={metrics.prefill_cached_tokens}, "
        f"compute_tokens={metrics.prefill_compute_tokens}, "
        f"hit_ratio={metrics.prefill_hit_ratio:.3f}, "
        f"throughput={metrics.prefill_tps:.2f} tok/s"
    )
    print(
        "decode: "
        f"gen_tokens={metrics.decode_tokens}, "
        f"throughput={metrics.decode_tps:.2f} tok/s"
    )


def main() -> None:
    args = parse_args()
    backends = [x.strip() for x in args.backends.split(",") if x.strip()]
    if not backends:
        raise ValueError("No backends provided.")
    if args.prefix_len < 256:
        print("Warning: prefix_len < 256 may reduce/disable cache hits with default block sizing.")

    shared_prefix = make_shared_prefix(args.prefix_len)
    seed_prompt = [shared_prefix]
    hit_prompts = make_prompts(
        args.num_prompts,
        args.prefix_len,
        args.suffix_len,
        shared_prefix=shared_prefix,
    )
    cold_prompts = make_prompts(
        args.num_prompts,
        args.prefix_len,
        args.suffix_len,
        shared_prefix=None,
    )

    seed_sp = SamplingParams(temperature=args.temperature, max_tokens=8, ignore_eos=True)
    bench_sp = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )

    results: dict[tuple[str, str], RunMetrics] = {}
    for backend in backends:
        print(f"\n=== Backend: {backend} ===")
        llm = LLM(
            args.model,
            kvcache_type=backend,
            tensor_parallel_size=1,
            enforce_eager=args.enforce_eager,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
        )
        try:
            # Warmup for kernel init/JIT effects.
            _ = llm.generate([[1, 2, 3, 4]], SamplingParams(temperature=0.8, max_tokens=4, ignore_eos=True), use_tqdm=False)

            # Seed prefix blocks into cache so the next phase has true cache hits.
            _ = run_requests(llm, seed_prompt, seed_sp)

            hit_metrics = run_requests(llm, hit_prompts, bench_sp)
            cold_metrics = run_requests(llm, cold_prompts, bench_sp)
            results[(backend, "cache_hit")] = hit_metrics
            results[(backend, "cold")] = cold_metrics

            print_metrics(f"{backend} cache_hit", hit_metrics)
            print_metrics(f"{backend} cold", cold_metrics)
            if hit_metrics.prefill_cached_tokens == 0:
                print("WARNING: expected prefill cache hits, but cached_tokens is 0.")
        finally:
            llm.exit()
            del llm
            torch.cuda.empty_cache()

    print("\n=== Summary (lower total_time_s is better) ===")
    for phase in ("cache_hit", "cold"):
        phase_rows = []
        for backend in backends:
            m = results[(backend, phase)]
            phase_rows.append((m.total_time_s, backend, m.prefill_tps, m.decode_tps, m.prefill_hit_ratio))
        phase_rows.sort(key=lambda x: x[0])
        print(f"\nPhase: {phase}")
        for rank, row in enumerate(phase_rows, start=1):
            total_time_s, backend, prefill_tps, decode_tps, hit_ratio = row
            print(
                f"{rank}. {backend}: total={total_time_s:.4f}s, "
                f"prefill_tps={prefill_tps:.2f}, decode_tps={decode_tps:.2f}, "
                f"prefill_hit_ratio={hit_ratio:.3f}"
            )


if __name__ == "__main__":
    main()
