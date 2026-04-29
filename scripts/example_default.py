#!/usr/bin/env python3
"""Run a small nano-vllm generation example with default KV cache/backend."""

import argparse
import os

from nanovllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"),
        help="Path to local HF model directory.",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    llm = LLM(
        args.model,
        kvcache_type="default",
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=1,
    )
    prompts = [
        "Explain in two short bullet points why KV cache helps LLM inference.",
        "Write one sentence about Tesla T4 suitability for inference.",
    ]
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        ignore_eos=False,
    )
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    for i, out in enumerate(outputs):
        print(f"\n=== DEFAULT OUTPUT {i} ===")
        print(out["text"])


if __name__ == "__main__":
    main()
