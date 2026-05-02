#!/usr/bin/env python3
"""Generate static TurboQuant Lloyd-Max centroid/boundary tables.

The runtime table in nanovllm/kvcache/turboquant.py stores values for a
unit-variance normal distribution. TurboQuant scales those values by
1 / sqrt(head_dim) at runtime because normalized key components have variance
approximately 1 / head_dim.
"""

from __future__ import annotations

import argparse
import math


def trapz(f, a: float, b: float, n: int) -> float:
    h = (b - a) / n
    result = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        result += f(a + i * h)
    return result * h


def gaussian_pdf(x: float, sigma2: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi * sigma2)) * math.exp(
        -(x * x) / (2.0 * sigma2)
    )


def lloyd_max_centroids(
    bits: int,
    sigma2: float = 1.0,
    iterations: int = 200,
    integration_steps: int = 200,
    tolerance: float = 1e-10,
) -> tuple[list[float], list[float]]:
    if bits <= 0:
        raise ValueError("bits must be positive")

    n_levels = 1 << bits
    sigma = math.sqrt(sigma2)

    def pdf(x: float) -> float:
        return gaussian_pdf(x, sigma2)

    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]
    for _ in range(iterations):
        boundaries = [
            (centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)
        ]
        edges = [lo * 3.0] + boundaries + [hi * 3.0]
        next_centroids = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            numerator = trapz(lambda x: x * pdf(x), a, b, integration_steps)
            denominator = trapz(pdf, a, b, integration_steps)
            next_centroids.append(
                numerator / denominator if denominator > 1e-15 else centroids[i]
            )
        if max(
            abs(next_centroids[i] - centroids[i]) for i in range(n_levels)
        ) < tolerance:
            centroids = next_centroids
            break
        centroids = next_centroids

    boundaries = [
        (centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)
    ]
    return centroids, boundaries


def format_tuple(values: list[float], indent: str = "            ") -> str:
    lines = ["        ("]
    for value in values:
        normalized = 0.0 if abs(value) < 5e-13 else value
        lines.append(f"{indent}{normalized:.10f},")
    lines.append("        )")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bits",
        type=int,
        nargs="+",
        default=[3, 4],
        help="Bit widths to generate. Default: 3 4.",
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        default=None,
        help=(
            "Optionally print values scaled for a specific head_dim. "
            "Without this, unit-variance tables are emitted."
        ),
    )
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--integration-steps", type=int, default=200)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scale = 1.0
    if args.head_dim is not None:
        if args.head_dim <= 0:
            raise ValueError("--head-dim must be positive")
        scale = 1.0 / math.sqrt(args.head_dim)

    print("_LLOYD_MAX_UNIT_TABLES = {")
    for bits in args.bits:
        centroids, boundaries = lloyd_max_centroids(
            bits,
            iterations=args.iterations,
            integration_steps=args.integration_steps,
            tolerance=args.tolerance,
        )
        centroids = [value * scale for value in centroids]
        boundaries = [value * scale for value in boundaries]
        print(f"    {bits}: (")
        print(f"{format_tuple(centroids)},")
        print(f"{format_tuple(boundaries)},")
        print("    ),")
    print("}")


if __name__ == "__main__":
    main()
