"""TurboQuant preset/config helpers."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass


TURBOQUANT_PRESETS: dict[str, tuple[int, int]] = {
    "turboquant": (4, 4),
    "turboquant_k4v4": (4, 4),
    "turboquant_k3v4": (3, 4),
    "turboquant_k3v3": (3, 3),
}


@dataclass(frozen=True)
class TurboQuantConfig:
    """Runtime configuration shared by the TurboQuant cache and backend."""

    preset: str = "turboquant_k4v4"
    key_bits: int = 4
    value_bits: int = 4
    norm_correction: bool = False
    use_random_signs: bool = False
    dense_decode_fallback: bool = False
    small_prefill_threshold: int = 256

    def __post_init__(self) -> None:
        if self.key_bits not in (3, 4) or self.value_bits not in (3, 4):
            raise ValueError(
                "TurboQuant supports only 3-bit and 4-bit key/value modes."
            )
        if self.preset in TURBOQUANT_PRESETS:
            expected = TURBOQUANT_PRESETS[self.preset]
            actual = (self.key_bits, self.value_bits)
            if actual != expected:
                raise ValueError(
                    f"TurboQuant preset {self.preset!r} expects k/v bits "
                    f"{expected}, got {actual}."
                )
        if self.small_prefill_threshold < 0:
            raise ValueError("small_prefill_threshold must be non-negative.")

    @property
    def num_key_levels(self) -> int:
        return 1 << self.key_bits

    @property
    def num_value_levels(self) -> int:
        return 1 << self.value_bits

    def packed_size_bytes(self, num_values: int, bits: int) -> int:
        return packed_size_bytes(num_values, bits)

    def key_packed_bytes(self, head_dim: int) -> int:
        return packed_size_bytes(head_dim, self.key_bits)

    def value_packed_bytes(self, head_dim: int) -> int:
        return packed_size_bytes(head_dim, self.value_bits)

    @property
    def metadata_bytes_per_head_token(self) -> int:
        # k_norm, v_scale, v_zero are stored as float16.
        return 6


def packed_size_bytes(num_values: int, bits: int) -> int:
    if bits not in (3, 4):
        raise ValueError(f"TurboQuant supports only 3-bit and 4-bit packing, got {bits}.")
    if num_values <= 0:
        raise ValueError("num_values must be positive.")
    if bits == 3 and num_values % 8 != 0:
        raise ValueError(
            "TurboQuant 3-bit packing requires head_dim to be divisible by 8."
        )
    if bits == 4 and num_values % 2 != 0:
        raise ValueError(
            "TurboQuant 4-bit packing requires head_dim to be divisible by 2."
        )
    return math.ceil(num_values * bits / 8)


def resolve_turboquant_config(kvcache_type: str) -> TurboQuantConfig | None:
    if kvcache_type not in TURBOQUANT_PRESETS:
        return None
    key_bits, value_bits = TURBOQUANT_PRESETS[kvcache_type]
    dense_decode_fallback = os.environ.get("NANOVLLM_TQ_DENSE_DECODE", "0") == "1"
    preset = "turboquant_k4v4" if kvcache_type == "turboquant" else kvcache_type
    return TurboQuantConfig(
        preset=preset,
        key_bits=key_bits,
        value_bits=value_bits,
        dense_decode_fallback=dense_decode_fallback,
    )


def turboquant_config_for_preset(preset: str) -> TurboQuantConfig:
    config = resolve_turboquant_config(preset)
    if config is None:
        raise ValueError(f"Unknown TurboQuant preset: {preset}")
    return config


def is_turboquant_type(kvcache_type: str) -> bool:
    return kvcache_type in TURBOQUANT_PRESETS
