"""Shared attention helpers."""

import torch


def normalize_decode_query(q: torch.Tensor) -> torch.Tensor:
    """Normalize decode query shape to [batch, num_heads, head_dim]."""
    if q.ndim == 4:
        if q.shape[1] != 1:
            raise ValueError(
                "Expected decode query tensor with sequence length 1 in dimension 1."
            )
        return q.squeeze(1)
    if q.ndim == 3:
        return q
    if q.ndim == 2:
        return q.unsqueeze(0)
    raise ValueError(f"Unsupported decode query shape: {tuple(q.shape)}")
