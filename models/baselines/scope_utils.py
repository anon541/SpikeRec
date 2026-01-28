"""Training scope utilities (repeat/new masks)."""

from __future__ import annotations

from typing import Tuple

import torch


def scope_masks(
    inputs: torch.Tensor, pos: torch.Tensor, scope: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return masks for valid targets and repeat/new scope selection."""
    valid_mask = pos != 0
    if not valid_mask.any():
        zero = torch.zeros_like(pos, dtype=torch.bool)
        return zero, zero, zero

    seq_len = inputs.shape[1]
    tri = torch.tril(torch.ones((seq_len, seq_len), device=inputs.device, dtype=torch.bool))
    matches = (pos.unsqueeze(2) == inputs.unsqueeze(1)) & (inputs.unsqueeze(1) != 0)
    rep_mask = (matches & tri.unsqueeze(0)).any(dim=2)
    rep_mask = rep_mask & valid_mask

    scope = scope or "all"
    if scope == "rep":
        selected = rep_mask
    elif scope == "new":
        selected = valid_mask & (~rep_mask)
    else:
        selected = valid_mask

    return selected, rep_mask, selected
