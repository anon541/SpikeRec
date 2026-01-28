"""Training debug helpers shared across baseline models."""

from __future__ import annotations

from typing import Optional

import torch


def log_and_accumulate_train_debug(
    model,
    model_name: str,
    scope: str,
    inputs: torch.Tensor,
    valid_mask: torch.Tensor,
    pos: torch.Tensor,
    rep_mask_batch: Optional[torch.Tensor],
    batch_mask: Optional[torch.Tensor],
) -> None:
    if not getattr(model.args, "debug", False):
        return
    dbg = getattr(model, "_dbg", None)
    if not isinstance(dbg, dict):
        return

    dbg["model"] = model_name
    dbg["batches"] = dbg.get("batches", 0) + 1
    token_count = int(valid_mask.sum().item()) if valid_mask is not None else 0
    dbg["tokens_all_sum"] = dbg.get("tokens_all_sum", 0) + token_count
    dbg["tokens_sel_sum"] = dbg.get("tokens_sel_sum", 0) + token_count

    scope = scope or "all"
    if scope == "rep":
        dbg["rep_batches_sum"] = dbg.get("rep_batches_sum", 0) + 1
    elif scope == "new":
        dbg["new_batches_sum"] = dbg.get("new_batches_sum", 0) + 1
    else:
        dbg["selected_batches_sum"] = dbg.get("selected_batches_sum", 0) + 1
