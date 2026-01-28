"""Shared helpers for availability (A_t) candidate filtering in evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass(frozen=True)
class CandidateSet:
    items: torch.Tensor
    buckets: Optional[torch.Tensor] = None


def build_step_candidates(args, steps: torch.Tensor) -> Dict[int, CandidateSet]:
    """Build availability candidates (A_t) for unique steps in the batch."""
    if steps.dim() == 2:
        step_tensor = steps[:, -1]
    else:
        step_tensor = steps

    unique_steps = torch.unique(step_tensor.detach().cpu())
    has_bucket = bool(
        getattr(args, "has_viewer_bucket", False)
        and getattr(args, "num_viewer_buckets", 0) > 0
    )
    candidates: Dict[int, CandidateSet] = {}

    av_tens = getattr(args, "av_tens", None)
    av_bucket_tens = getattr(args, "av_bucket_tens", None) if has_bucket else None

    for step_val in unique_steps.tolist():
        step = int(step_val)
        av = None
        av_bucket = None

        if av_tens is not None and step < av_tens.shape[0]:
            raw_av = av_tens[step]
            mask = raw_av != 0
            if not mask.any():
                continue
            av = raw_av[mask]
            if av_bucket_tens is not None and step < av_bucket_tens.shape[0]:
                av_bucket = av_bucket_tens[step][mask]
        else:
            av_list = getattr(args, "ts", {}).get(step, [])
            if not av_list:
                continue
            av = torch.LongTensor(av_list).to(args.device)
            if has_bucket:
                bucket_vals = getattr(args, "ts_bucket", {}).get(step, [])
                if bucket_vals:
                    av_bucket = torch.LongTensor(bucket_vals).to(args.device)
                    if av_bucket.shape[0] < av.shape[0]:
                        pad = torch.zeros(
                            av.shape[0] - av_bucket.shape[0],
                            dtype=torch.long,
                            device=args.device,
                        )
                        av_bucket = torch.cat([av_bucket, pad], dim=0)

        if av is not None and av.numel() > 0:
            candidates[step] = CandidateSet(items=av, buckets=av_bucket)

    return candidates
