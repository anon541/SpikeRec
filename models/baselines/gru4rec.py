"""GRU4Rec baseline and spike-aware variant."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from data.candidates import build_step_candidates
from models.baselines.liverec import (
    _compute_repeat_mask_vectorized,
    _extract_user_id,
    _repeat_flags_for_av,
    _update_tier_stats,
    sample_negs,
    sample_negs_k,
)
from models.baselines.scope_utils import scope_masks
from models.baselines.train_debug import log_and_accumulate_train_debug

class GRU4Rec(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.viewer_mode = getattr(args, "viewer_feat_mode", "off")

        self.item_embedding = nn.Embedding(args.N + 1, args.K, padding_idx=0)
        self.emb_dropout = nn.Dropout(p=0.2)

        self.use_bucket = bool(
            self.viewer_mode in ("bucket", "spike")
            and getattr(args, "has_viewer_bucket", False)
            and getattr(args, "num_viewer_buckets", 0) > 0
        )
        if self.use_bucket:
            num_buckets = int(getattr(args, "num_viewer_buckets", 0)) + 2
            self.bucket_emb = nn.Embedding(num_buckets, args.K, padding_idx=0)
        else:
            self.bucket_emb = None

        self.gru = nn.GRU(input_size=args.K, hidden_size=args.K, batch_first=True)

    def _encode(self, inputs: torch.Tensor, bucket_inputs: Optional[torch.Tensor] = None) -> torch.Tensor:
        seqs = self.item_embedding(inputs)
        if self.use_bucket and bucket_inputs is not None:
            seqs = seqs + self.bucket_emb(bucket_inputs)
        seqs = self.emb_dropout(seqs)

        lengths = (inputs != 0).sum(dim=1)
        safe_lengths = torch.clamp(lengths, min=1)

        # Reverse to handle left-padding before packing.
        seqs_rev = torch.flip(seqs, dims=[1])
        packed = pack_padded_sequence(
            seqs_rev, safe_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        out_rev, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=inputs.shape[1]
        )
        feats = torch.flip(out_rev, dims=[1])

        if (lengths == 0).any():
            feats = feats * (lengths > 0).view(-1, 1, 1)
        return feats

    def predict(self, feats: torch.Tensor, items: torch.Tensor, bucket_items: Optional[torch.Tensor] = None) -> torch.Tensor:
        item_embs = self.item_embedding(items)
        if self.use_bucket and bucket_items is not None:
            item_embs = item_embs + self.bucket_emb(bucket_items)
        return (feats * item_embs).sum(dim=-1)

    def _score_items(
        self,
        feats: torch.Tensor,
        items: torch.Tensor,
        bucket_items: Optional[torch.Tensor] = None,
        xtsy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.predict(feats, items, bucket_items)

    def _score_candidates(
        self,
        user_feat: torch.Tensor,
        items: torch.Tensor,
        bucket_items: Optional[torch.Tensor],
        step: int,
        item_embs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if item_embs is None:
            item_embs = self.item_embedding(items)
            if self.use_bucket and bucket_items is not None:
                item_embs = item_embs + self.bucket_emb(bucket_items[: item_embs.shape[0]])
        return (user_feat * item_embs).sum(dim=-1)

    def train_step(self, data: torch.Tensor) -> torch.Tensor:
        inputs, pos = data[:, :, 3], data[:, :, 5]
        xtsy = data[:, :, 6]
        bucket_inputs = None
        bucket_targets = None
        if self.use_bucket:
            idx_in = getattr(self.args, "bucket_input_idx", None)
            idx_tg = getattr(self.args, "bucket_target_idx", None)
            if idx_in is not None and idx_in < data.shape[2]:
                bucket_inputs = data[:, :, idx_in]
            if idx_tg is not None and idx_tg < data.shape[2]:
                bucket_targets = data[:, :, idx_tg]

        feats = self._encode(inputs, bucket_inputs)
        pos_logits = self._score_items(feats, pos, bucket_targets, xtsy)

        scope = getattr(self.args, "train_scope", "all") or "all"
        valid_mask, rep_mask_batch, batch_mask = scope_masks(inputs, pos, scope)
        loss_pos = (-torch.log(pos_logits[valid_mask].sigmoid() + 1e-24)).sum()

        num_negs = int(getattr(self.args, "num_negs", 1) or 1)
        loss_neg_sum = 0.0
        if num_negs == 1:
            neg_sample = sample_negs(data, self.args)
            if isinstance(neg_sample, tuple):
                neg, neg_bucket = neg_sample
            else:
                neg, neg_bucket = neg_sample, None
            neg = neg.to(self.args.device)
            if neg_bucket is not None:
                neg_bucket = neg_bucket.to(self.args.device)
            neg_logits = self._score_items(feats, neg, neg_bucket, xtsy)
            loss_neg_sum = (-torch.log(1 - neg_logits[valid_mask].sigmoid() + 1e-24)).sum()
        else:
            neg_items, neg_buckets = sample_negs_k(data, self.args, num_negs)
            for idx_neg, neg in enumerate(neg_items):
                neg = neg.to(self.args.device)
                neg_bucket = None
                if neg_buckets is not None:
                    neg_bucket = neg_buckets[idx_neg].to(self.args.device)
                neg_logits = self._score_items(feats, neg, neg_bucket, xtsy)
                loss_neg_sum = loss_neg_sum + (-torch.log(1 - neg_logits[valid_mask].sigmoid() + 1e-24)).sum()

        loss = loss_pos + (loss_neg_sum / max(1, num_negs))
        log_and_accumulate_train_debug(
            self, self.__class__.__name__, scope, inputs, valid_mask, pos, rep_mask_batch, batch_mask
        )
        return loss

    def compute_rank(self, data: torch.Tensor, store: dict, k: int = 10, **kwargs) -> dict:
        inputs = data[:, :, 3]
        pos = data[:, :, 5]
        xtsy = data[:, :, 6]
        inputs_ts = data[:, :, 2]
        bucket_inputs = None
        if self.use_bucket:
            idx_in = getattr(self.args, "bucket_input_idx", None)
            if idx_in is not None and idx_in < data.shape[2]:
                bucket_inputs = data[:, :, idx_in]

        feats = self._encode(inputs, bucket_inputs)
        detail_list = store.get("rank_details", None)
        include_seq = bool(detail_list is not None and getattr(self.args, "rank_dump_include_seq", False))
        include_hits = bool(detail_list is not None and getattr(self.args, "rank_dump_include_hits", False))
        miss_topk = int(getattr(self.args, "rank_dump_miss_topk", 0) or 0)
        track_new_ratio = bool(getattr(self.args, "track_new_ratio", False))

        mask = _compute_repeat_mask_vectorized(pos, store)

        candidates = kwargs.get("candidates")
        if candidates is None:
            candidates = build_step_candidates(self.args, xtsy)
        step_candidates = candidates or {}

        steps = xtsy[:, -1].cpu().numpy()
        unique_steps = {}
        for b, step in enumerate(steps):
            step = int(step)
            unique_steps.setdefault(step, []).append(b)

        step_embs: Dict[int, torch.Tensor] = {}
        step_av: Dict[int, torch.Tensor] = {}
        for step in unique_steps.keys():
            cand = step_candidates.get(step)
            if cand is None:
                continue
            av = cand.items
            if av.numel() == 0:
                continue
            av_embs = self.item_embedding(av)
            if self.use_bucket and cand.buckets is not None:
                av_embs = av_embs + self.bucket_emb(cand.buckets[: av_embs.shape[0]])
            step_embs[step] = av_embs
            step_av[step] = av

        for b in range(inputs.shape[0]):
            step = int(xtsy[b, -1].item())
            if step not in step_embs:
                continue
            av_embs = step_embs[step]
            av = step_av[step]

            scores = self._score_candidates(
                feats[b, -1, :],
                av,
                step_candidates[step].buckets if step in step_candidates else None,
                step,
                item_embs=av_embs,
            )
            order = torch.argsort(scores, descending=True)

            rep_flags = None
            new_flags = None
            if track_new_ratio or detail_list is not None:
                rep_flags = _repeat_flags_for_av(inputs[b], av)
                new_flags = ~rep_flags if rep_flags is not None else None

            if track_new_ratio and rep_flags is not None:
                top_idx = order[:k] if order.numel() >= k else order
                if top_idx.numel() > 0:
                    new_ratio = new_flags[top_idx].to(torch.float32).mean().item()
                    store.setdefault("new_ratio_topk", []).append(float(new_ratio))

            iseq = pos[b, -1] == av
            idx = torch.where(iseq)[0]
            if idx.numel() == 0:
                continue
            rank = torch.where(order == idx)[0].item()

            _update_tier_stats(store, self.args, int(pos[b, -1].item()), rank, bool(mask[b]))
            if mask[b]:
                store["rrep"] += [rank]
            else:
                store["rnew"] += [rank]
            store["rall"] += [rank]

            if detail_list is not None:
                is_hit = rank == 0
                should_log = True
                if miss_topk > 0 and rank < miss_topk and not (include_hits and is_hit):
                    should_log = False
                if should_log:
                    target_idx = idx.item()
                    rep_rank = None
                    new_rank = None
                    rep_count = int(rep_flags.sum().item()) if rep_flags is not None else 0
                    new_count = int(new_flags.sum().item()) if new_flags is not None else 0
                    if rep_flags is not None and rep_flags.numel() > 0:
                        if rep_flags[target_idx]:
                            rep_order = order[rep_flags[order]]
                            rep_pos = torch.where(rep_order == target_idx)[0]
                            if rep_pos.numel() > 0:
                                rep_rank = int(rep_pos.item())
                        else:
                            new_order = order[new_flags[order]]
                            new_pos = torch.where(new_order == target_idx)[0]
                            if new_pos.numel() > 0:
                                new_rank = int(new_pos.item())
                    detail_rec = {
                        "user_id": _extract_user_id(data[b, :, 4]),
                        "target_item": int(pos[b, -1].item()),
                        "target_step": int(xtsy[b, -1].item()),
                        "is_repeat": bool(mask[b].item()),
                        "overall_rank": int(rank),
                        "rep_rank": rep_rank,
                        "new_rank": new_rank,
                        "total_candidates": int(av.shape[0]),
                        "rep_candidates": rep_count,
                        "new_candidates": new_count,
                        "hit1": rank < 1,
                        "hit5": rank < 5,
                        "hit10": rank < 10,
                        "history_items": None,
                        "history_steps": None,
                    }
                    if include_seq:
                        seq_mask = inputs[b] != 0
                        if seq_mask.any():
                            hist_items = inputs[b, seq_mask].tolist()
                            hist_steps = inputs_ts[b, seq_mask].tolist()
                            detail_rec["history_items"] = [int(v) for v in hist_items]
                            detail_rec["history_steps"] = [int(v) for v in hist_steps]
                    detail_list.append(detail_rec)

        return store
