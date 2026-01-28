"""BERT4Rec baseline (masked item prediction)."""

from __future__ import annotations

import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.candidates import build_step_candidates
from models.baselines.liverec import (
    Attention,
    _compute_repeat_mask_vectorized,
    _extract_user_id,
    _repeat_flags_for_av,
    _update_tier_stats,
)
from models.baselines.train_debug import log_and_accumulate_train_debug


class BERT4Rec(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.viewer_mode = getattr(args, "viewer_feat_mode", "off")

        self.mask_token = args.N + 1
        self.item_embedding = nn.Embedding(args.N + 2, args.K, padding_idx=0)
        self.pos_emb = nn.Embedding(args.seq_len + 1, args.K)
        self.emb_dropout = nn.Dropout(p=0.2)
        self.mask_prob = float(getattr(args, "bert_mask_prob", 0.2) or 0.2)
        self.ce_over_av = bool(getattr(args, "bert_ce_over_av", True))

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

        self.att = Attention(args, args.num_att, args.num_heads, causality=False)

    def _extend_sequence(
        self, inputs: torch.Tensor, bucket_inputs: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bs = inputs.shape[0]
        mask_col = torch.full(
            (bs, 1), self.mask_token, device=inputs.device, dtype=inputs.dtype
        )
        seq = torch.cat([inputs, mask_col], dim=1)
        bucket_seq = None
        if self.use_bucket and bucket_inputs is not None:
            pad_bucket = torch.zeros(
                (bs, 1), device=bucket_inputs.device, dtype=bucket_inputs.dtype
            )
            bucket_seq = torch.cat([bucket_inputs, pad_bucket], dim=1)
        return seq, bucket_seq

    def _encode(
        self, log_seqs: torch.Tensor, bucket_seqs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        seqs = self.item_embedding(log_seqs)
        seqs *= self.item_embedding.embedding_dim**0.5
        if self.use_bucket and bucket_seqs is not None:
            seqs = seqs + self.bucket_emb(bucket_seqs)

        positions = torch.arange(log_seqs.shape[1], device=self.args.device)
        positions = positions.unsqueeze(0).expand(log_seqs.shape[0], -1)
        seqs = seqs + self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)
        timeline_mask = (log_seqs == 0).to(self.args.device)
        return self.att(seqs, timeline_mask)

    def _score_items(
        self,
        user_feat: torch.Tensor,
        items: torch.Tensor,
        timestamps: torch.Tensor,
        bucket_items: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        item_embs = self.item_embedding(items)
        if self.use_bucket and bucket_items is not None:
            item_embs = item_embs + self.bucket_emb(bucket_items)
        return (user_feat * item_embs).sum(dim=-1)

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

    def _sample_negs_for_positions(
        self, pos_items: torch.Tensor, ts: torch.Tensor, num_negs: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        pos_list = pos_items.detach().cpu().tolist()
        ts_list = ts.detach().cpu().tolist()
        negs = torch.zeros(
            (len(pos_list), num_negs),
            dtype=torch.long,
            device=pos_items.device,
        )
        neg_buckets = (
            torch.zeros_like(negs) if self.use_bucket else None
        )
        for idx, (p, t) in enumerate(zip(pos_list, ts_list)):
            av = getattr(self.args, "ts", {}).get(int(t), [])
            buckets = None
            if self.use_bucket:
                buckets = getattr(self.args, "ts_bucket", {}).get(int(t), [])
            if not av:
                for k in range(num_negs):
                    negs[idx, k] = random.randint(1, self.args.N)
                    if neg_buckets is not None:
                        neg_buckets[idx, k] = 0
                continue
            if len(av) == 1 and av[0] == p:
                for k in range(num_negs):
                    negs[idx, k] = random.randint(1, self.args.N)
                    if neg_buckets is not None:
                        neg_buckets[idx, k] = 0
                continue
            for k in range(num_negs):
                while True:
                    ridx = random.randrange(len(av))
                    cand = int(av[ridx])
                    if cand != p:
                        negs[idx, k] = cand
                        if neg_buckets is not None and buckets and ridx < len(buckets):
                            neg_buckets[idx, k] = int(buckets[ridx])
                        elif neg_buckets is not None:
                            neg_buckets[idx, k] = 0
                        break
        return negs, neg_buckets

    def _av_ce_loss(
        self,
        feats: torch.Tensor,
        pos_items: torch.Tensor,
        pos_ts: torch.Tensor,
    ) -> torch.Tensor:
        if feats.numel() == 0:
            return torch.zeros((), device=self.args.device)

        step_candidates = build_step_candidates(self.args, pos_ts)
        unique_steps = torch.unique(pos_ts.detach().cpu())
        total_loss = torch.zeros((), device=self.args.device)
        total_count = 0

        for step_val in unique_steps.tolist():
            step = int(step_val)
            cand = step_candidates.get(step)
            if cand is None:
                continue
            av = cand.items
            if av.numel() == 0:
                continue
            mask = pos_ts == step
            if not mask.any():
                continue
            step_feats = feats[mask]
            step_pos = pos_items[mask]
            if step_feats.numel() == 0:
                continue

            av_embs = self.item_embedding(av)
            if self.use_bucket and cand.buckets is not None:
                av_embs = av_embs + self.bucket_emb(cand.buckets[: av_embs.shape[0]])

            scores = torch.matmul(step_feats, av_embs.transpose(0, 1))
            scores = self._maybe_add_candidate_bias(scores, step_feats, av, step, cand.buckets)
            match = step_pos.unsqueeze(1) == av.unsqueeze(0)
            has_match = match.any(dim=1)
            if not has_match.any():
                continue
            target_idx = match.float().argmax(dim=1)
            total_loss = total_loss + F.cross_entropy(
                scores[has_match], target_idx[has_match], reduction="sum"
            )
            total_count += int(has_match.sum().item())

        if total_count == 0:
            return torch.zeros((), device=self.args.device)
        return total_loss / total_count

    def _maybe_add_candidate_bias(
        self,
        scores: torch.Tensor,
        step_feats: torch.Tensor,
        items: torch.Tensor,
        step: int,
        bucket_items: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return scores

    def train_step(self, data: torch.Tensor) -> torch.Tensor:
        inputs, pos = data[:, :, 3], data[:, :, 5]
        xtsy = data[:, :, 6]
        inputs_ts = data[:, :, 2]
        bucket_inputs = None
        bucket_targets = None
        if self.use_bucket:
            idx_in = getattr(self.args, "bucket_input_idx", None)
            idx_tg = getattr(self.args, "bucket_target_idx", None)
            if idx_in is not None and idx_in < data.shape[2]:
                bucket_inputs = data[:, :, idx_in]
            if idx_tg is not None and idx_tg < data.shape[2]:
                bucket_targets = data[:, :, idx_tg]

        mask_positions = (
            (inputs != 0)
            & (torch.rand(inputs.shape, device=inputs.device) < self.mask_prob)
        )
        seq, bucket_seq = self._extend_sequence(inputs, bucket_inputs)
        seq[:, :-1][mask_positions] = self.mask_token

        labels = torch.zeros_like(seq)
        labels[:, :-1][mask_positions] = inputs[mask_positions]
        labels[:, -1] = pos[:, -1]

        label_ts = torch.zeros_like(seq)
        label_ts[:, :-1] = inputs_ts
        label_ts[:, -1] = xtsy[:, -1]

        label_buckets = None
        if self.use_bucket and bucket_inputs is not None and bucket_targets is not None:
            label_buckets = torch.zeros_like(seq)
            label_buckets[:, :-1] = bucket_inputs
            label_buckets[:, -1] = bucket_targets[:, -1]

        label_mask = labels != 0
        if not label_mask.any():
            return torch.zeros((), device=self.args.device)

        feats = self._encode(seq, bucket_seq)
        feat_masked = feats[label_mask]
        pos_items = labels[label_mask]
        pos_ts = label_ts[label_mask]
        if self.ce_over_av:
            loss = self._av_ce_loss(feat_masked, pos_items, pos_ts)
        else:
            pos_bucket = label_buckets[label_mask] if label_buckets is not None else None
            pos_logits = self._score_items(feat_masked, pos_items, pos_ts, pos_bucket)

            num_negs = int(getattr(self.args, "num_negs", 1) or 1)
            neg_items, neg_buckets = self._sample_negs_for_positions(
                pos_items, pos_ts, num_negs
            )
            neg_embs = self.item_embedding(neg_items)
            if self.use_bucket and neg_buckets is not None:
                neg_embs = neg_embs + self.bucket_emb(neg_buckets)
            neg_logits = (feat_masked.unsqueeze(1) * neg_embs).sum(dim=-1)

            loss_pos = (-torch.log(pos_logits.sigmoid() + 1e-24)).sum()
            loss_neg = (-torch.log(1 - neg_logits.sigmoid() + 1e-24)).sum()
            loss = loss_pos + (loss_neg / max(1, num_negs))

        log_and_accumulate_train_debug(
            self,
            self.__class__.__name__,
            getattr(self.args, "train_scope", "all"),
            inputs,
            mask_positions,
            pos,
            None,
            None,
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

        seq, bucket_seq = self._extend_sequence(inputs, bucket_inputs)
        feats = self._encode(seq, bucket_seq)
        user_feats = feats[:, -1, :]

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
        unique_steps: Dict[int, list] = {}
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
            cand_buckets = step_candidates[step].buckets if step in step_candidates else None

            scores = self._score_candidates(
                user_feats[b],
                av,
                cand_buckets,
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
