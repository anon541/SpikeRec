"""Spike-aware BERT4Rec variant."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.baselines.bert4rec import BERT4Rec
from models.baselines.liverec import _vectorized_spike_cache_lookup

FEATURE_DIM = 4


class BERT4RecSpike(BERT4Rec):
    def __init__(self, args):
        super().__init__(args)
        self.trend_integration_mode = str(
            getattr(args, "trend_integration_mode", "delta") or "delta"
        )
        self.use_percentile_features = bool(getattr(args, "use_percentile_features", False))
        self.use_percentile_3d = bool(getattr(args, "use_percentile_3d", False))
        self.use_hybrid_features = bool(getattr(args, "use_hybrid_features", False))
        self.ablation_mask_global = bool(getattr(args, "ablation_mask_global", False))
        self.ablation_mask_self = bool(getattr(args, "ablation_mask_self", False))
        self.ablation_mask_window = bool(getattr(args, "ablation_mask_window", False))
        self.ablation_mask_user_repr = bool(getattr(args, "ablation_mask_user_repr", False))

        hidden_dim = max(16, int(self.args.K / 2))
        self.spike_head = nn.Sequential(
            nn.Linear(self.args.K + FEATURE_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.spike_bias_weight = float(getattr(args, "spike_bias_weight", 1.0) or 1.0)

        self._spike_feature_cache: Dict[Tuple[str, int], torch.Tensor] = {}
        self._zero_feature = torch.zeros(FEATURE_DIM, dtype=torch.float32)

    def _get_streamer_name(self, item_id: int) -> Tuple[str, bool]:
        id_to_streamer = getattr(self.args, "id_to_streamer", None)
        if id_to_streamer is None:
            return "", False
        streamer = id_to_streamer.get(item_id)
        if streamer is None:
            return "", False
        return streamer, True

    def _build_feature_from_trends(self, streamer: str, timestamp: int) -> torch.Tensor:
        viewer_trends = getattr(self.args, "viewer_trends", None)
        if viewer_trends is None:
            return self._zero_feature.clone()

        if self.use_percentile_3d:
            g_rank, s_rank, w_rank = viewer_trends.compute_spike_features(
                streamer, timestamp, use_percentile_3d=True
            )
            if self.ablation_mask_global:
                g_rank = 0.0
            if self.ablation_mask_self:
                s_rank = 0.0
            if self.ablation_mask_window:
                w_rank = 0.0
            return torch.tensor([g_rank, s_rank, w_rank, 0.0], dtype=torch.float32)

        if self.use_percentile_features:
            g_rank, s_rank, ratio, conf = viewer_trends.compute_spike_features(
                streamer, timestamp, use_percentile=True
            )
            if self.ablation_mask_global:
                g_rank = 0.0
            if self.ablation_mask_self:
                s_rank = 0.0
            return torch.tensor([g_rank, s_rank, ratio, conf], dtype=torch.float32)

        if self.use_hybrid_features:
            _, s_rank, ratio, conf = viewer_trends.compute_spike_features(
                streamer, timestamp, use_percentile=True
            )
            if self.ablation_mask_self:
                s_rank = 0.0
            viewer_count = viewer_trends.get_viewer_count(streamer, timestamp)
            log_count = math.log1p(max(viewer_count, 0))
            return torch.tensor([log_count, s_rank, ratio, conf], dtype=torch.float32)

        viewer_z, viewer_ratio, viewer_conf = viewer_trends.compute_spike_features(
            streamer, timestamp
        )
        viewer_count = viewer_trends.get_viewer_count(streamer, timestamp)
        log_count = math.log1p(max(viewer_count, 0))
        return torch.tensor([viewer_z, viewer_ratio, log_count, viewer_conf], dtype=torch.float32)

    def _cached_spike_feature(self, streamer: str, timestamp: int) -> torch.Tensor:
        key = (streamer, timestamp)
        cached = self._spike_feature_cache.get(key)
        if cached is not None:
            return cached
        vec = self._build_feature_from_trends(streamer, timestamp)
        self._spike_feature_cache[key] = vec
        return vec

    def _spike_features_for_candidates(
        self, items: torch.Tensor, step: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        cache_feats = getattr(self, "validation_spike_cache_feats", None)
        cache_items = getattr(self, "validation_spike_cache_items", None)
        cache_steps = getattr(self, "validation_spike_cache_steps", None)
        if cache_feats is not None and cache_items is not None and cache_steps is not None:
            class _CacheView:
                def __init__(self, feats, items, steps):
                    self.feats = feats
                    self.items = items
                    self.steps = steps

                def __len__(self):
                    return int(self.feats.shape[0])

            sc = _CacheView(cache_feats, cache_items, cache_steps)
            spike_feats_tensor, valid_indices = _vectorized_spike_cache_lookup(
                items, step, sc, items.device
            )
            if spike_feats_tensor is not None:
                idx_tensor = torch.tensor(valid_indices, dtype=torch.long, device=items.device)
                return spike_feats_tensor, idx_tensor
        return None, None

    def _spike_bias_for_items(
        self, items: torch.Tensor, steps: torch.Tensor, user_repr: torch.Tensor
    ) -> torch.Tensor:
        device = items.device
        flat_items = items.reshape(-1)
        flat_steps = steps.reshape(-1)
        flat_repr = user_repr.reshape(-1, user_repr.shape[-1])
        mask = flat_items != 0
        if not mask.any():
            return torch.zeros_like(items, dtype=torch.float32, device=device)

        feats = []
        reps = []
        for item_id, step, rep in zip(
            flat_items[mask].tolist(), flat_steps[mask].tolist(), flat_repr[mask]
        ):
            streamer, ok = self._get_streamer_name(int(item_id))
            if not ok:
                feats.append(self._zero_feature.clone())
            else:
                feats.append(self._cached_spike_feature(streamer, int(step)))
            reps.append(rep)

        feat_tensor = torch.stack(feats, dim=0).to(device)
        rep_tensor = torch.stack(reps, dim=0)
        if self.ablation_mask_user_repr:
            rep_tensor = torch.zeros_like(rep_tensor)
        spike_in = torch.cat([rep_tensor, feat_tensor], dim=-1)
        bias = self.spike_head(spike_in).squeeze(-1) * self.spike_bias_weight

        out = torch.zeros(flat_items.shape[0], dtype=torch.float32, device=device)
        out[mask] = bias
        return out.view_as(items)

    def _spike_bias_for_candidates(
        self, items: torch.Tensor, step: int, user_repr: torch.Tensor
    ) -> Optional[torch.Tensor]:
        feat_tensor, idx_tensor = self._spike_features_for_candidates(items, step)
        if feat_tensor is None or idx_tensor is None:
            return None
        user = user_repr
        if self.ablation_mask_user_repr:
            user = torch.zeros_like(user)
        user = user.unsqueeze(0).expand(feat_tensor.shape[0], -1)
        spike_in = torch.cat([user, feat_tensor], dim=-1)
        bias = self.spike_head(spike_in).squeeze(-1) * self.spike_bias_weight
        scores = torch.zeros(items.shape[0], device=items.device)
        scores[idx_tensor] = bias
        return scores

    def _get_spike_bias_for_items(
        self, items: torch.Tensor, step: int, raw_score: bool = False
    ) -> torch.Tensor:
        if items.numel() == 0:
            return torch.zeros(0, device=items.device, dtype=torch.float32)
        user_repr = torch.zeros(self.args.K, device=items.device)
        bias = self._spike_bias_for_candidates(items, step, user_repr)
        if bias is None:
            bias = torch.zeros(items.shape[0], device=items.device, dtype=torch.float32)
        bias = bias.detach()
        if raw_score and self.spike_bias_weight:
            return bias / float(self.spike_bias_weight)
        return bias

    def _score_items(
        self,
        user_feat: torch.Tensor,
        items: torch.Tensor,
        timestamps: torch.Tensor,
        bucket_items: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scores = super()._score_items(user_feat, items, timestamps, bucket_items)
        if self.trend_integration_mode == "delta":
            scores = scores + self._spike_bias_for_items(items, timestamps, user_feat)
        return scores

    def _score_candidates(
        self,
        user_feat: torch.Tensor,
        items: torch.Tensor,
        bucket_items: Optional[torch.Tensor],
        step: int,
        item_embs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scores = super()._score_candidates(
            user_feat, items, bucket_items, step, item_embs=item_embs
        )
        if self.trend_integration_mode == "delta":
            bias = self._spike_bias_for_candidates(items, step, user_feat)
            if bias is not None:
                scores = scores + bias
        return scores

    def _maybe_add_candidate_bias(
        self,
        scores: torch.Tensor,
        step_feats: torch.Tensor,
        items: torch.Tensor,
        step: int,
        bucket_items: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.trend_integration_mode != "delta":
            return scores
        feat_tensor, idx_tensor = self._spike_features_for_candidates(items, step)
        if feat_tensor is None or idx_tensor is None:
            return scores
        user = step_feats
        if self.ablation_mask_user_repr:
            user = torch.zeros_like(user)
        user_expanded = user.unsqueeze(1).expand(-1, feat_tensor.shape[0], -1)
        feat_expanded = feat_tensor.unsqueeze(0).expand(user.shape[0], -1, -1)
        spike_in = torch.cat([user_expanded, feat_expanded], dim=-1)
        bias_valid = self.spike_head(spike_in).squeeze(-1) * self.spike_bias_weight
        bias_full = torch.zeros(
            user.shape[0], items.shape[0], device=items.device, dtype=bias_valid.dtype
        )
        bias_full[:, idx_tensor] = bias_valid
        return scores + bias_full
