import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter
from data.processing.sampling import *
from models.baselines.scope_utils import scope_masks
from models.baselines.train_debug import log_and_accumulate_train_debug
from data.candidates import build_step_candidates
import math


# ============================================================================
# Common validation optimization functions (EP-INF-008)
# ============================================================================

def _sample_validation_candidates(av, pos_item, val_sample_candidates, device):
    """
    Sample candidates for faster validation (EP-INF-007).
    Always includes target item if available.
    
    Args:
        av: Available items tensor
        pos_item: Target item
        val_sample_candidates: Number of candidates to sample (0 = use all)
        device: PyTorch device
    
    Returns:
        Sampled available items tensor
    """
    if val_sample_candidates <= 0 or av.numel() <= val_sample_candidates:
        return av
    
    target_in_av = (av == pos_item).any()
    
    if target_in_av:
        # Include target + random sample of others
        non_target_mask = av != pos_item
        non_target_av = av[non_target_mask]
        
        if non_target_av.numel() > val_sample_candidates - 1:
            sample_indices = torch.randperm(non_target_av.numel(), device=device)[:val_sample_candidates-1]
            sampled_av = non_target_av[sample_indices]
            return torch.cat([pos_item.unsqueeze(0), sampled_av])
        else:
            return torch.cat([pos_item.unsqueeze(0), non_target_av])
    else:
        # Target not available (offline), random sample
        if av.numel() > val_sample_candidates:
            sample_indices = torch.randperm(av.numel(), device=device)[:val_sample_candidates]
            return av[sample_indices]
        return av


def _sample_av_with_embeddings(av, av_embs, pos_item, val_sample_candidates, device):
    """
    Candidate sampling + embedding subset 반환 (validation 최적화용).
    기존 _sample_validation_candidates는 embedding과 동기화되지 않아 재계산 필요가 있었음.
    """
    if val_sample_candidates <= 0 or av.numel() <= val_sample_candidates:
        return av, av_embs

    target_in_av = (av == pos_item).any()
    if target_in_av:
        non_target_mask = av != pos_item
        non_target_av = av[non_target_mask]
        non_target_embs = av_embs[non_target_mask]

        if non_target_av.numel() > val_sample_candidates - 1:
            sample_indices = torch.randperm(non_target_av.numel(), device=device)[:val_sample_candidates-1]
            sampled_av = non_target_av[sample_indices]
            sampled_embs = non_target_embs[sample_indices]
        else:
            sampled_av = non_target_av
            sampled_embs = non_target_embs
        av = torch.cat([pos_item.unsqueeze(0), sampled_av])
        av_embs = torch.cat([av_embs[~non_target_mask][:1], sampled_embs])
        return av, av_embs

    # Target이 없으면 random sample
    sample_indices = torch.randperm(av.numel(), device=device)[:val_sample_candidates]
    return av[sample_indices], av_embs[sample_indices]


def _compute_repeat_mask_vectorized(pos, store=None):
    """
    Vectorized computation of repeat mask (EP-INF-006 optimization).
    Checks if target item appears in history (repeat vs new).
    
    Args:
        pos: Position tensor [B, seq_len] where pos[:,-1] is target
        store: Optional store dict to append ratio values
    
    Returns:
        mask: [B] bool tensor (True if target is in history = repeat)
    """
    pos_targets = pos[:,-1]  # [B]
    pos_history = pos[:,:-1]  # [B, seq_len-1]
    pos_targets_expanded = pos_targets.unsqueeze(1)  # [B, 1]
    history_mask = (pos_history != 0)  # [B, seq_len-1]
    matches = (pos_targets_expanded == pos_history) & history_mask  # [B, seq_len-1]
    mask = matches.any(dim=1)  # [B]
    
    if store is not None:
        # Store ratios (vectorized)
        ratios = mask.float().cpu().numpy().tolist()
        store['ratio'].extend(ratios)
    
    return mask


def _safe_find_rank(pos_item, av, order):
    """
    Safely find rank of target item (EP-INF-007).
    Returns None if target not found (offline case).
    
    Args:
        pos_item: Target item
        av: Available items
        order: Sorted indices
    
    Returns:
        rank (int) or None
    """
    iseq = pos_item == av
    idx = torch.where(iseq)[0]
    if idx.numel() == 0:
        return None  # Target not available
    
    rank_in_av = torch.where(order == idx)[0]
    if rank_in_av.numel() == 0:
        return None  # Should not happen
    
    return rank_in_av.item()


def _vectorized_spike_cache_lookup(av_topk, step, spike_cache, device):
    """
    Vectorized lookup of spike features from pre-computed cache (EP-INF-007).
    
    Args:
        av_topk: Top-k candidate items
        step: Current timestep
        spike_cache: Pre-computed spike features cache dict
        device: PyTorch device
    
    Returns:
        (spike_feats_tensor, valid_indices) or (None, [])
    """
    if spike_cache is None or len(spike_cache) == 0:
        return None, []

    # Fast path: tensorized cache (set in eval.py) - FULLY VECTORIZED
    cache_feats = getattr(spike_cache, "feats", None)
    cache_items = getattr(spike_cache, "items", None)
    cache_steps = getattr(spike_cache, "steps", None)
    if cache_feats is not None and cache_items is not None and cache_steps is not None:
        step_mask = cache_steps == step
        if not step_mask.any():
            return None, []
        items_step = cache_items[step_mask]
        feats_step = cache_feats[step_mask]
        
        # Vectorized lookup: use broadcasting instead of dict lookup
        # av_topk: [K], items_step: [M] -> find matches
        av_topk_expanded = av_topk.unsqueeze(1)  # [K, 1]
        items_step_expanded = items_step.unsqueeze(0)  # [1, M]
        matches = (av_topk_expanded == items_step_expanded)  # [K, M]
        
        # Find valid indices: which items in av_topk have matches
        valid_mask = matches.any(dim=1)  # [K]
        if not valid_mask.any():
            return None, []
        
        valid_indices = torch.where(valid_mask)[0].tolist()  # [num_valid]
        
        # For each valid item, get the first matching position in items_step
        # argmax on CUDA does not support bool, so cast before reduction
        match_positions = matches[valid_mask].to(torch.int64).argmax(dim=1)  # [num_valid]
        gather_tensor = match_positions.to(device)
        
        return feats_step[gather_tensor], valid_indices

    # Fallback: dict lookup
    item_ids_np = av_topk.cpu().numpy()
    spike_feats_list = []
    valid_indices = []
    for idx, item_id_int in enumerate(item_ids_np):
        cache_key = (int(item_id_int), step)
        if cache_key in spike_cache:
            spike_feats_list.append(spike_cache[cache_key])
            valid_indices.append(idx)
    if not spike_feats_list:
        return None, []
    spike_feats_tensor = torch.tensor(spike_feats_list, device=device, dtype=torch.float32)
    return spike_feats_tensor, valid_indices


def _build_step_av_embeddings(args, xtsy, item_embedding, bucket_emb=None, use_bucket=False):
    """
    timestep별 available candidates와 embedding을 미리 계산.
    반환: {step: (av_tensor, av_embs_tensor)}
    """
    steps = xtsy[:, -1].cpu().numpy()
    unique_steps = {}
    for b, step in enumerate(steps):
        step = int(step)
        unique_steps.setdefault(step, []).append(b)

    step_map = {}
    for step in unique_steps.keys():
        av_list = args.ts.get(step, [])
        if len(av_list) == 0:
            continue
        if hasattr(args, 'av_tens') and args.av_tens is not None and step < args.av_tens.shape[0]:
            av = args.av_tens[step]
            av = av[av != 0]
        else:
            av = torch.LongTensor(av_list).to(args.device)
        if av.numel() == 0:
            continue

        av_embs = item_embedding(av)
        if use_bucket and bucket_emb is not None:
            bucket_vals = getattr(args, 'ts_bucket', {}).get(step, [])
            if len(bucket_vals) > 0:
                if hasattr(args, 'av_bucket_tens') and args.av_bucket_tens is not None and step < args.av_bucket_tens.shape[0]:
                    av_bucket = args.av_bucket_tens[step]
                    av_bucket = av_bucket[av_bucket != 0]
                else:
                    av_bucket = torch.LongTensor(bucket_vals).to(args.device)
                if av_bucket.shape[0] < av_embs.shape[0]:
                    pad = torch.zeros(av_embs.shape[0]-av_bucket.shape[0], dtype=torch.long, device=args.device)
                    av_bucket = torch.cat([av_bucket, pad], dim=0)
                av_embs = av_embs + bucket_emb(av_bucket[:av_embs.shape[0]])
        step_map[step] = (av, av_embs)
    return step_map


def _get_spike_feats_from_cache_or_trends(model, av_subset, step):
    """
    캐시가 있으면 캐시 사용, 없으면 viewer_trends에서 batch 계산.
    """
    spike_cache = getattr(model, 'validation_spike_cache', None)
    cache_feats = getattr(model, 'validation_spike_cache_feats', None)
    cache_items = getattr(model, 'validation_spike_cache_items', None)
    cache_steps = getattr(model, 'validation_spike_cache_steps', None)
    if cache_feats is not None and cache_items is not None and cache_steps is not None:
        # wrap into a lightweight object so _vectorized_spike_cache_lookup can use tensors
        class _CacheView:
            pass
        sc = _CacheView()
        sc.feats = cache_feats
        sc.items = cache_items
        sc.steps = cache_steps
        spike_cache = sc
    if spike_cache:
        spike_feats_tensor, valid_indices = _vectorized_spike_cache_lookup(av_subset, step, spike_cache, model.args.device)
        if spike_feats_tensor is not None:
            return spike_feats_tensor, valid_indices

    viewer_trends = getattr(model.args, 'viewer_trends', None)
    id_to_streamer = getattr(model.args, 'id_to_streamer', None)
    if viewer_trends is None or id_to_streamer is None:
        return None, []

    streamer_timestamp_pairs = []
    valid_indices = []
    for idx, item_id in enumerate(av_subset):
        streamer_name = id_to_streamer.get(int(item_id.item()), None)
        if streamer_name:
            streamer_timestamp_pairs.append((streamer_name, step))
            valid_indices.append(idx)

    if not streamer_timestamp_pairs:
        return None, []

    spike_feats_array = viewer_trends.compute_spike_features_batch(
        streamer_timestamp_pairs,
        use_percentile=bool(getattr(model.args, "use_percentile_features", False)),
        use_percentile_3d=bool(getattr(model.args, "use_percentile_3d", False)),
        use_hybrid=bool(getattr(model.args, "use_hybrid_features", False)),
        window_size=getattr(model.args, "ablation_window_size", None),
    )
    spike_feats_tensor = torch.from_numpy(spike_feats_array).to(model.args.device)
    return spike_feats_tensor, valid_indices


VALIDATION_TIMESTAMP_LOG = Path("validation_timestamps.log")


def _log_validation_timestamp_in_model(args, message, elapsed=None, step=None, batch_idx=None):
    """Validation 단계에서만 timestamp를 로깅하는 함수 (모델 내부에서 사용)"""
    # args._validation_phase가 존재하고 'val'로 시작할 때만 로깅
    if hasattr(args, '_validation_phase') and args._validation_phase and str(args._validation_phase).lower().startswith('val'):
        phase = args._validation_phase
        timestamp = datetime.utcnow().isoformat()
        log_line = f"[{timestamp}] [compute_rank]"
        if batch_idx is not None:
            log_line += f" [batch {batch_idx}]"
        if step is not None:
            log_line += f" [step {step}]"
        log_line += f" {message}"
        if elapsed is not None:
            log_line += f" (elapsed: {elapsed:.4f}s)"
        
        # 콘솔에도 출력 (사용자 요청)
        print(log_line)

        log_line += "\n"
        
        # 현재 디렉토리에 로그 파일 생성 (사용자 선호사항)
        try:
            with open(VALIDATION_TIMESTAMP_LOG, 'a') as f:
                f.write(log_line)
        except Exception:
            pass  # 로깅 실패해도 실행은 계속


def count_bucket_states(log_seqs: torch.Tensor) -> torch.Tensor:
    """Return per-position bucket ids: 0 pad/new, 1 seen once, 2 seen >=2."""
    mask = log_seqs != 0
    if not mask.any():
        return torch.zeros_like(log_seqs)

    seq_len = log_seqs.shape[1]
    device = log_seqs.device
    eq = (log_seqs.unsqueeze(2) == log_seqs.unsqueeze(1))
    eq = eq & mask.unsqueeze(2) & mask.unsqueeze(1)
    tril = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
    counts = (eq & tril.unsqueeze(0)).sum(dim=2)

    state = torch.zeros_like(log_seqs)
    state[mask & (counts == 1)] = 1
    state[mask & (counts >= 2)] = 2
    return state


def _extract_user_id(user_seq: torch.Tensor) -> int:
    """Return the last non-zero user id in the sequence (or 0 if padded)."""
    if user_seq is None:
        return 0
    nz = user_seq[user_seq != 0]
    if nz.numel() == 0:
        return 0
    return int(nz[-1].item())


def _repeat_flags_for_av(inputs_row: torch.Tensor, av: torch.Tensor) -> torch.Tensor:
    """Boolean mask over availability set indicating repeat items."""
    if av.numel() == 0:
        return torch.zeros(av.shape[0], dtype=torch.bool, device=av.device)
    nonzero_inputs = inputs_row[inputs_row != 0]
    if nonzero_inputs.numel() == 0:
        return torch.zeros(av.shape[0], dtype=torch.bool, device=av.device)
    return (nonzero_inputs.unsqueeze(1) == av.unsqueeze(0)).any(dim=0)


def ndcg_k(target_item: torch.Tensor, topk_items: torch.Tensor, topk_scores: torch.Tensor, k: int) -> float:
    """Compute NDCG@k for a single target item."""
    if topk_items.numel() == 0:
        return 0.0
    # Find rank of target item in topk
    matches = (topk_items == target_item.item())
    if not matches.any():
        return 0.0
    rank = matches.nonzero(as_tuple=True)[0].item()
    # NDCG@k = 1 / log2(rank + 2) if rank < k, else 0
    if rank < k:
        return 1.0 / math.log2(rank + 2)
    return 0.0


def _update_tier_stats(store, args, item_id, rank, is_repeat):
    tier_map = getattr(args, "streamer_tier", None)
    if not tier_map:
        return
    tier = tier_map.get(int(item_id), None)
    if tier is None:
        return
    tier_store = store.setdefault("tier_stats", {})
    stats = tier_store.setdefault(tier, {"all": [], "rep": [], "new": []})
    stats["all"].append(rank)
    if is_repeat:
        stats["rep"].append(rank)
    else:
        stats["new"].append(rank)


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class Attention(nn.Module):
    def __init__(self, args, num_att, num_heads, causality=False):
        super(Attention, self).__init__()
        self.args = args
        self.causality = causality

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(args.K, eps=1e-8)

        for _ in range(num_att):
            new_attn_layernorm = nn.LayerNorm(args.K, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = nn.MultiheadAttention(args.K,
                                                    num_heads,
                                                    0.2)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = nn.LayerNorm(args.K, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.K, 0.2)
            self.forward_layers.append(new_fwd_layer)

    def forward(self, seqs, timeline_mask=None):
        if self.causality:
            tl = seqs.shape[1]  # time dim len for enforce causality
            attention_mask = ~torch.tril(torch.ones((tl, tl),
                                         dtype=torch.bool,
                                         device=self.args.device))
        else:
            attention_mask = None

        if timeline_mask is not None:
            seqs *= ~timeline_mask.unsqueeze(-1)

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                            attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            if timeline_mask is not None:
                seqs *= ~timeline_mask.unsqueeze(-1)

        return self.last_layernorm(seqs)


class LiveRec(nn.Module):
    def __init__(self, args):
        super(LiveRec, self).__init__()
        self.args = args
        self.viewer_mode = getattr(args, 'viewer_feat_mode', 'off')

        self.item_embedding = nn.Embedding(args.N+1, args.K, padding_idx=0)
        self.pos_emb = nn.Embedding(args.seq_len, args.K)
        self.emb_dropout = nn.Dropout(p=0.2)
        self.use_bucket = bool(self.viewer_mode in ("bucket", "spike") and getattr(args, 'has_viewer_bucket', False) and getattr(args, 'num_viewer_buckets', 0) > 0)
        if self.use_bucket:
            num_buckets = int(getattr(args, 'num_viewer_buckets', 0)) + 2
            self.bucket_emb = nn.Embedding(num_buckets, args.K, padding_idx=0)
        else:
            self.bucket_emb = None

        # Sequence encoding attention
        self.att = Attention(args,
                             args.num_att,
                             args.num_heads,
                             causality=True)

        # Availability attention
        self.att_ctx = Attention(args,
                                 args.num_att_ctx,
                                 args.num_heads_ctx,
                                 causality=False)

        # Time interval embedding
        self.boundaries = torch.LongTensor([0]+list(range(77,3000+144, 144))).to(args.device)
        self.rep_emb = nn.Embedding(len(self.boundaries)+2, args.K, padding_idx=0)

    def forward(self, log_seqs, bucket_seqs=None):
        seqs = self.item_embedding(log_seqs)
        seqs *= self.item_embedding.embedding_dim ** 0.5
        if self.use_bucket and bucket_seqs is not None:
            seqs += self.bucket_emb(bucket_seqs)

        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.args.device))

        seqs = self.emb_dropout(seqs)

        timeline_mask = (log_seqs == 0).to(self.args.device)

        feats = self.att(seqs, timeline_mask)

        return feats

    def predict(self, feats, inputs, items, ctx=None, data=None, bucket_items=None):
        if ctx is not None:
            i_embs = ctx
        else:
            i_embs = self.item_embedding(items)
            if self.use_bucket and bucket_items is not None:
                i_embs += self.bucket_emb(bucket_items)

        return (feats * i_embs).sum(dim=-1)

    def compute_rank(self, data, store, k=10, **kwargs):
        inputs = data[:,:,3]  # inputs
        pos    = data[:,:,5]  # targets
        xtsy   = data[:,:,6]  # targets ts
        inputs_ts = data[:,:,2]  # input timestamps
        bucket_inputs = None
        if self.use_bucket:
            idx_in = getattr(self.args, 'bucket_input_idx', None)
            if idx_in is not None and idx_in < data.shape[2]:
                bucket_inputs = data[:,:,idx_in]

        feats = self(inputs, bucket_inputs)
        detail_list = store.get('rank_details', None)
        include_seq = bool(detail_list is not None and getattr(self.args, 'rank_dump_include_seq', False))
        include_hits = bool(detail_list is not None and getattr(self.args, 'rank_dump_include_hits', False))
        miss_topk = int(getattr(self.args, 'rank_dump_miss_topk', 0) or 0)
        track_new_ratio = bool(getattr(self.args, "track_new_ratio", False))
        step_candidates = kwargs.get("candidates") or {}
        step_candidates = kwargs.get("candidates") or {}

        if self.args.fr_ctx:
            ctx, batch_inds = self.get_ctx_att(data, feats)

        # Vectorized mask computation (EP-INF-006 optimization)
        mask = _compute_repeat_mask_vectorized(pos, store)

        candidates = kwargs.get("candidates")
        if candidates is None:
            candidates = build_step_candidates(self.args, xtsy)
        step_candidates = candidates or {}

        candidates = kwargs.get("candidates")
        if candidates is None:
            candidates = build_step_candidates(self.args, xtsy)
        step_candidates = candidates or {}

        # Optimize: batch availability processing using av_tens
        # Group by step to avoid redundant embedding computations
        steps = xtsy[:,-1].cpu().numpy()
        unique_steps = {}
        for b, step in enumerate(steps):
            step = int(step)
            if step not in unique_steps:
                unique_steps[step] = []
            unique_steps[step].append(b)
        
        # Pre-compute embeddings for unique steps (only if not using fr_ctx)
        step_embs = {}
        step_av = {}
        if not self.args.fr_ctx:
            for step, batch_indices in unique_steps.items():
                cand = step_candidates.get(step)
                if cand is None:
                    continue
                av = cand.items
                if av.numel() == 0:
                    continue
                av_embs = self.item_embedding(av)
                if self.use_bucket and cand.buckets is not None:
                    av_embs += self.bucket_emb(cand.buckets[:av_embs.shape[0]])
                step_embs[step] = av_embs
                step_av[step] = av
        
        # Process each sample using pre-computed embeddings
        for b in range(inputs.shape[0]):
            step = int(xtsy[b,-1].item())
            
            if self.args.fr_ctx:
                cand = step_candidates.get(step)
                if cand is None:
                    continue
                av = cand.items
                ctx_expand = torch.zeros(self.args.av_tens.shape[1],self.args.K,device=self.args.device)
                ctx_expand[batch_inds[b,-1,:],:] = ctx[b,-1,:,:]
                scores = (feats[b,-1,:] * ctx_expand).sum(-1)
                scores = scores[:av.shape[0]]
            else:
                if step not in step_embs:
                    continue
                av_embs = step_embs[step]
                av = step_av[step]
                scores = (feats[b,-1,:] * av_embs).sum(-1)

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
                    store.setdefault('new_ratio_topk', []).append(float(new_ratio))

            iseq = pos[b,-1] == av
            idx = torch.where(iseq)[0]
            if idx.numel() == 0:
                continue
            rank = torch.where(order==idx)[0].item()

            _update_tier_stats(store, self.args, int(pos[b,-1].item()), rank, bool(mask[b]))

            if mask[b]:
                store['rrep'] += [rank]
            else:
                store['rnew'] += [rank]
            store['rall'] += [rank]

            if detail_list is not None:
                is_hit = (rank == 0)
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
                        "user_id": _extract_user_id(data[b,:,4]),
                        "target_item": int(pos[b,-1].item()),
                        "target_step": int(xtsy[b,-1].item()),
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

    def get_ctx_att(self,data,feats,neg=None):
        if not self.args.fr_ctx: return None

        inputs,pos,xtsy = data[:,:,3],data[:,:,5],data[:,:,6]

        # unbatch indices
        ci = torch.nonzero(inputs, as_tuple=False)
        flat_xtsy = xtsy[ci[:,0],ci[:,1]]

        av = self.args.av_tens[flat_xtsy,:]
        av_embs = self.item_embedding(av)
        if self.use_bucket:
            av_bucket_tens = getattr(self.args, 'av_bucket_tens', None)
            if av_bucket_tens is not None:
                av_bucket = av_bucket_tens[flat_xtsy,:]
                av_embs += self.bucket_emb(av_bucket)

        # repeat consumption: time interval embeddings
        if self.args.fr_rep:
            av_rep_batch = self.get_av_rep(data)
            av_rep_flat  = av_rep_batch[ci[:,0],ci[:,1]]
            rep_enc = self.rep_emb(av_rep_flat)
            av_embs += rep_enc

        flat_feats = feats[ci[:,0],ci[:,1],:]
        flat_feats = flat_feats.unsqueeze(1).expand(flat_feats.shape[0],
                                                    self.args.av_tens.shape[-1],
                                                    flat_feats.shape[1])

        scores = (av_embs * flat_feats).sum(-1)
        inds   = scores.topk(self.args.topk_att,dim=1).indices

        # embed selected items
        seqs = torch.gather(av_embs, 1, inds.unsqueeze(2) \
                    .expand(-1,-1,self.args.K))

        seqs = self.att_ctx(seqs)

        def expand_att(items):
            av_pos = torch.where(av==items[ci[:,0],ci[:,1]].unsqueeze(1))[1]
            is_in = torch.any(inds == av_pos.unsqueeze(1),1)

            att_feats = torch.zeros(av.shape[0],self.args.K).to(self.args.device)
            att_feats[is_in,:] = seqs[is_in,torch.where(av_pos.unsqueeze(1) == inds)[1],:]

            out = torch.zeros(inputs.shape[0],inputs.shape[1],self.args.K).to(self.args.device)
            out[ci[:,0],ci[:,1],:] = att_feats
            return out

        # training
        if pos is not None and neg is not None:
            return expand_att(pos),expand_att(neg)
        # testing
        else:
            out = torch.zeros(inputs.shape[0],inputs.shape[1],seqs.shape[1],self.args.K).to(self.args.device)
            out[ci[:,0],ci[:,1],:] = seqs
            batch_inds = torch.zeros(inputs.shape[0],inputs.shape[1],inds.shape[1],dtype=torch.long).to(self.args.device)
            batch_inds[ci[:,0],ci[:,1],:] = inds
            return out,batch_inds

    def train_step(self, data, use_ctx=False):
        inputs,pos = data[:,:,3],data[:,:,5]
        bucket_inputs = None
        bucket_targets = None
        if self.use_bucket:
            idx_in = getattr(self.args, 'bucket_input_idx', None)
            idx_tg = getattr(self.args, 'bucket_target_idx', None)
            if idx_in is not None and idx_in < data.shape[2]:
                bucket_inputs = data[:,:,idx_in]
            if idx_tg is not None and idx_tg < data.shape[2]:
                bucket_targets = data[:,:,idx_tg]
        feats = self(inputs, bucket_inputs)

        num_negs = int(getattr(self.args, 'num_negs', 1) or 1)
        single_neg = None
        single_neg_bucket = None
        if num_negs == 1:
            neg_sample = sample_negs(data,self.args)
            if isinstance(neg_sample, tuple):
                single_neg, single_neg_bucket = neg_sample
            else:
                single_neg, single_neg_bucket = neg_sample, None
            single_neg = single_neg.to(self.args.device)
            if single_neg_bucket is not None:
                single_neg_bucket = single_neg_bucket.to(self.args.device)

        ctx_pos,ctx_neg = None,None
        if self.args.fr_ctx:
            if num_negs == 1 and single_neg is not None:
                ctx_pos, ctx_neg = self.get_ctx_att(data,feats,single_neg)
            else:
                ctx_pos, _ = self.get_ctx_att(data,feats,neg=pos)

        bucket_arg_pos = None if self.args.fr_ctx else bucket_targets
        pos_logits = self.predict(feats,inputs,pos,ctx_pos,data,bucket_arg_pos)
        scope = getattr(self.args, 'train_scope', 'all') or 'all'
        valid_mask, rep_mask_batch, batch_mask = scope_masks(inputs, pos, scope)

        if bool(getattr(self.args, 'train_last_only', False)):
            vm_last = torch.zeros_like(valid_mask)
            vm_last[:, -1] = valid_mask[:, -1]
            valid_mask = vm_last

        loss_pos = (-torch.log(pos_logits[valid_mask].sigmoid()+1e-24)).sum()
        loss_neg_sum = 0.0
        if num_negs == 1:
            bucket_arg_neg = None
            if not self.args.fr_ctx:
                bucket_arg_neg = single_neg_bucket if 'single_neg_bucket' in locals() else None
                if bucket_arg_neg is not None:
                    bucket_arg_neg = bucket_arg_neg.to(self.args.device)
            neg_logits = self.predict(feats,inputs,single_neg,ctx_neg,data,bucket_arg_neg)
            loss_neg_sum = (-torch.log(1-neg_logits[valid_mask].sigmoid()+1e-24)).sum()
        else:
            neg_items, neg_buckets = sample_negs_k(data,self.args,num_negs)
            for idx_neg, neg in enumerate(neg_items):
                neg = neg.to(self.args.device)
                ctx_neg_i = None
                if self.args.fr_ctx:
                    _,ctx_neg_i = self.get_ctx_att(data,feats,neg)
                bucket_arg = None
                if not self.args.fr_ctx and neg_buckets is not None:
                    bucket_arg = neg_buckets[idx_neg].to(self.args.device)
                neg_logits = self.predict(feats,inputs,neg,ctx_neg_i,data,bucket_arg)
                loss_neg_sum = loss_neg_sum + (-torch.log(1-neg_logits[valid_mask].sigmoid()+1e-24)).sum()
        loss = loss_pos + (loss_neg_sum / max(1,num_negs))

        log_and_accumulate_train_debug(self, 'LiveRec', scope, inputs, valid_mask, pos, rep_mask_batch, batch_mask)

        return loss

    def get_av_rep(self,data):
        bs     = data.shape[0]
        inputs = data[:,:,3] # inputs
        xtsb   = data[:,:,2] # inputs ts
        xtsy   = data[:,:,6] # targets ts

        av_batch  = self.args.av_tens[xtsy.view(-1),:]
        av_batch  = av_batch.view(xtsy.shape[0],xtsy.shape[1],-1)
        av_batch *= (xtsy!=0).unsqueeze(2) # masking pad inputs
        av_batch  = av_batch.to(self.args.device)

        mask_caus = 1-torch.tril(torch.ones(self.args.seq_len,self.args.seq_len),diagonal=-1)
        mask_caus = mask_caus.unsqueeze(0).unsqueeze(3)
        mask_caus = mask_caus.expand(bs,-1,-1,self.args.av_tens.shape[-1])
        mask_caus = mask_caus.type(torch.bool).to(self.args.device)

        tile = torch.arange(self.args.seq_len).unsqueeze(0).repeat(bs,1).to(self.args.device)

        bm   = (inputs.unsqueeze(2).unsqueeze(3)==av_batch.unsqueeze(1).expand(-1,self.args.seq_len,-1,-1))
        bm  &= mask_caus

        sm   = bm.type(torch.int).argmax(1)
        sm   = torch.any(bm,1) * sm

        sm   = (torch.gather(xtsy, 1, tile).unsqueeze(2) -
                torch.gather(xtsb.unsqueeze(2).expand(-1,-1,self.args.av_tens.shape[-1]), 1, sm))
        sm   = torch.bucketize(sm, self.boundaries)+1
        sm   = torch.any(bm,1) * sm

        sm  *= av_batch!=0
        sm  *= inputs.unsqueeze(2)!=0
        return sm


class SASRec(nn.Module):
    def __init__(self, args):
        super(SASRec, self).__init__()
        self.args = args
        self.viewer_mode = getattr(args, 'viewer_feat_mode', 'off')

        self.item_embedding = nn.Embedding(args.N+1, args.K, padding_idx=0)
        self.pos_emb = nn.Embedding(args.seq_len, args.K)
        self.emb_dropout = nn.Dropout(p=0.2)
        self.use_bucket = bool(self.viewer_mode in ("bucket", "spike") and getattr(args, 'has_viewer_bucket', False) and getattr(args, 'num_viewer_buckets', 0) > 0)
        if self.use_bucket:
            num_buckets = int(getattr(args, 'num_viewer_buckets', 0)) + 2
            self.bucket_emb = nn.Embedding(num_buckets, args.K, padding_idx=0)
        else:
            self.bucket_emb = None

        self.att = Attention(args,
                             args.num_att,
                             args.num_heads,
                             causality=True)

    def forward(self, log_seqs, bucket_seqs=None):
        seqs = self.item_embedding(log_seqs)
        seqs *= self.item_embedding.embedding_dim ** 0.5
        if self.use_bucket and bucket_seqs is not None:
            seqs += self.bucket_emb(bucket_seqs)

        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.args.device))

        seqs = self.emb_dropout(seqs)

        timeline_mask = (log_seqs == 0).to(self.args.device)

        feats = self.att(seqs, timeline_mask)
        return feats

    def predict(self, feats, inputs, items, bucket_items=None):
        item_embs = self.item_embedding(items)
        if self.use_bucket and bucket_items is not None:
            item_embs += self.bucket_emb(bucket_items)
        return (feats * item_embs).sum(dim=-1)

    def compute_rank(self, data, store, k=10, **kwargs):
        inputs = data[:,:,3] # inputs
        pos    = data[:,:,5] # targets
        xtsy   = data[:,:,6] # targets ts
        inputs_ts = data[:,:,2] # inputs timestamps
        bucket_inputs = None
        if self.use_bucket:
            idx_in = getattr(self.args, 'bucket_input_idx', None)
            if idx_in is not None and idx_in < data.shape[2]:
                bucket_inputs = data[:,:,idx_in]

        feats = self(inputs, bucket_inputs)
        detail_list = store.get('rank_details', None)
        include_seq = bool(detail_list is not None and getattr(self.args, 'rank_dump_include_seq', False))
        include_hits = bool(detail_list is not None and getattr(self.args, 'rank_dump_include_hits', False))
        miss_topk = int(getattr(self.args, 'rank_dump_miss_topk', 0) or 0)
        track_new_ratio = bool(getattr(self.args, "track_new_ratio", False))
        step_candidates = kwargs.get("candidates") or {}

        # Vectorized mask computation (EP-INF-006 optimization)
        mask = _compute_repeat_mask_vectorized(pos, store)

        # Optimize: batch availability processing using av_tens
        # Group by step to avoid redundant embedding computations
        steps = xtsy[:,-1].cpu().numpy()
        unique_steps = {}
        for b, step in enumerate(steps):
            step = int(step)
            if step not in unique_steps:
                unique_steps[step] = []
            unique_steps[step].append(b)
        
        # Pre-compute embeddings for unique steps
        step_embs = {}
        step_av = {}
        for step, batch_indices in unique_steps.items():
            cand = step_candidates.get(step)
            if cand is None:
                continue
            av = cand.items
            if av.numel() == 0:
                continue
            av_embs = self.item_embedding(av)
            if self.use_bucket and cand.buckets is not None:
                av_embs += self.bucket_emb(cand.buckets[:av_embs.shape[0]])
            step_embs[step] = av_embs
            step_av[step] = av
        
        # Process each sample using pre-computed embeddings
        for b in range(inputs.shape[0]):
            step = int(xtsy[b,-1].item())
            if step not in step_embs:
                continue
            
            av_embs = step_embs[step]
            av = step_av[step]

            scores  = (feats[b,-1,:] * av_embs).sum(-1)

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
                    store.setdefault('new_ratio_topk', []).append(float(new_ratio))

            iseq = pos[b,-1] == av
            idx  = torch.where(iseq)[0]
            if idx.numel() == 0:
                continue
            rank = torch.where(order==idx)[0].item()

            _update_tier_stats(store, self.args, int(pos[b,-1].item()), rank, bool(mask[b]))

            if mask[b]:
                store['rrep'] += [rank]
            else:
                store['rnew'] += [rank]
            store['rall'] += [rank]

            if detail_list is not None:
                is_hit = (rank == 0)
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
                        "user_id": _extract_user_id(data[b,:,4]),
                        "target_item": int(pos[b,-1].item()),
                        "target_step": int(xtsy[b,-1].item()),
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

    def train_step(self, data):
        inputs,pos = data[:,:,3],data[:,:,5]
        bucket_inputs = None
        bucket_targets = None
        if self.use_bucket:
            idx_in = getattr(self.args, 'bucket_input_idx', None)
            idx_tg = getattr(self.args, 'bucket_target_idx', None)
            if idx_in is not None and idx_in < data.shape[2]:
                bucket_inputs = data[:,:,idx_in]
            if idx_tg is not None and idx_tg < data.shape[2]:
                bucket_targets = data[:,:,idx_tg]
        feats = self(inputs, bucket_inputs)
        pos_logits = self.predict(feats,inputs,pos,bucket_targets)
        scope = getattr(self.args, 'train_scope', 'all') or 'all'
        valid_mask, rep_mask_batch, batch_mask = scope_masks(inputs, pos, scope)
        loss_pos = (-torch.log(pos_logits[valid_mask].sigmoid()+1e-24)).sum()
        num_negs = int(getattr(self.args, 'num_negs', 1) or 1)
        loss_neg_sum = 0.0
        if num_negs == 1:
            neg_sample = sample_negs(data,self.args)
            if isinstance(neg_sample, tuple):
                neg, neg_bucket = neg_sample
            else:
                neg, neg_bucket = neg_sample, None
            neg = neg.to(self.args.device)
            if neg_bucket is not None:
                neg_bucket = neg_bucket.to(self.args.device)
            neg_logits = self.predict(feats,inputs,neg,neg_bucket)
            loss_neg_sum = (-torch.log(1-neg_logits[valid_mask].sigmoid()+1e-24)).sum()
        else:
            neg_items, neg_buckets = sample_negs_k(data,self.args,num_negs)
            for idx_neg, neg in enumerate(neg_items):
                neg = neg.to(self.args.device)
                neg_bucket = None
                if neg_buckets is not None:
                    neg_bucket = neg_buckets[idx_neg].to(self.args.device)
                neg_logits = self.predict(feats,inputs,neg,neg_bucket)
                loss_neg_sum = loss_neg_sum + (-torch.log(1-neg_logits[valid_mask].sigmoid()+1e-24)).sum()
        loss  = loss_pos + (loss_neg_sum / max(1,num_negs))
        log_and_accumulate_train_debug(self, 'SASRec', scope, inputs, valid_mask, pos, rep_mask_batch, batch_mask)
        return loss
