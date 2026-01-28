import random
import torch
import json
from datetime import datetime
import time


def _bucket_lookup(args, t, item):
    if not getattr(args, 'has_viewer_bucket', False):
        return 0
    cache = getattr(args, '_ts_bucket_cache', None)
    if cache is None:
        cache = {}
        setattr(args, '_ts_bucket_cache', cache)
    if t not in cache:
        streams = args.ts.get(t, [])
        buckets = getattr(args, 'ts_bucket', {}).get(t, [])
        cache[t] = dict(zip(streams, buckets))
    return cache[t].get(item, 0)


def sample_av(p,t,args):
    # availability sampling
    av = args.ts[t]
    while True:
        ridx = random.randint(0,len(av)-1)
        ri   = av[ridx]
        if p!=ri:
            return ri
 

def sample_uni(p,t,args):
    # uniform sampling
    while True:
        ri = random.randint(0,args.N-1)
        if p!=ri:
            return ri


def _sample_negs_online(pos, xts, args, k: int):
    """Sample negatives uniformly from currently online items using args.av_tens."""
    t0 = time.time()
    neg_list = [torch.zeros_like(pos) for _ in range(k)]
    has_bucket = bool(getattr(args, 'has_viewer_bucket', False))
    bucket_lists = [torch.zeros_like(pos) for _ in range(k)] if has_bucket else None
    av_tens = getattr(args, 'av_tens', None)
    av_bucket = getattr(args, 'av_bucket_tens', None) if has_bucket else None
    if av_tens is None:
        # fall back to availability sampler if tensor is missing
        ci = torch.nonzero(pos, as_tuple=False)
        for idx in range(ci.shape[0]):
            b = int(ci[idx,0].item()); s = int(ci[idx,1].item())
            p = int(pos[b,s].item()); t = int(xts[b,s].item())
            for kk in range(k):
                neg_list[kk][b,s] = sample_av(p,t,args)
        return neg_list, bucket_lists

    device = pos.device
    if av_tens.device != device:
        av_tens = av_tens.to(device)
    if has_bucket and av_bucket is not None and av_bucket.device != device:
        av_bucket = av_bucket.to(device)
    ci = torch.nonzero(pos, as_tuple=False)
    if ci.shape[0] == 0:
        return neg_list, bucket_lists

    ts = xts[ci[:,0],ci[:,1]]
    ps = pos[ci[:,0],ci[:,1]]

    avail = av_tens[ts]  # [M, max_av]
    mask = (avail != 0) & (avail != ps.unsqueeze(1))
    has_candidate = mask.any(dim=1)

    fallback_rows = torch.nonzero(~has_candidate, as_tuple=False).squeeze(-1)
    if fallback_rows.numel() == 0:
        fallback_rows = None

    for kk in range(k):
        rand = torch.rand(avail.shape, device=device)
        rand[~mask] = -1.0
        idx = rand.argmax(dim=1)
        chosen = avail.gather(1, idx.unsqueeze(1)).squeeze(1)
        neg_list[kk][ci[:,0],ci[:,1]] = chosen
        if fallback_rows is not None and fallback_rows.numel() > 0:
            for fi in fallback_rows.tolist():
                b = int(ci[fi,0].item()); s = int(ci[fi,1].item())
                p = int(ps[fi].item()); t = int(ts[fi].item())
                row_avail = avail[fi]
                candidates = row_avail[(row_avail != 0) & (row_avail != ps[fi])]
                if candidates.numel() > 0:
                    ridx = torch.randint(0, candidates.numel(), (1,), device=device)
                    neg_val = candidates[ridx].squeeze(0)
                else:
                    neg_val = torch.tensor(sample_uni(p, t, args), device=device, dtype=pos.dtype)
                neg_list[kk][b,s] = neg_val
                if bucket_lists is not None:
                    bucket_lists[kk][b,s] = 0
        if bucket_lists is not None and av_bucket is not None:
            bucket_vals = av_bucket[ts, idx]
            bucket_lists[kk][ci[:,0],ci[:,1]] = bucket_vals

    if getattr(args, 'debug', False):
        try:
            total = ci.shape[0] * k
            fallback_cnt = (0 if fallback_rows is None else fallback_rows.numel() * k)
            if not hasattr(args, '_neg_dbg') or not isinstance(args._neg_dbg, dict):
                args._neg_dbg = {
                    'total': 0,
                    'chosen_rep': 0,
                    'chosen_new': 0,
                    'fallback': 0,
                    'attempts_sum': 0,
                    'accept': 0,
                    'reject': 0,
                    'time_ms': 0.0,
                    'pos_rep': 0,
                    'pos_new': 0,
                    'posrep_to_rep': 0,
                    'posnew_to_new': 0,
                }
            d = args._neg_dbg
            d['total'] += total
            d['chosen_rep'] += 0
            d['chosen_new'] += total
            d['fallback'] += fallback_cnt
            d['attempts_sum'] += total
            d['accept'] += total
            d['reject'] += 0
            d['time_ms'] += (time.time() - t0) * 1000.0
        except Exception:
            pass

    return neg_list, bucket_lists


def sample_negs_k(data, args, k: int):
    pos,xts = data[:,:,5],data[:,:,6]
    inputs   = data[:,:,3]
    Bs, S = pos.shape
    neg_list = [torch.zeros_like(pos) for _ in range(k)]
    has_bucket = bool(getattr(args, 'has_viewer_bucket', False))
    bucket_lists = [torch.zeros_like(pos) for _ in range(k)] if has_bucket else None

    neg_mode = getattr(args, 'neg_mode', 'auto') or 'auto'
    if neg_mode == 'online':
        return _sample_negs_online(pos, xts, args, k)

    ci = torch.nonzero(pos, as_tuple=False)
    ps = pos[ci[:,0],ci[:,1]].tolist()
    ts = xts[ci[:,0],ci[:,1]].tolist()

    # timing for debug aggregation
    t0 = time.time()
    # precompute user history sets per batch row
    b_indices = [int(x.item()) for x in torch.unique(ci[:,0])]
    hist_sets = {b: set(inputs[b, inputs[b,:]!=0].tolist()) for b in b_indices}
    # cache availability sets per unique time
    unique_ts = list(set(ts))
    av_set_cache = {t: set(args.ts[t]) for t in unique_ts}

    # per-epoch debug accumulators (local) to minimize attribute lookups
    dbg_total = 0
    dbg_chosen_rep = 0
    dbg_chosen_new = 0
    dbg_fallback = 0
    dbg_attempts_sum = 0
    dbg_accept = 0
    dbg_reject = 0
    # posgroup-specific mapping counters
    dbg_pos_rep = 0
    dbg_pos_new = 0
    dbg_posrep_to_rep = 0
    dbg_posnew_to_new = 0

    # decide mode once per call (matches previous behavior)
    mix_p = float(getattr(args, 'neg_mix_p', 0.5) or 0.5)

    for idx in range(ci.shape[0]):
        b = int(ci[idx,0].item()); s = int(ci[idx,1].item())
        p = ps[idx]; t = ts[idx]

        av_list = args.ts[t] if t in args.ts else []
        if not av_list:
            # fallback
            for kk in range(k):
                neg_list[kk][b, s] = sample_av(p,t,args)
                if bucket_lists is not None:
                    bucket_lists[kk][b,s] = _bucket_lookup(args, t, neg_list[kk][b,s].item())
            # debug
            if getattr(args, 'debug', False):
                dbg_total += k
                dbg_fallback += k
            continue
        av_set = av_set_cache[t]

        # choose base mode for this event
        if neg_mode == 'auto':
            scope = getattr(args, 'train_scope', 'all') or 'all'
            base_mode = scope
        else:
            base_mode = neg_mode  # 'rep' | 'new' | 'all'

        hist_items = hist_sets[b]
        # small rep-candidates from history∩availability (history is small)
        rep_candidates = [h for h in hist_items if (h != p and h in av_set)]
        # build set once for fast membership; avoid per-trial loops for 'new'
        # (trial_indices kept only for backward compatibility in case of fallback)
        max_trials_base = 0
        trial_indices = []

        # sample K negatives
        for kk in range(k):
            # decide this negative's mode
            if base_mode == 'all':
                mode = 'rep' if (random.random() < 0.5) else 'new'
            elif base_mode == 'mixture':
                mode = 'rep' if (random.random() < mix_p) else 'new'
            elif base_mode == 'posgroup':
                mode = 'rep' if (p in hist_items) else 'new'
            else:
                mode = base_mode

            chosen_name = mode
            found = False
            attempts = 0

            if mode == 'rep':
                if rep_candidates:
                    ri = rep_candidates[random.randint(0, len(rep_candidates)-1)]
                    neg_list[kk][b, s] = ri
                    if bucket_lists is not None:
                        bucket_lists[kk][b,s] = _bucket_lookup(args, t, ri)
                    found = True
                    attempts = 1
                else:
                    # no valid rep candidate
                    neg_list[kk][b, s] = sample_av(p,t,args)
                    if bucket_lists is not None:
                        bucket_lists[kk][b,s] = _bucket_lookup(args, t, neg_list[kk][b,s].item())
                    found = False
                    attempts = 1
            else:  # 'new'
                # fast path: set difference to get new-only pool
                new_pool = None
                if av_set:
                    # exclude positive and history items
                    if p in av_set:
                        # copy to avoid mutating cached set
                        new_pool = [a for a in av_set if (a != p and a not in hist_items)]
                    else:
                        new_pool = [a for a in av_set if (a not in hist_items)]
                if new_pool:
                    ri = new_pool[random.randint(0, len(new_pool)-1)]
                    neg_list[kk][b, s] = ri
                    if bucket_lists is not None:
                        bucket_lists[kk][b,s] = _bucket_lookup(args, t, ri)
                    found = True
                    attempts = 1
                else:
                    neg_list[kk][b, s] = sample_av(p,t,args)
                    if bucket_lists is not None:
                        bucket_lists[kk][b,s] = _bucket_lookup(args, t, neg_list[kk][b,s].item())
                    found = False
                    attempts = 1

            # accumulate debug
            if getattr(args, 'debug', False):
                dbg_total += 1
                dbg_attempts_sum += attempts
                if chosen_name == 'rep':
                    dbg_chosen_rep += 1
                else:
                    dbg_chosen_new += 1
                # posgroup mapping counters (actual mapping based on sampled item)
                if neg_mode == 'posgroup':
                    is_rep_pos = (p in hist_items)
                    try:
                        ri_actual = int(neg_list[kk][b, s].item())
                    except Exception:
                        ri_actual = None
                    is_rep_neg = (ri_actual in hist_items) if ri_actual is not None else False
                    if is_rep_pos:
                        dbg_pos_rep += 1
                        if is_rep_neg:
                            dbg_posrep_to_rep += 1
                    else:
                        dbg_pos_new += 1
                        if (not is_rep_neg):
                            dbg_posnew_to_new += 1
                if not found:
                    dbg_fallback += 1
                    dbg_reject += attempts
                else:
                    dbg_accept += 1
                    dbg_reject += max(0, attempts - 1)

    # write debug aggregates
    if getattr(args, 'debug', False):
        try:
            if not hasattr(args, '_neg_dbg') or not isinstance(args._neg_dbg, dict):
                args._neg_dbg = {
                    'total': 0,
                    'chosen_rep': 0,
                    'chosen_new': 0,
                    'fallback': 0,
                    'attempts_sum': 0,
                    'accept': 0,
                    'reject': 0,
                    'time_ms': 0.0,
                    'pos_rep': 0,
                    'pos_new': 0,
                    'posrep_to_rep': 0,
                    'posnew_to_new': 0,
                }
            d = args._neg_dbg
            d['total'] += dbg_total
            d['chosen_rep'] += dbg_chosen_rep
            d['chosen_new'] += dbg_chosen_new
            d['fallback'] += dbg_fallback
            d['attempts_sum'] += dbg_attempts_sum
            d['accept'] += dbg_accept
            d['reject'] += dbg_reject
            d['time_ms'] += (time.time() - t0) * 1000.0
            d['pos_rep'] += dbg_pos_rep
            d['pos_new'] += dbg_pos_new
            d['posrep_to_rep'] += dbg_posrep_to_rep
            d['posnew_to_new'] += dbg_posnew_to_new
        except Exception:
            pass

    # stack result if needed by callers
    return neg_list, bucket_lists


def sample_negs(data,args):
    # backward-compatible single negative sampler using the multi-K path
    neg_items, neg_buckets = sample_negs_k(data, args, 1)
    if neg_buckets is None:
        return neg_items[0]
    return neg_items[0], neg_buckets[0]
