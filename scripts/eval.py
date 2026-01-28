import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.candidates import build_step_candidates  # noqa: E402
from data.loader import _dataset_cache_key, get_dataloaders, load_data  # noqa: E402
from models.registry import get_model_type  # noqa: E402
from scripts import arguments as arguments_cli  # noqa: E402
from scripts.config_utils import (  # noqa: E402
    DEFAULT_LOG_JSONL,
    DEFAULT_LOG_TXT,
    apply_config_overrides,
    ensure_log_paths,
    load_config_dict,
)

def _to_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def _spike_cache_key(args):
    mode = getattr(args, "viewer_feat_mode", "off") or "off"
    if bool(getattr(args, "use_percentile_3d", False)):
        feat = "p3d"
    elif bool(getattr(args, "use_percentile_features", False)):
        feat = "p4"
    elif bool(getattr(args, "use_hybrid_features", False)):
        feat = "hyb"
    else:
        feat = "z"
    win = getattr(args, "ablation_window_size", None)
    win_tag = f"w{int(win)}" if win is not None else "wauto"
    mask_user = "nouser" if bool(getattr(args, "ablation_mask_user_repr", False)) else "user"
    return f"{mode}_{feat}_{win_tag}_{mask_user}"


RANK_DUMP_FIELDS = [
    "phase",
    "phase_bucket",
    "user_id",
    "target_item",
    "target_step",
    "is_repeat",
    "overall_rank0",
    "overall_rank1",
    "rep_rank0",
    "rep_rank1",
    "new_rank0",
    "new_rank1",
    "total_candidates",
    "rep_candidates",
    "new_candidates",
    "hit1",
    "hit5",
    "hit10",
    "checkpoint",
    "history_length",
    "history_items",
    "history_steps",
]

EVAL_SCOPE_DEBUG = Path("docs/analysis/eval_scope_debug.jsonl")
VALIDATION_TIMESTAMP_LOG = Path("validation_timestamps.log")


def _log_validation_timestamp(phase, message, elapsed=None, batch_idx=None):
    """Validation 단계에서만 timestamp를 로깅하는 함수"""
    # phase가 존재하고 'val'로 시작할 때만 로깅
    if phase and phase.lower().startswith('val'):
        timestamp = datetime.utcnow().isoformat()
        log_line = f"[{timestamp}] {message}"
        if batch_idx is not None:
            log_line = f"[{timestamp}] [batch {batch_idx}] {message}"
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


def _phase_bucket(phase_label):
    if not phase_label:
        return ""
    pl = str(phase_label).lower()
    if pl.startswith("test"):
        return "test"
    if pl.startswith("val"):
        return "val"
    if pl.startswith("train"):
        return "train"
    return pl


def _should_dump_rank(args, phase):
    path = getattr(args, "rank_dump_csv", "") or ""
    if not path:
        return False
    target = getattr(args, "rank_dump_phase", "test") or "test"
    if target == "any":
        return True
    bucket = _phase_bucket(phase)
    return bucket == target


def _write_rank_dump(args, records, phase):
    path = getattr(args, "rank_dump_csv", "") or ""
    if not path or not records:
        return
    phase_raw = phase or ""
    phase_bucket = _phase_bucket(phase)
    ckpt = getattr(args, "checkpoint", "") or ""
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    need_header = True
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        need_header = False
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RANK_DUMP_FIELDS)
        if need_header:
            writer.writeheader()
        for rec in records:
            row = {
                "phase": phase_raw,
                "phase_bucket": phase_bucket,
                "user_id": rec.get("user_id"),
                "target_item": rec.get("target_item"),
                "target_step": rec.get("target_step"),
                "is_repeat": int(bool(rec.get("is_repeat"))),
                "overall_rank0": rec.get("overall_rank"),
                "overall_rank1": (rec.get("overall_rank") + 1) if rec.get("overall_rank") is not None else None,
                "rep_rank0": rec.get("rep_rank"),
                "rep_rank1": (rec.get("rep_rank") + 1) if rec.get("rep_rank") is not None else None,
                "new_rank0": rec.get("new_rank"),
                "new_rank1": (rec.get("new_rank") + 1) if rec.get("new_rank") is not None else None,
                "total_candidates": rec.get("total_candidates"),
                "rep_candidates": rec.get("rep_candidates"),
                "new_candidates": rec.get("new_candidates"),
                "hit1": int(bool(rec.get("hit1"))),
                "hit5": int(bool(rec.get("hit5"))),
                "hit10": int(bool(rec.get("hit10"))),
                "checkpoint": ckpt,
                "history_length": 0,
                "history_items": "",
                "history_steps": "",
            }
            hist_items = rec.get("history_items")
            hist_steps = rec.get("history_steps")
            if hist_items:
                row["history_length"] = len(hist_items)
                row["history_items"] = json.dumps(hist_items)
            if hist_steps:
                row["history_steps"] = json.dumps(hist_steps)
            writer.writerow(row)
    if getattr(args, "debug", False):
        rep_rows = sum(1 for rec in records if rec.get("is_repeat"))
        new_rows = len(records) - rep_rows
        print("[rank_dump] phase={} ({}) wrote {} rows (rep={}, new={}) -> {}".format(
            phase_raw or "n/a",
            phase_bucket or "-",
            len(records),
            rep_rows,
            new_rows,
            path,
        ))


def _loggable_args(args):
    simple_types = (str, int, float, bool)
    exclude_keys = {"ts", "av_tens", "av_bucket_tens", "ts_bucket"}
    result = {}
    try:
        for key, val in vars(args).items():
            if key in exclude_keys or key.startswith("_"):
                continue
            if isinstance(val, Path):
                result[key] = str(val)
            elif isinstance(val, simple_types) or val is None:
                result[key] = val
    except Exception:
        pass
    return result


def save_scores(scores, args, path_txt=None, path_jsonl=None, checkpoint_path=None):
    if path_txt is None:
        path_txt = getattr(args, 'log_txt', DEFAULT_LOG_TXT)
    if path_jsonl is None:
        path_jsonl = getattr(args, 'log_jsonl', DEFAULT_LOG_JSONL)
    path_txt = Path(path_txt)
    path_txt.parent.mkdir(parents=True, exist_ok=True)
    path_jsonl = Path(path_jsonl)
    path_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args_dict = _loggable_args(args)
    ckpt = checkpoint_path or getattr(args, "checkpoint", "") or ""

    # 1) Human-readable TXT (generalized)
    with open(path_txt, 'a') as fout:
        fout.write("timestamp: {}\n".format(datetime.utcnow().isoformat()))
        if ckpt:
            checkpoint_str = str(ckpt) if isinstance(ckpt, Path) else ckpt
            checkpoint_path_obj = Path(checkpoint_str)
            fout.write("checkpoint: {}\n".format(checkpoint_str))
            fout.write("checkpoint_filename: {}\n".format(checkpoint_path_obj.name))
            fout.write("checkpoint_path: {}\n".format(str(checkpoint_path_obj.resolve())))
        if args_dict:
            # stable order for reproducibility
            joined = ", ".join(["{}={}".format(k, args_dict[k]) for k in sorted(args_dict.keys())])
            fout.write("args: {}\n".format(joined))

        # print available groups and metric names dynamically
        for group_name, group_vals in scores.items():
            if isinstance(group_vals, dict):
                numeric_only = group_vals and all(isinstance(group_vals[mk], (int, float)) for mk in group_vals)
                if numeric_only:
                    metrics_kv = ", ".join(
                        ["{}={:.5f}".format(mk, float(group_vals[mk])) for mk in sorted(group_vals.keys())]
                    )
                    fout.write("{}: {}\n".format(group_name, metrics_kv))
                elif group_name == "tier_breakdown":
                    fout.write("{}:\n".format(group_name))
                    for tier_label in sorted(group_vals.keys()):
                        entry = group_vals[tier_label]
                        parts = []
                        if isinstance(entry, dict):
                            all_stats = entry.get("all")
                            if all_stats:
                                parts.append("all_h01={:.3f}".format(float(all_stats.get("h01", 0.0))))
                                parts.append("all_h05={:.3f}".format(float(all_stats.get("h05", 0.0))))
                                parts.append(f"all_cnt={entry.get('all_count', 0)}")
                            rep_stats = entry.get("rep")
                            if rep_stats:
                                parts.append("rep_h01={:.3f}".format(float(rep_stats.get("h01", 0.0))))
                                parts.append(f"rep_cnt={entry.get('rep_count', 0)}")
                            new_stats = entry.get("new")
                            if new_stats:
                                parts.append("new_h01={:.3f}".format(float(new_stats.get("h01", 0.0))))
                                parts.append(f"new_cnt={entry.get('new_count', 0)}")
                        fout.write("  - {}: {}\n".format(tier_label, ", ".join(parts) if parts else "n/a"))
                else:
                    fout.write("{}: {}\n".format(group_name, json.dumps(group_vals)))
            else:
                try:
                    fout.write("{}: {:.5f}\n".format(group_name, float(group_vals)))
                except Exception:
                    fout.write("{}: {}\n".format(group_name, group_vals))
        fout.write("\n" + "-" * 60 + "\n\n")

    # 2) Machine-readable JSONL in current directory
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "args": args_dict,
        "scores": json.loads(json.dumps(scores, default=_to_serializable)),
    }
    if ckpt:
        # Store checkpoint as string (Path object -> str)
        checkpoint_str = str(ckpt) if isinstance(ckpt, Path) else ckpt
        record["checkpoint"] = checkpoint_str
        # Also store checkpoint filename for easy reference
        checkpoint_path_obj = Path(checkpoint_str)
        record["checkpoint_filename"] = checkpoint_path_obj.name
        record["checkpoint_path"] = str(checkpoint_path_obj.resolve())
    with open(path_jsonl, 'a') as fj:
        fj.write(json.dumps(record, ensure_ascii=False) + "\n")
 

def print_scores(scores):
    preferred_order = ['all', 'new', 'rep']
    groups = [g for g in preferred_order if g in scores] + [g for g in scores.keys() if g not in preferred_order]
    seen = set()
    ordered_groups = []
    for g in groups:
        if g not in seen:
            ordered_groups.append(g)
            seen.add(g)

    for group in ordered_groups:
        vals = scores[group]
        if isinstance(vals, dict) and group == "tier_breakdown":
            tiers = sorted(
                vals.items(),
                key=lambda kv: kv[1].get("all_count", 0),
                reverse=True,
            )
            print(f"{group}: {len(tiers)} buckets")
            for tier_label, tier_stats in tiers:
                parts = []
                all_stats = tier_stats.get("all")
                if all_stats:
                    parts.append("all_h01={:.3f}".format(float(all_stats.get("h01", 0.0))))
                    parts.append("all_h05={:.3f}".format(float(all_stats.get("h05", 0.0))))
                    parts.append(f"all_cnt={tier_stats.get('all_count', 0)}")
                rep_stats = tier_stats.get("rep")
                if rep_stats:
                    parts.append("rep_h01={:.3f}".format(float(rep_stats.get("h01", 0.0))))
                    parts.append(f"rep_cnt={tier_stats.get('rep_count', 0)}")
                new_stats = tier_stats.get("new")
                if new_stats:
                    parts.append("new_h01={:.3f}".format(float(new_stats.get("h01", 0.0))))
                    parts.append(f"new_cnt={tier_stats.get('new_count', 0)}")
                print("  - {}: {}".format(tier_label, ", ".join(parts)))
        elif isinstance(vals, dict) and vals and all(isinstance(vals[mn], (int, float)) for mn in vals.keys()):
            items = ["{}: {:.5f}".format(mn, float(vals[mn])) for mn in sorted(vals.keys())]
            print('{}: {}'.format(group, ' '.join(items)))
        elif isinstance(vals, dict):
            print(f"{group}: {len(vals)} buckets")
        else:
            try:
                print('{}: {:.5f}'.format(group, float(vals)))
            except Exception:
                print('{}: {}'.format(group, vals))

def metrics(a):
    a   = np.array(a)
    tot = float(len(a))
    if tot == 0:
        return {
          'h01': 0.0,
          'h05': 0.0,
          'h10': 0.0,
          'ndcg01': 0.0,
          'ndcg05': 0.0,
          'ndcg10': 0.0,
        }

    return {
      'h01': (a<1).sum()/tot,
      'h05': (a<5).sum()/tot,
      'h10': (a<10).sum()/tot,
      'ndcg01': np.sum([1 / np.log2(rank + 2) for rank in a[a<1]])/tot,
      'ndcg05': np.sum([1 / np.log2(rank + 2) for rank in a[a<5]])/tot,
      'ndcg10': np.sum([1 / np.log2(rank + 2) for rank in a[a<10]])/tot,
    }

def compute_recall(model, _loader, args, maxit=100000, phase=None, eval_slice=None):
    dump_ranks = _should_dump_rank(args, phase)
    store = {
        'rrep': [],
        'rnew': [],
        'rall': [],
        'ratio': [],
        'new_ratio_topk': [],
        'tier_stats': {},
    }
    if dump_ranks:
        store['rank_details'] = []

    # Spike Slice Analysis: Pre-calculate all spike scores if needed
    all_spike_scores = {}
    if eval_slice and eval_slice.startswith('spike_'):
        if not (hasattr(model, "spike_bias_weight") or model.__class__.__name__ == "SpikeScoreBiasSASRec"):
            print(f"[warn] eval_slice={eval_slice} requested, but the model is not a SpikeScoreBiasSASRec model. Skipping.")
        else:
            print(f"[slice_eval] Pre-calculating spike scores for slice: {eval_slice}")
            # Collect all unique (item_id, timestep) pairs from the loader
            unique_pairs = set()
            for i, data in enumerate(_loader):
                if maxit is not None and i >= maxit:
                    break
                xtsy = data[:, :, 6]  # timesteps [B, seq_len]
                steps_batch = xtsy[:, -1].cpu().numpy().astype(np.int32)  # [B]
                for step in steps_batch:
                    if step in args.ts:
                        available_items = args.ts[step]
                        unique_pairs.update((item, int(step)) for item in available_items)
            
            # Get spike scores for all pairs
            if unique_pairs:
                unique_pairs_list = list(unique_pairs)
                item_ids = torch.LongTensor([p[0] for p in unique_pairs_list]).to(args.device)
                steps = [p[1] for p in unique_pairs_list]
                
                # Group by step for batch computation
                step_to_items = {}
                for item_id, step in unique_pairs_list:
                    if step not in step_to_items:
                        step_to_items[step] = []
                    step_to_items[step].append(item_id)

                for step, items in tqdm(step_to_items.items(), desc="Pre-calculating spike scores"):
                    item_tensor = torch.LongTensor(items).to(args.device)
                    # Assuming _get_spike_bias_for_items returns the raw scores before scaling
                    scores = model._get_spike_bias_for_items(item_tensor, int(step), raw_score=True)
                    for item_id, score in zip(items, scores.cpu().numpy()):
                        all_spike_scores[(item_id, step)] = float(score)
            
            print(f"[slice_eval] Calculated {len(all_spike_scores)} spike scores.")


    # Validation에만 timestamp 로깅 시작
    is_validation = phase and phase.lower().startswith('val')
    validation_start_time = None
    # feature cache (disk) support for spike or other heavy features
    feature_cache_path = None
    if is_validation:
        validation_start_time = time.perf_counter()
        # args에 phase 정보 저장 (compute_rank에서 사용)
        args._validation_phase = phase
        _log_validation_timestamp(phase, "=== Validation started ===")
        _log_validation_timestamp(phase, f"Total batches: {len(_loader)}, maxit: {maxit}")
        # Pre-compute spike features for validation (if model uses spike features)
        uses_spike = (
            getattr(args, "viewer_feat_mode", "off") == "spike"
            or hasattr(model, "spike_embedding")
            or hasattr(model, "spike_mlp")
        )
        # SpikeScoreBiasSASRec 모델 확인
        uses_spike_bias = hasattr(model, "spike_bias_weight") or model.__class__.__name__ == "SpikeScoreBiasSASRec"
        
        # Pre-compute timeout 설정 (작은 maxit에서도 빠른 캐시 생성 허용)
        precompute_timeout = 20.0  # 20초 timeout
        # val_maxit가 작아도 캐시가 없으면 최소한의 pre-compute를 수행해 러닝타임 단축
        should_precompute = maxit is None or maxit <= 2000
        
        if uses_spike and hasattr(args, 'viewer_trends') and args.viewer_trends is not None:
            # Disk cache path
            cache_dir = Path(getattr(args, "cache_dir", "cache/"))
            cache_dir.mkdir(parents=True, exist_ok=True)
            dataset_key = _dataset_cache_key(Path(getattr(args, 'dataset', 'data')))
            cache_name = f"spike_cache_{dataset_key}_{_spike_cache_key(args)}_{phase or 'val'}"
            feature_cache_path = cache_dir / f"{cache_name}.npz"

            if feature_cache_path.exists():
                load_start = time.perf_counter()
                npz = np.load(feature_cache_path)
                item_ids = npz["item_ids"]
                steps = npz["steps"]
                feats = npz["feats"]
                if feats.ndim == 1:
                    feats = feats.reshape(-1, 1)
                if feats.shape[1] == 3:
                    pad = np.zeros((feats.shape[0], 1), dtype=feats.dtype)
                    feats = np.concatenate([feats, pad], axis=1)
                try:
                    model.validation_spike_cache_items = torch.from_numpy(item_ids).to(args.device)
                    model.validation_spike_cache_steps = torch.from_numpy(steps).to(args.device)
                    model.validation_spike_cache_feats = torch.from_numpy(feats).to(args.device)
                    model.validation_spike_cache = {}
                except Exception:
                    validation_spike_cache = {(int(i), int(s)): feats[idx] for idx, (i, s) in enumerate(zip(item_ids, steps))}
                    model.validation_spike_cache = validation_spike_cache
                load_time = time.perf_counter() - load_start
                cache_size = int(len(item_ids))
                _log_validation_timestamp(phase, f"[cache] Loaded spike cache from {feature_cache_path} ({cache_size:,} entries, {load_time:.2f}s)")
            else:
                if should_precompute:
                    _log_validation_timestamp(phase, "Pre-computing spike features for validation...")
                    precompute_start = time.perf_counter()

                    # Collect all unique (item_id, timestep) pairs (respect maxit) - VECTORIZED
                    unique_pairs = set()
                    batch_count = 0
                    for i, data in enumerate(_loader):
                        # Timeout 체크
                        if time.perf_counter() - precompute_start > precompute_timeout:
                            _log_validation_timestamp(phase, f"Pre-compute timeout ({precompute_timeout}s), using partial cache")
                            break
                        
                        xtsy = data[:,:,6]  # timesteps [B, seq_len]
                        steps_batch = xtsy[:,-1].cpu().numpy().astype(np.int32)  # [B]
                        
                        # Vectorized: collect all (item, step) pairs for this batch
                        for b_idx, step in enumerate(steps_batch):
                            if step in args.ts:
                                available_items = args.ts[step]
                                # Use set update for faster bulk addition
                                unique_pairs.update((item, int(step)) for item in available_items)
                        
                        batch_count += 1
                        if maxit is not None and i >= maxit:
                            break

                        # quick exit for very small val_maxit to avoid over-precompute
                        if maxit is not None and len(unique_pairs) > maxit * 200:
                            # heuristic: assume ~200 candidates per batch on average
                            break

                _log_validation_timestamp(phase, f"Found {len(unique_pairs):,} unique (item, timestep) pairs")

                # Batch pre-compute spike features - VECTORIZED
                if len(unique_pairs) > 0:
                    id_to_streamer = getattr(args, 'id_to_streamer', {})
                    # Vectorized: filter valid pairs and build lists in one pass
                    unique_pairs_list = list(unique_pairs)
                    streamer_timestamp_pairs = []
                    valid_indices = []
                    
                    for idx, (item_id, step) in enumerate(unique_pairs_list):
                        streamer_name = id_to_streamer.get(item_id, None)
                        if streamer_name:
                            streamer_timestamp_pairs.append((streamer_name, step))
                            valid_indices.append(idx)

                    _log_validation_timestamp(phase, f"Computing spike features for {len(streamer_timestamp_pairs):,} pairs...")
                    spike_feats_array = args.viewer_trends.compute_spike_features_batch(
                        streamer_timestamp_pairs,
                        use_percentile=bool(getattr(args, "use_percentile_features", False)),
                        use_percentile_3d=bool(getattr(args, "use_percentile_3d", False)),
                        use_hybrid=bool(getattr(args, "use_hybrid_features", False)),
                        window_size=getattr(args, "ablation_window_size", None),
                    )

                    # Build cache dictionary - VECTORIZED
                    validation_spike_cache = {}
                    item_ids = []
                    steps = []
                    feats = []
                    # Pre-allocate lists for better performance
                    num_valid = len(valid_indices)
                    item_ids = [unique_pairs_list[valid_indices[i]][0] for i in range(num_valid)]
                    steps = [unique_pairs_list[valid_indices[i]][1] for i in range(num_valid)]
                    feats = [spike_feats_array[i] for i in range(num_valid)]
                    
                    # Build cache dict in one pass
                    for i in range(num_valid):
                        item_id = item_ids[i]
                        step = steps[i]
                        feat = feats[i]
                        validation_spike_cache[(item_id, step)] = feat

                    # tensorized cache for fast lookup (avoid list→tensor per batch)
                    try:
                        feats_np = np.asarray(feats, dtype=np.float32)
                        ids_np = np.asarray(item_ids, dtype=np.int32)
                        steps_np = np.asarray(steps, dtype=np.int32)
                        model.validation_spike_cache = validation_spike_cache
                        model.validation_spike_cache_items = torch.from_numpy(ids_np).to(args.device)
                        model.validation_spike_cache_steps = torch.from_numpy(steps_np).to(args.device)
                        model.validation_spike_cache_feats = torch.from_numpy(feats_np).to(args.device)
                    except Exception:
                        model.validation_spike_cache = validation_spike_cache

                    # Save to disk for reuse
                    try:
                        np.savez_compressed(feature_cache_path, item_ids=np.array(item_ids, dtype=np.int32),
                                            steps=np.array(steps, dtype=np.int32),
                                            feats=np.array(feats, dtype=np.float32))
                        _log_validation_timestamp(phase, f"[cache] Saved spike cache -> {feature_cache_path}")
                    except Exception as cache_exc:
                        _log_validation_timestamp(phase, f"[cache] Save failed: {cache_exc}")

                    precompute_time = time.perf_counter() - precompute_start
                    _log_validation_timestamp(phase, f"Pre-compute complete: {len(validation_spike_cache):,} entries cached ({precompute_time:.2f}s)")
                else:
                    model.validation_spike_cache = {}
        
        # Pre-compute spike bias for SpikeScoreBiasSASRec (cache만 사용, pre-compute는 skip)
        if uses_spike_bias and hasattr(args, 'viewer_trends') and args.viewer_trends is not None:
            cache_dir = Path(getattr(args, "cache_dir", "cache/"))
            cache_dir.mkdir(parents=True, exist_ok=True)
            dataset_key = _dataset_cache_key(Path(getattr(args, 'dataset', 'data')))
            cache_name = f"spike_bias_cache_{dataset_key}_{_spike_cache_key(args)}_{phase or 'val'}"
            bias_cache_path = cache_dir / f"{cache_name}.npz"
            
            force_rebuild = maxit is not None and maxit <= 200  # 작은 maxit에서는 캐시 커버리지 보장을 위해 항상 재계산
            if bias_cache_path.exists() and not force_rebuild:
                load_start = time.perf_counter()
                npz = np.load(bias_cache_path)
                item_ids = npz["item_ids"]
                steps = npz["steps"]
                biases = npz["biases"]
                
                # Tensorized cache for fast lookup
                ids_np = np.asarray(item_ids, dtype=np.int32)
                steps_np = np.asarray(steps, dtype=np.int32)
                biases_np = np.asarray(biases, dtype=np.float32)
                
                class _BiasCacheView:
                    pass
                bias_cache = _BiasCacheView()
                bias_cache.items = torch.from_numpy(ids_np).to(args.device)
                bias_cache.steps = torch.from_numpy(steps_np).to(args.device)
                bias_cache.biases = torch.from_numpy(biases_np).to(args.device)
                model.validation_bias_cache = bias_cache
                
                load_time = time.perf_counter() - load_start
                _log_validation_timestamp(phase, f"[cache] Loaded spike bias cache from {bias_cache_path} ({len(item_ids):,} entries, {load_time:.2f}s)")
            else:
                # Pre-compute bias cache if needed
                if should_precompute:
                    _log_validation_timestamp(phase, f"[cache] Spike bias cache not found, pre-computing...")
                    precompute_bias_start = time.perf_counter()
                    
                    # Collect unique (item_id, timestep) pairs (reuse from spike features if available)
                    unique_pairs = set()
                    # 1) 항상 validation loader의 timesteps에 대응하는 모든 availability 아이템을 수집 (maxit 범위 내)
                    batch_count = 0
                    for i, data in enumerate(_loader):
                        if time.perf_counter() - precompute_bias_start > precompute_timeout:
                            _log_validation_timestamp(phase, f"Bias pre-compute timeout ({precompute_timeout}s), using partial cache")
                            break
                        if maxit is not None and i >= maxit:
                            break
                        
                        xtsy = data[:,:,6]  # timesteps [B, seq_len]
                        steps_batch = xtsy[:,-1].cpu().numpy().astype(np.int32)  # [B]
                        for step in steps_batch:
                            if step in args.ts:
                                available_items = args.ts[step]
                                unique_pairs.update((item, int(step)) for item in available_items)
                        batch_count += 1
                    
                    # 2) spike feature cache가 있으면 거기 포함된 pair들도 포함 (중복 무시)
                    if hasattr(model, 'validation_spike_cache') and model.validation_spike_cache:
                        if hasattr(model, 'validation_spike_cache_items') and hasattr(model, 'validation_spike_cache_steps'):
                            items_np = model.validation_spike_cache_items.cpu().numpy()
                            steps_np = model.validation_spike_cache_steps.cpu().numpy()
                            unique_pairs.update((int(items_np[i]), int(steps_np[i])) for i in range(len(items_np)))
                        else:
                            unique_pairs.update(set(model.validation_spike_cache.keys()))
                    
                    _log_validation_timestamp(phase, f"Computing spike bias for {len(unique_pairs):,} unique (item, timestep) pairs...")
                    
                    # Compute biases using model's method (batch by step for efficiency)
                    id_to_streamer = getattr(args, 'id_to_streamer', {})
                    unique_pairs_list = list(unique_pairs)
                    
                    # Group by step to minimize redundant lookups
                    step_to_items = {}
                    for item_id, step in unique_pairs_list:
                        if step not in step_to_items:
                            step_to_items[step] = []
                        step_to_items[step].append(item_id)
                    
                    # Compute biases step by step (vectorized within each step)
                    bias_item_ids = []
                    bias_steps = []
                    bias_values = []
                    
                    for step, item_list in step_to_items.items():
                        item_tensor = torch.LongTensor(item_list).to(args.device)
                        biases = model._get_spike_bias_for_items(item_tensor, int(step))
                        biases_cpu = biases.cpu().numpy()
                        
                        for idx, item_id in enumerate(item_list):
                            bias_item_ids.append(item_id)
                            bias_steps.append(step)
                            bias_values.append(float(biases_cpu[idx]))
                    
                    # Save to disk
                    try:
                        np.savez_compressed(bias_cache_path, 
                                          item_ids=np.array(bias_item_ids, dtype=np.int32),
                                          steps=np.array(bias_steps, dtype=np.int32),
                                          biases=np.array(bias_values, dtype=np.float32))
                        
                        # Load into model (same as cache hit path)
                        ids_np = np.asarray(bias_item_ids, dtype=np.int32)
                        steps_np = np.asarray(bias_steps, dtype=np.int32)
                        biases_np = np.asarray(bias_values, dtype=np.float32)
                        
                        class _BiasCacheView:
                            pass
                        bias_cache = _BiasCacheView()
                        bias_cache.items = torch.from_numpy(ids_np).to(args.device)
                        bias_cache.steps = torch.from_numpy(steps_np).to(args.device)
                        bias_cache.biases = torch.from_numpy(biases_np).to(args.device)
                        model.validation_bias_cache = bias_cache
                        
                        precompute_bias_time = time.perf_counter() - precompute_bias_start
                        _log_validation_timestamp(phase, f"[cache] Pre-computed and saved spike bias cache -> {bias_cache_path} ({len(bias_item_ids):,} entries, {precompute_bias_time:.2f}s)")
                    except Exception as cache_exc:
                        _log_validation_timestamp(phase, f"[cache] Bias cache save failed: {cache_exc}")
                        model.validation_bias_cache = None
                else:
                    _log_validation_timestamp(phase, f"[cache] Spike bias cache not found, skipping pre-compute (maxit={maxit})")
                    model.validation_bias_cache = None

    else:
        # validation이 아닐 때는 phase 정보 제거
        if hasattr(args, '_validation_phase'):
            delattr(args, '_validation_phase')
        if hasattr(model, 'validation_spike_cache'):
            delattr(model, 'validation_spike_cache')

    model.eval()
    with torch.no_grad():
        batch_times = []
        for i,data in tqdm(enumerate(_loader)):
            if is_validation:
                batch_start = time.perf_counter()
            
            data_load_start = time.perf_counter()
            data = data.to(args.device)
            data_load_time = time.perf_counter() - data_load_start
            if data_load_time > 0.01:
                _log_validation_timestamp(phase, f"[timing] Batch {i} data load: {data_load_time:.3f}s")
            
            if is_validation:
                batch_start_time = time.perf_counter()
            
            compute_start = time.perf_counter()
            candidates = build_step_candidates(args, data[:, :, 6])
            # Pass all_spike_scores to compute_rank
            store = model.compute_rank(
                data,
                store,
                k=10,
                candidates=candidates,
                all_spike_scores=all_spike_scores,
                eval_slice=eval_slice,
            )
            compute_time = time.perf_counter() - compute_start
            
            if is_validation:
                batch_time = time.perf_counter() - batch_start_time
                batch_times.append(batch_time)
                # Log timing details for slow batches
                if batch_time > 0.5 or i % 50 == 0:
                    avg_so_far = sum(batch_times) / len(batch_times)
                    _log_validation_timestamp(phase, f"Progress: {i}/{min(maxit or 999999, len(_loader))} batches, avg: {avg_so_far:.3f}s/batch (compute: {compute_time:.3f}s, load: {data_load_time:.3f}s)")
            
            if maxit is not None and i >= maxit: break
    
    if is_validation:
        validation_total_time = time.perf_counter() - validation_start_time
        avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0.0
        _log_validation_timestamp(phase, f"=== Validation completed ===")
        _log_validation_timestamp(phase, f"Total batches processed: {len(batch_times)}")
        _log_validation_timestamp(phase, f"Average batch time: {avg_batch_time:.4f}s")
        _log_validation_timestamp(phase, f"Total validation time: {validation_total_time:.4f}s")
        # phase 정보 제거
        if hasattr(args, '_validation_phase'):
            delattr(args, '_validation_phase')

    result = {
        'rep': metrics(store['rrep']),
        'new': metrics(store['rnew']),
        'all': metrics(store['rall']),
        'ratio': np.mean(store['ratio']) if store['ratio'] else 0.0,
        'rep_count': len(store['rrep']),
        'new_count': len(store['rnew']),
        'all_count': len(store['rall']),
    }
    if store['new_ratio_topk']:
        ratios = np.array(store['new_ratio_topk'])
        result['new_ratio_k10'] = float(ratios.mean())

    tier_stats = store.get('tier_stats', {})
    if tier_stats:
        tier_result = {}
        for tier_label, stats in tier_stats.items():
            entry = {}
            if stats.get('all'):
                entry['all'] = metrics(stats['all'])
                entry['all_count'] = len(stats['all'])
            if stats.get('rep'):
                entry['rep'] = metrics(stats['rep'])
                entry['rep_count'] = len(stats['rep'])
            if stats.get('new'):
                entry['new'] = metrics(stats['new'])
                entry['new_count'] = len(stats['new'])
            tier_result[tier_label] = entry
        result['tier_breakdown'] = tier_result

    if getattr(args, 'debug', False):
        try:
            rec = {
                'time': datetime.utcnow().isoformat(),
                'phase': phase,
                'rep_count': result['rep_count'],
                'new_count': result['new_count'],
                'all_count': result['all_count'],
                'ratio': float(result['ratio']),
            }
            EVAL_SCOPE_DEBUG.parent.mkdir(parents=True, exist_ok=True)
            with open(EVAL_SCOPE_DEBUG, 'a') as fj:
                fj.write(json.dumps(rec) + "\n")
        except Exception:
            pass

    if dump_ranks:
        _write_rank_dump(args, store.get('rank_details', []), phase)

    return result

def compute_rank(data,store,k=10):
   inputs,pos,_ = convert_batch(data,self.args,sample_neg=False)

   feats = self(inputs,data)
   
   xtsy = torch.zeros_like(pos)
   xtsy[data.x_s_batch,data.x_s[:,3]] = data.xts
   
   if self.args.fr_ctx:
       ctx,batch_inds = self.get_ctx_att(data,inputs,feats)

   if self.args.fr_ctx==False and self.args.fr_rep==True:
       rep_enc = self.rep_emb(self.get_av_rep(inputs,data))
   
   # identify repeated interactions in the batch 
   mask = torch.ones_like(pos[:,-1]).type(torch.bool)
   for b in range(pos.shape[0]):
       avt = pos[b,:-1]
       avt = avt[avt!=0]
       mask[b] = pos[b,-1] in avt
       store['ratio'] += [float(pos[b,-1] in avt)]
       
   for b in range(inputs.shape[0]):
       step = xtsy[b,-1].item()
       av = torch.LongTensor(self.args.ts[step]).to(self.args.device)
       av_embs = self.item_embedding(av)
 
       if self.args.fr_ctx==False and self.args.fr_rep:         
           # get rep
           reps = inputs[b,inputs[b,:]!=0].unsqueeze(1)==av
           a = (step-xtsy[b,inputs[b,:]!=0]).unsqueeze(1).repeat(1,len(av)) * reps
           if torch.any(torch.any(reps,1)):
               a = a[torch.any(reps,1),:]
               a[a==0]=99999
               a = a.min(0).values*torch.any(reps,0)
               sm  = torch.bucketize(a, self.boundaries)+1
               sm  = sm*torch.any(reps,0)
               sm  = self.rep_emb(sm) 
               av_embs += sm
     
       if self.args.fr_ctx:         
           ctx_expand = torch.zeros(self.args.av_tens.shape[1],self.args.K,device=self.args.device)
           ctx_expand[batch_inds[b,-1,:],:] = ctx[b,-1,:,:]
           scores  = (feats[b,-1,:] * ctx_expand).sum(-1) 
           scores  = scores[:len(av)]
       else:
           scores  = (feats[b,-1,:] * av_embs).sum(-1) 

       iseq = pos[b,-1] == av
       idx  = torch.where(iseq)[0]
       rank = torch.where(torch.argsort(scores, descending=True)==idx)[0].item()

       if mask[b]: # rep
           store['rrep'] += [rank]
       else:
           store['rnew'] += [rank]
       store['rall'] += [rank]
   
   return store


def main():
    args = arguments_cli.arg_parse()
    config_path = getattr(args, "config", "") or ""
    if config_path:
        overrides = load_config_dict(config_path)
        apply_config_overrides(args, overrides)
    ensure_log_paths(args)
    arguments_cli.print_args(args)
    args.device = torch.device(args.device)

    checkpoint = getattr(args, "checkpoint", "") or ""
    if not checkpoint:
        raise ValueError("Evaluation requires --checkpoint to load weights")

    # Add new CLI argument for slicing
    if not hasattr(args, 'eval_slice'):
        args.eval_slice = None # e.g., 'spike_high', 'spike_low'

    data_fu = load_data(args)
    _, val_loader, test_loader = get_dataloaders(data_fu, args)
    _, model_cls = get_model_type(args)

    model = model_cls(args).to(args.device)
    try:
        state = torch.load(checkpoint, map_location=args.device)
        model.load_state_dict(state)
    except Exception as exc:
        print(f"[warn] strict load failed: {exc} -> retrying with strict=False")
        state = torch.load(checkpoint, map_location=args.device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        try:
            if missing or unexpected:
                print(f"[warn] missing keys: {missing}, unexpected: {unexpected}")
        except Exception:
            pass

    split = getattr(args, "eval_split", "test")
    if split == "val":
        target_loader = val_loader
    else:
        target_loader = test_loader

    # 옵션: 평가 배치 수 제한 (timeout 방지를 위한 soft cap)
    eval_maxit = getattr(args, "eval_maxit", None)
    scores = compute_recall(
        model,
        target_loader,
        args,
        maxit=eval_maxit,
        phase=split,
        eval_slice=args.eval_slice,
    )
    print("Evaluation results ({}, slice: {})".format(split, args.eval_slice or 'none'))
    print("=" * 11)
    print_scores(scores)
    save_scores(scores, args, checkpoint_path=checkpoint)


if __name__ == "__main__":
    main()
