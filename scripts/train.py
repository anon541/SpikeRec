"""Training entrypoint for LiveRec/SASRec baselines with config support."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
import torch.optim as optim
from tqdm import tqdm
import pickle

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.loader import get_dataloaders, load_data  # noqa: E402
from models.registry import get_model_type  # noqa: E402
from scripts import arguments as arguments_cli  # noqa: E402
from scripts.config_utils import (  # noqa: E402
    DEFAULT_LOG_TXT,
    apply_config_overrides,
    ensure_log_paths,
    load_config_dict,
)
from scripts.eval import compute_recall, print_scores, save_scores  # noqa: E402


def _normalize_early_stop_metric(name: str) -> str:
    """
    Normalize user-friendly early-stop metric names to keys used in scores['all'].

    Supported examples:
      - 'h01', 'hit1', 'hit@1', '1'   -> 'h01'
      - 'h05', 'hit5', 'hit@5', '5'   -> 'h05'
      - 'h10', 'hit10', 'hit@10', '10'-> 'h10'
      - 'ndcg01', 'ndcg1'             -> 'ndcg01'
      - 'ndcg05', 'ndcg5'             -> 'ndcg05'
      - 'ndcg10'                      -> 'ndcg10'
    """
    if not name:
        return "h01"
    x = str(name).strip().lower()

    # strip common prefixes
    if x.startswith("hit@"):
        x = x[4:]
    elif x.startswith("hit"):
        x = x[3:]
    elif x.startswith("ndcg@"):
        x = "ndcg" + x[5:]

    # allow raw keys (h01, h05, h10, ndcg01, ...)
    if x in {"h01", "h05", "h10", "ndcg01", "ndcg05", "ndcg10"}:
        return x

    # numeric-only -> map to hXX
    if x in {"1", "01"}:
        return "h01"
    if x in {"5", "05"}:
        return "h05"
    if x in {"10"}:
        return "h10"

    # ndcg variants
    if x in {"ndcg1", "ndcg01"}:
        return "ndcg01"
    if x in {"ndcg5", "ndcg05"}:
        return "ndcg05"
    if x in {"ndcg10"}:
        return "ndcg10"

    # fallback: original string if it matches a known metric key pattern,
    # otherwise default to h01 to avoid KeyError.
    if x in {"h1", "h5", "h10"}:
        return {"h1": "h01", "h5": "h05", "h10": "h10"}[x]

    return "h01"


def _print_dry_run_summary_fast(args, checkpoint_path: str) -> None:
    """Fast dry-run summary with minimal data loading for validation."""
    dataset_path = getattr(args, "dataset", "dataset")
    batch_size = getattr(args, 'batch_size', 128)
    
    # Load only a tiny sample to validate data format (100 rows)
    try:
        from data.loader import load_data
        # Temporarily set debug_nrows for fast validation
        original_debug_nrows = getattr(args, 'debug_nrows', 0)
        args.debug_nrows = 100  # Load only 100 rows for validation
        data_fu = load_data(args)
        args.debug_nrows = original_debug_nrows  # Restore
        
        print("[dry-run] dataset path: {}".format(dataset_path))
        print("[dry-run] sample loaded: {} rows, {} users, {} streamers".format(
            len(data_fu), data_fu.user.nunique(), data_fu.streamer.nunique()
        ))
        print("[dry-run] batch_size: {}".format(batch_size))
        print("[dry-run] seq_len: {}".format(getattr(args, 'seq_len', 20)))
        print("[dry-run] model: {}".format(getattr(args, 'model', 'Unknown')))
        print(f"[dry-run] checkpoint path -> {checkpoint_path}")
        print("[dry-run] ✓ Config validation passed - ready to train")
    except Exception as e:
        # If data loading fails, still show config info
        print("[dry-run] dataset path: {}".format(dataset_path))
        print("[dry-run] batch_size: {}".format(batch_size))
        print("[dry-run] seq_len: {}".format(getattr(args, 'seq_len', 20)))
        print("[dry-run] model: {}".format(getattr(args, 'model', 'Unknown')))
        print(f"[dry-run] checkpoint path -> {checkpoint_path}")
        print(f"[dry-run] ⚠ Data loading failed: {e}")
        print("[dry-run] Config validation passed, but data loading needs fixing")


def main():
    args = arguments_cli.arg_parse()
    config_path = getattr(args, "config", "") or ""
    if config_path:
        overrides = load_config_dict(config_path)
        apply_config_overrides(args, overrides)
    ensure_log_paths(args)
    arguments_cli.print_args(args)
    args.device = torch.device(args.device)

    checkpoint_path, model_cls = get_model_type(args)
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    # Dry-run: run 1 epoch with minimal data to catch training/eval errors
    # IMPORTANT: This must use the same code path as actual training to catch errors
    if getattr(args, "dry_run", False):
        print("[dry-run] Running 1 epoch with minimal data to validate training pipeline...")
        
        # Set minimal data limits for dry-run
        original_debug_nrows = getattr(args, 'debug_nrows', 0)
        original_debug_max_users = getattr(args, 'debug_max_users', 0)
        original_num_epochs = getattr(args, 'num_epochs', 200)
        original_val_maxit = getattr(args, 'val_maxit', None)
        original_caching = getattr(args, 'caching', False)
        original_viewer_trends_path = getattr(args, 'viewer_trends_path', None)
        original_viewer_feat_mode = getattr(args, 'viewer_feat_mode', 'off')
        
        # Very minimal settings for fast dry-run
        args.debug_nrows = 200  # Load only 200 rows (reduced from 500)
        args.debug_max_users = 5  # Limit to 5 users (reduced from 20)
        args.num_epochs = 1  # Run only 1 epoch
        args.val_maxit = 3  # Validate on only 3 batches (reduced from 5)
        args.caching = False  # Disable caching for dry-run (faster)
        args.dry_run = True  # Mark as dry-run to skip expensive operations in loader
        
        # For viewer spike models, load a small sample of viewer trends
        dry_run_viewer_trends_sample = (original_viewer_feat_mode == 'spike' and original_viewer_trends_path)
        
        if dry_run_viewer_trends_sample:
            # Keep viewer_feat_mode enabled but skip loading trends in load_data
            # We'll create dummy trends after data loading
            args.viewer_trends_path = None  # Skip loading in load_data
            args._dry_run_max_trend_rows = 5000  # Limit to 5k rows for faster dry-run
            print(f"[dry-run] Will create dummy viewer trends after data loading")
        else:
            # Skip viewer trends loading for non-spike models
            args.viewer_trends_path = None
            # Keep viewer_feat_mode for bucket models (they don't need trends)
            # Only disable for spike models that need trends
            if original_viewer_feat_mode == 'spike':
                args.viewer_feat_mode = 'off'
                print(f"[dry-run] Temporarily disabling viewer_feat_mode ({original_viewer_feat_mode} -> off) for fast validation")
            # For bucket mode, keep it enabled (no trends needed)
        
        try:
            # For viewer spike models, create dummy viewer trends before load_data
            if dry_run_viewer_trends_sample:
                print("[dry-run] Creating minimal dummy viewer trends...")
                from data.loader import ViewerTrendLookup
                args.viewer_trends = ViewerTrendLookup(None)
                # Create minimal dummy trends - will be populated after data loading
                args.viewer_trends.trends = {}
                print("[dry-run] Dummy viewer trends initialized (will be populated after data loading)")
            
            # Load data with minimal limits
            print("[dry-run] Loading data...")
            data_fu = load_data(args)
            print(f"[dry-run] Loaded {len(data_fu)} rows, {data_fu.user.nunique()} users, {data_fu.streamer.nunique()} streamers")
            topk_att = getattr(args, "topk_att", None)
            if topk_att is not None and getattr(args, "max_avail", 0):
                args.topk_att = min(int(topk_att), int(args.max_avail))
            
            # After loading, populate dummy viewer trends with actual streamers
            if dry_run_viewer_trends_sample and hasattr(data_fu, 'streamer_raw'):
                sample_streamers = set(data_fu['streamer_raw'].unique())
                print(f"[dry-run] Populating dummy viewer trends for {len(sample_streamers)} streamers...")
                from data.loader import ViewerTrendLookup
                import random
                # Recreate ViewerTrendLookup if it was set to None by load_data
                if args.viewer_trends is None:
                    args.viewer_trends = ViewerTrendLookup(None)
                dummy_trends = {}
                max_step = getattr(args, 'max_step', 6000)
                for streamer in sample_streamers:
                    for ts in range(0, max_step + 1, 100):  # Every 100 timesteps
                        dummy_trends[(streamer, ts)] = random.randint(10, 1000)
                args.viewer_trends.trends = dummy_trends
                print(f"[dry-run] Created {len(dummy_trends):,} dummy viewer trend entries")
                # Set streamer mapping
                if hasattr(args, 'id_to_streamer'):
                    umap = {v: k for k, v in args.id_to_streamer.items()}
                    args.viewer_trends.set_streamer_mapping(umap, args.id_to_streamer)
            
            print("[dry-run] Creating data loaders...")
            loaders = get_dataloaders(data_fu, args)
            train_loader, val_loader, test_loader = loaders
            
            print(f"[dry-run] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
            
            # Initialize model
            print("[dry-run] Initializing model...")
            model = model_cls(args).to(args.device)
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
            
            print("[dry-run] Starting training (max 5 batches)...")
            
            # Run minimal training (max 5 batches)
            model.train()
            loss_all = 0.0
            loss_cnt = 0
            
            for batch_idx, data in enumerate(train_loader):
                data = data.to(args.device)
                loss = model.train_step(data)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_all += loss.item()
                loss_cnt += 1
                
                # Limit to 5 batches for dry-run (reduced from 10)
                if batch_idx >= 4:
                    break
            
            print(f"[dry-run] Training completed: {loss_cnt} batches, avg loss: {loss_all / max(1, loss_cnt):.5f}")
            
            # Run validation - use exact same code path as actual training loop
            print("[dry-run] Running validation (max 3 batches)...")
            # IMPORTANT: Use the exact same call as line 407 in actual training loop
            # This ensures we catch any import/scope issues
            try:
                scores = compute_recall(model, val_loader, args, maxit=getattr(args, 'val_maxit', 500), phase='val')
                print("[dry-run] Validation scores:")
                print_scores(scores)
                
                print("[dry-run] ✓ Training pipeline validation passed!")
            except UnboundLocalError as e:
                print(f"[dry-run] ✗ UnboundLocalError caught: {e}")
                print("[dry-run] This indicates a scope/import issue in the actual training code path")
                import traceback
                traceback.print_exc()
                raise
            
        except Exception as e:
            print(f"[dry-run] ✗ Error during training pipeline validation:")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # Restore original values
            args.debug_nrows = original_debug_nrows
            args.debug_max_users = original_debug_max_users
            args.num_epochs = original_num_epochs
            args.val_maxit = original_val_maxit
            args.caching = original_caching
            args.viewer_trends_path = original_viewer_trends_path
            args.viewer_feat_mode = original_viewer_feat_mode
            args.dry_run = False
        
        return
    
    # Normal training: load data and generate sequences
    print("[train] Loading data...")
    t0 = time.perf_counter()
    data_fu = load_data(args)
    t1 = time.perf_counter()
    print(f"[train] Data loaded in {t1-t0:.2f}s. Creating data loaders...")
    t2 = time.perf_counter()
    loaders = get_dataloaders(data_fu, args)
    t3 = time.perf_counter()
    train_loader, val_loader, test_loader = loaders
    print(f"[train] Data loaders ready in {t3-t2:.2f}s (total: {t3-t0:.2f}s).")
    
    # Val-only mode: quick validation check without training
    if getattr(args, 'val_only', False):
        print("[train] VAL-ONLY mode: loading checkpoint for validation...")
        if not args.checkpoint or not os.path.exists(args.checkpoint):
            print(f"[ERROR] Val-only mode requires --checkpoint, got: {args.checkpoint}")
            return
        
        print(f"[train] Loading checkpoint: {args.checkpoint}")
        model = model_cls(args).to(args.device)
        ckpt_obj = torch.load(args.checkpoint, map_location=args.device)
        if isinstance(ckpt_obj, dict) and 'model_state_dict' in ckpt_obj:
            state_dict = ckpt_obj['model_state_dict']
        else:
            state_dict = ckpt_obj
        model.load_state_dict(state_dict)
        print("[train] Checkpoint loaded. Running validation...")
        
        scores = compute_recall(model, val_loader, args, maxit=getattr(args, 'val_maxit', 500), phase='val')
        print("\n" + "=" * 50)
        print("VALIDATION RESULTS (val-only mode)")
        print("=" * 50)
        print_scores(scores)
        print("=" * 50)
        return

    if args.model in ['REP']:
        data_tr = data_fu[data_fu.stop < args.pivot_1]
        model = model_cls(args, data_tr)
        scores = compute_recall(model, test_loader, args)
        print("Final score")
        print("=" * 11)
        print_scores(scores)
        return

    print(f"[train] Initializing model: {args.model}")
    model = model_cls(args).to(args.device)
    # Optional: resume weights from checkpoint if provided (for fine-tuning / diagnostics)
    resume_path = getattr(args, "checkpoint", "") or ""
    if resume_path:
        try:
            if os.path.exists(resume_path):
                print(f"[train] Resuming model weights from checkpoint: {resume_path}")
                ckpt_obj = torch.load(resume_path, map_location=args.device)
                if isinstance(ckpt_obj, dict) and 'model_state_dict' in ckpt_obj:
                    state_dict = ckpt_obj['model_state_dict']
                else:
                    state_dict = ckpt_obj
                model.load_state_dict(state_dict)
            else:
                print(f"[train] Warning: checkpoint not found at {resume_path}, training from scratch.")
        except Exception as resume_exc:
            print(f"[train] Warning: failed to load checkpoint {resume_path}: {resume_exc}")
            print("[train] Continuing training from scratch.")
    print(f"[train] Model initialized. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    print(f"[train] Optimizer ready. Starting training...")

    # Early stopping: track best validation metric and patience counter.
    # Metric is configurable via args.early_stop_metric, defaulting to Hit@1 ('h01')
    # to preserve existing behavior.
    early_metric_key = _normalize_early_stop_metric(
        getattr(args, "early_stop_metric", "h01")
    )
    best_val = float("-inf")
    best_max = args.early_stop
    best_cnt = best_max

    print("training...")

    # 전체 학습에서는 timeout 비활성화, 속도 테스트에서는 5분 제한
    if getattr(args, 'disable_timeout', False):
        MAX_RUNTIME_SECONDS = float('inf')  # 무제한
        print("[timeout] Timeout disabled for full training")
    else:
        MAX_RUNTIME_SECONDS = 300  # 5분 제한 (speed test용)
        print(f"[timeout] Timeout enabled: {MAX_RUNTIME_SECONDS}s limit")

    training_start_time = time.perf_counter()
    
    for epoch in range(args.num_epochs):
        epoch_start_time = time.perf_counter()
        loss_all = 0.0
        loss_cnt = 0
        if getattr(args, 'debug', False):
            model._dbg = {
                'model': model.__class__.__name__,
                'batches': 0,
                'rep_batches_sum': 0,
                'new_batches_sum': 0,
                'selected_batches_sum': 0,
                'tokens_all_sum': 0,
                'tokens_sel_sum': 0,
            }
            if hasattr(model, 'reset_gate_debug'):
                try:
                    model.reset_gate_debug()
                except Exception:
                    pass
            if hasattr(model, 'reset_state_debug'):
                try:
                    model.reset_state_debug()
                except Exception:
                    pass
        model.train()
        bench_every = 50
        bn = 0
        t_data = t_step = t_bwd = t_opt = t_tot = 0.0
        train_batch_start = time.perf_counter()
        for data in tqdm(train_loader):
            # 5분 제한 체크 (매 10 배치마다만 체크하여 오버헤드 최소화)
            if bn % 10 == 0:
                elapsed = time.perf_counter() - training_start_time
                if elapsed > MAX_RUNTIME_SECONDS:
                    print(f"\n[timeout] Training exceeded {MAX_RUNTIME_SECONDS}s limit. Stopping at batch {bn} of epoch {epoch}.")
                    print(f"[timeout] Elapsed: {elapsed:.2f}s, Training time: {time.perf_counter() - train_batch_start:.2f}s")
                    return
            ts = time.perf_counter()
            data = data.to(args.device)
            t1 = time.perf_counter()
            optimizer.zero_grad()

            loss = model.train_step(data)
            t2 = time.perf_counter()

            loss_all += loss.item()
            # Count valid samples: loss is computed over all timesteps where inputs != 0
            # valid_mask = (inputs != 0) includes all timesteps, so count all valid targets
            loss_cnt += (data[:, :, 5] != 0).sum()  # Count all valid timesteps (targets != 0)

            loss.backward()
            t3 = time.perf_counter()
            optimizer.step()
            t4 = time.perf_counter()

            if torch.isnan(loss):
                print("loss is nan !")

            bn += 1
            t_data += (t1 - ts)
            t_step += (t2 - t1)
            t_bwd += (t3 - t2)
            t_opt += (t4 - t3)
            t_tot += (t4 - ts)
            if getattr(args, 'debug', False) and bn >= bench_every:
                def ms(v): return 1000.0 * (v / max(1, bn))
                print(f"[bench_train] epoch={epoch} data={ms(t_data):.2f}ms step={ms(t_step):.2f}ms "
                      f"bwd={ms(t_bwd):.2f}ms opt={ms(t_opt):.2f}ms total={ms(t_tot):.2f}ms")
                if hasattr(model, '_step_bench') and isinstance(getattr(model, '_step_bench'), dict):
                    sb = model._step_bench
                    c = max(1, sb.get('calls', 1))

                    def msb(k): return 1000.0 * (sb.get(k, 0.0) / c)
                    print("[bench_step] epoch={} sas_fwd={:.2f}ms new_pos={:.2f}ms rep_pos={:.2f}ms "
                          "gate_pos={:.2f}ms neg_sample={:.2f}ms new_neg={:.2f}ms rep_neg={:.2f}ms "
                          "gate_neg={:.2f}ms bce={:.2f}ms total={:.2f}ms".format(
                              epoch, msb('sas_fwd'), msb('new_pos'), msb('rep_pos'), msb('gate_pos'),
                              msb('neg_sample'), msb('new_neg'), msb('rep_neg'), msb('gate_neg'),
                              msb('bce'), msb('total')
                          ))
                    try:
                        if hasattr(model, 'reset_step_bench'):
                            model.reset_step_bench()
                    except Exception:
                        pass
                bn = 0
                t_data = t_step = t_bwd = t_opt = t_tot = 0.0
        # Rising auxiliary task (SASRecRisingAux) epoch-level stats
        if hasattr(model, "_rising_dbg") and isinstance(model._rising_dbg, dict):
            rd = model._rising_dbg
            batches = max(1, int(rd.get("batches", 0)))
            click_avg = float(rd.get("click_sum", 0.0)) / batches
            aux_raw_avg = float(rd.get("aux_sum", 0.0)) / batches
            pos_rate = float(rd.get("pos_sum", 0.0)) / batches
            aux_w = float(getattr(model, "rising_aux_weight", 0.0))
            # effective ratio: (lambda * aux_loss) / click_loss
            eff_ratio = 0.0
            if click_avg > 0.0:
                eff_ratio = (aux_w * aux_raw_avg) / click_avg
            print(
                "[epoch_rising_aux] epoch={:03d} click_loss={:.6f} aux_loss_raw={:.6f} "
                "aux_weight={:.3f} aux_vs_click={:.3f} pos_rate={:.3f}".format(
                    epoch,
                    click_avg,
                    aux_raw_avg,
                    aux_w,
                    eff_ratio,
                    pos_rate,
                )
            )
            # reset for next epoch
            model._rising_dbg = None

        if getattr(args, 'debug', False) and hasattr(model, '_dbg'):
            dbg = model._dbg
            tokens_ratio = (dbg['tokens_sel_sum'] / dbg['tokens_all_sum']) if dbg['tokens_all_sum'] else 0.0
            print("[epoch_scope] epoch={:03d} model={} scope={} batches={} rep_batches={} "
                  "new_batches={} selected_batches={} tokens={}/{} ({:.2%})".format(
                      epoch,
                      dbg.get('model'),
                      getattr(args, 'train_scope', 'all') or 'all',
                      dbg.get('batches'),
                      dbg.get('rep_batches_sum'),
                      dbg.get('new_batches_sum'),
                      dbg.get('selected_batches_sum'),
                      dbg.get('tokens_sel_sum'),
                      dbg.get('tokens_all_sum'),
                      tokens_ratio,
                  ))
        if getattr(args, 'debug', False) and hasattr(model, '_gate_dbg') and isinstance(model._gate_dbg, dict):
            gd = model._gate_dbg

            def _avg(sum_key, cnt_key):
                c = gd.get(cnt_key, 0)
                s = gd.get(sum_key, 0.0)
                return (s / c) if c else 0.0
            fuse = getattr(model, 'fuse_mode', 'prob')
            print("[epoch_gate] epoch={:03d} calib={} fuse={} g_mean={:.4f} g_rep={:.4f} "
                  "g_new={:.4f} p_rep(rep)={:.4f} p_new(new)={:.4f} fused_rep={:.4f} "
                  "fused_new={:.4f} a_new={:.3f} b_new={:.3f} a_rep={:.3f} b_rep={:.3f}".format(
                      epoch,
                      gd.get('calib', 'affine'), fuse,
                      _avg('g_all_sum', 'g_all_cnt'),
                      _avg('g_rep_sum', 'g_rep_cnt'),
                      _avg('g_new_sum', 'g_new_cnt'),
                      _avg('prep_rep_sum', 'prep_rep_cnt'),
                      _avg('pnew_new_sum', 'pnew_new_cnt'),
                      _avg('fused_rep_sum', 'fused_rep_cnt'),
                      _avg('fused_new_sum', 'fused_new_cnt'),
                      model._gate_dbg.get('a_new', 0.0), model._gate_dbg.get('b_new', 0.0),
                      model._gate_dbg.get('a_rep', 0.0), model._gate_dbg.get('b_rep', 0.0),
                  ))
        if getattr(args, 'debug', False) and hasattr(model, '_state_dbg') and isinstance(model._state_dbg, dict):
            sd = model._state_dbg
            tok = max(1, int(sd.get('tokens', 0)))
            online = float(sd.get('online', 0)) / tok
            offline = float(sd.get('offline', 0)) / tok
            state_norm_avg = sd.get('state_norm_sum', 0.0) / tok
            item_norm_avg = sd.get('item_norm_sum', 0.0) / tok
            weight_norm = sd.get('weight_norm_last', 0.0)
            print("[epoch_state] epoch={:03d} tokens={} online={:.2%} offline={:.2%} item_norm={:.4f} "
                  "state_norm={:.4f} weight_norm={:.4f}".format(
                      epoch, tok, online, offline, item_norm_avg, state_norm_avg, weight_norm
                  ))
            model._state_dbg = None
        if getattr(args, 'debug', False) and hasattr(model, '_feat_dbg') and isinstance(model._feat_dbg, dict):
            fd = model._feat_dbg
            tok = max(1, int(fd.get('tokens', 0)))

            def _avg_feat(key):
                return (fd.get(key, 0.0) / tok) if tok > 0 else 0.0
            print("[epoch_feat] epoch={:03d} avg_norm count={:.4f} bucket={:.4f} cont={:.4f} "
                  "cyc={:.4f} tokens={}".format(
                      epoch,
                      _avg_feat('count_norm_sum'),
                      _avg_feat('bucket_norm_sum'),
                      _avg_feat('cont_norm_sum'),
                      _avg_feat('cyc_norm_sum'),
                      tok,
                  ))
            model._feat_dbg = None
        if getattr(args, 'debug', False) and hasattr(args, '_neg_dbg'):
            nd = args._neg_dbg
            tot = max(1, nd.get('total', 0))
            accept = nd.get('accept', 0)
            reject = nd.get('reject', 0)
            attempts_avg = nd.get('attempts_sum', 0) / tot
            print("[epoch_neg] epoch={:03d} scope={} chosen(rep/new)={}/{} fallback={} "
                  "attempts_avg={:.2f} accept/reject={}/{} time_ms={:.1f}".format(
                      epoch,
                      getattr(args, 'train_scope', 'all') or 'all',
                      nd.get('chosen_rep', 0), nd.get('chosen_new', 0),
                      nd.get('fallback', 0),
                      attempts_avg,
                      accept, reject,
                      nd.get('time_ms', 0.0),
                  ))
            if 'pos_rep' in nd and 'posrep_to_rep' in nd:

                pr = nd.get('pos_rep', 0)
                pn = nd.get('pos_new', 0)
                m1 = (nd.get('posrep_to_rep', 0) / pr) if pr > 0 else 0.0
                m2 = (nd.get('posnew_to_new', 0) / pn) if pn > 0 else 0.0
                print("[epoch_neg_map] epoch={:03d} pos_rep={}→rep={:.2%} pos_new={}→new={:.2%}".format(
                    epoch, pr, m1, pn, m2
                ))
            args._neg_dbg = None

        # Validation scheduling: run every N epochs
        val_every = getattr(args, 'val_every', 1)
        should_validate = (epoch % val_every == 0) or (epoch == args.num_epochs - 1)
        
        if should_validate:
            scores = compute_recall(model, val_loader, args, maxit=getattr(args, 'val_maxit', 500), phase='val')
            print('Epoch: {:03d}, Loss: {:.5f}'.format(epoch, loss_all / loss_cnt))
            print_scores(scores)
            if getattr(args, 'debug', False):
                print("[eval_scope] phase=val rep_count={} new_count={} all_count={} ratio={:.4f}".format(
                    scores.get('rep_count'), scores.get('new_count'), scores.get('all_count'), scores.get('ratio')
                ))

            # Use configurable early-stop metric; default is scores['all']['h01'].
            current_val = scores.get('all', {}).get(early_metric_key, None)
            if current_val is None:
                # Fallback to h01 if the requested metric is missing
                fallback_key = "h01"
                current_val = scores.get('all', {}).get(fallback_key, 0.0)
                if getattr(args, 'debug', False):
                    print(
                        f"[early_stop] Requested metric '{early_metric_key}' "
                        f"not found in scores['all']; falling back to '{fallback_key}'."
                    )

            early_stop_min_epoch = int(getattr(args, "early_stop_min_epoch", 0) or 0)
            if epoch < early_stop_min_epoch:
                if getattr(args, 'debug', False):
                    print(
                        f"[early_stop] Warmup active (epoch {epoch} < {early_stop_min_epoch}); "
                        "skip early-stop update."
                    )
                continue

            if current_val > best_val:
                best_val = current_val
                # Save checkpoint - checkpoint_path may have been modified by _ensure_unique_path
                actual_checkpoint_path = Path(checkpoint_path)
                torch.save(model.state_dict(), actual_checkpoint_path)
                # Update checkpoint_path to actual saved file path
                checkpoint_path = str(actual_checkpoint_path)
                best_cnt = best_max
            else:
                best_cnt -= 1
                if best_cnt == 0:
                    print(f"[early_stop] No improvement for {best_max} validation checks. Stopping at epoch {epoch}.")
                    break
        else:
            # Skip validation this epoch
            print('Epoch: {:03d}, Loss: {:.5f} [validation skipped, next at epoch {}]'.format(
                epoch, loss_all / loss_cnt, epoch + (val_every - epoch % val_every)
            ))
            # Save checkpoint periodically even without validation
            if epoch % max(1, val_every) == 0:
                actual_checkpoint_path = Path(checkpoint_path)
                torch.save(model.state_dict(), actual_checkpoint_path)
                checkpoint_path = str(actual_checkpoint_path)

    # Load the actual checkpoint that was saved (may have -1, -2 suffix)
    model = model_cls(args).to(args.device)
    actual_checkpoint_path = Path(checkpoint_path)
    # compute_recall / rank_dump에서도 어떤 체크포인트 기준인지 알 수 있도록 전달
    try:
        setattr(args, "checkpoint", str(actual_checkpoint_path))
    except Exception:
        pass
    try:
        state = torch.load(actual_checkpoint_path, map_location=args.device)
        model.load_state_dict(state)
    except Exception as e:
        print(f"[warn] strict load failed: {e}\n       -> retrying with strict=False for backward compatibility")
        state = torch.load(actual_checkpoint_path, map_location=args.device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        try:
            if missing or unexpected:
                print(f"[warn] missing keys: {missing}, unexpected: {unexpected}")
        except Exception:
            pass

    scores = compute_recall(model, test_loader, args, phase='test')
    print("Final score")
    print("=" * 11)
    print('Epoch: {:03d}, Loss: {:.5f}'.format(epoch, loss_all / loss_cnt))
    print_scores(scores)
    if getattr(args, 'debug', False):
        print("[eval_scope] phase=test rep_count={} new_count={} all_count={} ratio={:.4f}".format(
            scores.get('rep_count'), scores.get('new_count'), scores.get('all_count'), scores.get('ratio')
        ))
    # Save scores with actual checkpoint path (includes -1, -2 suffix if applicable)
    save_scores(scores, args, checkpoint_path=checkpoint_path)
    try:
        log_path = Path(getattr(args, "log_txt", DEFAULT_LOG_TXT))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path_obj = Path(checkpoint_path)
        with open(log_path, 'a') as flog:
            flog.write(f"{datetime.utcnow().isoformat()} final_checkpoint {checkpoint_path}\n")
            flog.write(f"{datetime.utcnow().isoformat()} final_checkpoint_filename {checkpoint_path_obj.name}\n")
            flog.write(f"{datetime.utcnow().isoformat()} final_checkpoint_path {checkpoint_path_obj.resolve()}\n")
    except Exception as log_exc:
        print(f"[warn] failed to append checkpoint log: {log_exc}")


if __name__ == "__main__":
    main()
 
