import argparse
from prettytable import PrettyTable


def print_args(parse_args):
    table = PrettyTable()
    table.field_names = ["Arg.", "Value"]
    for name, value in vars(parse_args).items():
        table.add_row([name, value])
    print(table)


def build_parser():
    parser = argparse.ArgumentParser(description="LiveRec - Twitch")

    # general training / model options
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--config", type=str, help="YAML/JSON config file with CLI overrides")
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true",
                        help="Validate config/imports without training")
    parser.add_argument("--dry-run-fast", "--dry_run_fast", dest="dry_run_fast", action="store_true",
                        help="Fast dry-run: skip sequence generation, use cache if available")
    parser.add_argument("--val-maxit", "--val_maxit", dest="val_maxit", type=int, default=None,
                        help="Max batches for validation evaluation (default: min(100, len(val_loader)))")
    parser.add_argument(
        "--eval-maxit",
        "--eval_maxit",
        dest="eval_maxit",
        type=int,
        default=None,
        help="Max batches for evaluation in scripts/eval.py (test/val); None = no cap",
    )
    parser.add_argument("--val-only", "--val_only", dest="val_only", action="store_true",
                        help="Load and test validation set only (skip train/test for fast validation check)")
    parser.add_argument("--val-every", "--val_every", dest="val_every", type=int, default=1,
                        help="Run validation every N epochs (default: 1 = every epoch)")
    parser.add_argument("--val-sample-candidates", "--val_sample_candidates", dest="val_sample_candidates", type=int, default=0,
                        help="Sample N candidates during validation (0 = use all, >0 = random sample for speed)")
    parser.add_argument("--dataset", help="Input dataset path")
    parser.add_argument("--viewer_trends_path", type=str, help="Path to streamer_viewer_trends.parquet for original viewer trends")
    parser.add_argument("--model", type=str, help="Model to train (LiveRec, SASRec, etc.)")
    parser.add_argument("--model_to", dest="mto", type=str, help="Filename (without .pt) for saving checkpoints")
    parser.add_argument("--model_path", type=str, help="Directory to save checkpoints")
    parser.add_argument("--cache_dir", type=str, help="Directory for cached preprocessed data")
    parser.add_argument("--caching", action="store_true", help="Enable cached dataloaders when available")
    parser.add_argument("--device", type=str, help="Torch device (e.g., cpu, cuda)")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging/debug helpers")
    parser.add_argument("--debug_batch_log_prob", type=float, help="Probability to dump per-batch debug JSON (0 disables)")
    parser.add_argument("--debug_nrows", type=int, help="Limit rows loaded from dataset when debugging")
    parser.add_argument("--debug_max_users", type=int, help="Limit users loaded from dataset when debugging")
    parser.add_argument("--checkpoint", type=str, help="Path to pretrained checkpoint for evaluation-only runs")
    parser.add_argument("--log_txt", type=str, help="Human-readable log output path")
    parser.add_argument("--log_jsonl", type=str, help="JSONL log output path")

    # evaluation helpers
    parser.add_argument("--rank_dump_csv", type=str, default="", help="Per-example rank diagnostics CSV (empty to disable)")
    parser.add_argument("--rank_dump_phase", type=str, choices=["train", "val", "test", "any"], default="test",
                        help="Which evaluation phase to dump ranks for")
    parser.add_argument("--rank_dump_miss_topk", type=int, default=0,
                        help="Only log rows with overall rank >= this value (0 logs everything)")
    parser.add_argument("--rank_dump_include_seq", action="store_true",
                        help="Include serialized history sequences in rank dump records")
    parser.add_argument("--rank_dump_include_hits", action="store_true",
                        help="Record hit cases even when rank_dump_miss_topk filters them out")
    parser.add_argument("--eval_split", type=str, choices=["test", "val"], default="test",
                        help="Evaluation split for analysis/eval-only scripts")
    parser.add_argument("--eval_slice", type=str, default=None,
                        choices=["spike_high", "spike_low"],
                        help="Evaluate on a slice of data (e.g., high/low spike scores)")

    # optimization hyper-parameters
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--l2", type=float, help="L2 regularization strength")
    parser.add_argument("--batch_size", type=int, help="Mini-batch size")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--early_stop", type=int, help="Patience (epochs) for early stopping on validation recall")
    parser.add_argument(
        "--early_stop_metric",
        type=str,
        help=(
            "Metric name in scores['all'] used for early stopping "
            "(e.g., h01, h05, h10, ndcg01, ndcg10). "
            "Defaults to h01 for backward compatibility."
        ),
    )
    parser.add_argument(
        "--early_stop_min_epoch",
        type=int,
        help="Minimum epoch before early stopping is allowed (warmup window).",
    )
    parser.add_argument("--train_last_only", action="store_true", help="Train loss only on final step of each sequence")

    # architecture dimensions
    parser.add_argument("--dim", dest="K", type=int, help="Embedding dimension")
    parser.add_argument("--seq_len", type=int, help="Max sequence length to consider")
    parser.add_argument("--num_att", type=int, help="Number of attention blocks for sequence encoder")
    parser.add_argument("--num_att_ctx", type=int, help="Number of attention blocks for availability context encoder")
    parser.add_argument("--num_heads", type=int, help="Multi-head attention heads for sequence encoder")
    parser.add_argument("--num_heads_ctx", type=int, help="Multi-head attention heads for availability encoder")
    parser.add_argument("--topk_att", type=int, help="Number of availability candidates to keep after attention")
    parser.add_argument("--viewer_feat_mode", type=str, choices=["off", "bucket", "spike"], default="off",
                        help="Viewer feature integration mode (default: off)")
    parser.add_argument("--caser_n_v", type=int, help="Number of vertical conv filters (Caser)")
    parser.add_argument("--caser_n_h", type=int, help="Number of horizontal conv filters per kernel (Caser)")
    parser.add_argument("--caser_horiz_kernels", type=str, help="Comma-separated horizontal kernel sizes (Caser)")
    parser.add_argument("--bert_mask_prob", type=float, help="Mask probability for BERT4Rec")

    # Rising auxiliary task (SASRecRisingAux) options
    parser.add_argument(
        "--rising_aux_weight",
        type=float,
        help="Weight λ for rising auxiliary loss (0 disables aux loss)",
    )
    parser.add_argument(
        "--rising_z_threshold",
        type=float,
        help="Z-score threshold for defining rising=1 (viewer_z >= threshold)",
    )
    parser.add_argument(
        "--rising_ratio_threshold",
        type=float,
        help="Viewer ratio threshold for rising=1 (viewer_ratio >= threshold)",
    )

    # modelling options specific to LiveRec variants
    parser.add_argument("--fr_ctx", action="store_true", help="Enable availability context attention (LiveRec)")
    parser.add_argument("--fr_rep", action="store_true", help="Enable repeat interval embeddings (LiveRec)")

    # training scope / negatives
    parser.add_argument("--train_scope", type=str, choices=["all", "rep", "new"], help="Subset of interactions to train on")
    parser.add_argument(
        "--neg_mode",
        type=str,
        choices=["auto", "rep", "new", "all", "mixture", "posgroup", "online"],
        help="Negative sampling mode",
    )
    parser.add_argument("--neg_mix_p", type=float, help="P(repeat-style negative) when neg_mode=mixture")
    parser.add_argument("--num_negs", type=int, help="Number of negatives sampled per positive")

    parser.set_defaults(
        seed=42,
        config="",
        dry_run=False,
        dataset="dataset",
        lr=0.0005,
        l2=0.1,
        batch_size=100,
        num_epochs=150,
        early_stop=15,
        early_stop_metric="h01",
        early_stop_min_epoch=0,
        K=64,
        seq_len=16,
        num_att=2,
        num_att_ctx=2,
        num_heads=4,
        num_heads_ctx=4,
        topk_att=64,
        model="LiveRec",
        mto="liverec",
        model_path="checkpoints",
        cache_dir="cache/",
        device="cuda",
        train_scope="all",
        neg_mode="online",
        neg_mix_p=0.5,
        num_negs=1,
        viewer_feat_mode="off",
        rising_aux_weight=0.0,
        rising_z_threshold=1.5,
        rising_ratio_threshold=1.2,
        debug_batch_log_prob=0.0,
        debug_nrows=0,
        debug_max_users=0,
        checkpoint="",
        log_txt="",
        log_jsonl="",
        val_maxit=None,
        eval_maxit=None,
        dry_run_fast=False,
        val_only=False,
        val_every=1,
        val_sample_candidates=0,
        caser_n_v=None,
        caser_n_h=None,
        caser_horiz_kernels=None,
        bert_mask_prob=0.2,
    )
    return parser


def arg_parse():
    parser = build_parser()
    return parser.parse_args()
