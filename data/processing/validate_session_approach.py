#!/usr/bin/env python
"""Validate session-first sampling approach on a small sample dataset.

This script analyzes the feasibility of session-first sampling by:
1. Loading a sample dataset (full_resample_v2 or full_sample_500k).
2. Simulating session-first sampling.
3. Identifying inconsistencies (sessions not selected but included via user interactions).
4. Measuring the impact of preserving full user histories vs. session-only interactions.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import duckdb
import pandas as pd


def load_sample_data(path: Path, max_rows: int = 100_000) -> pd.DataFrame:
    """Load sample data from parquet or CSV."""
    if path.suffix == ".parquet":
        # Load only a sample for faster processing
        try:
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(path)
            # Read first row group or slice to max_rows
            table = parquet_file.read_row_groups([0])
            if table.num_rows > max_rows:
                table = table.slice(0, max_rows)
            df = table.to_pandas()
        except Exception:
            # Fallback: read full file and take head
            df = pd.read_parquet(path).head(max_rows)
        print(f"[validate] Loaded {len(df):,} rows from {path} (limited to {max_rows:,})")
    else:
        df = pd.read_csv(path, header=None, names=["user", "stream", "streamer", "start", "stop"], nrows=max_rows)
        print(f"[validate] Loaded {len(df):,} rows from {path} (limited to {max_rows:,})")
    return df


def analyze_session_first_approach(df: pd.DataFrame, sample_sessions: int = 100) -> dict:
    """Simulate session-first sampling and identify inconsistencies."""
    print(f"\n[validate] Analyzing session-first approach...")
    print(f"[validate] Total interactions: {len(df):,}")
    
    # Step 1: Identify unique sessions
    sessions = df.groupby(["streamer", "start", "stop"]).agg({
        "user": "count",
        "stream": "nunique"
    }).reset_index()
    sessions.columns = ["streamer", "start", "stop", "viewer_count", "unique_streams"]
    sessions = sessions.sort_values("viewer_count", ascending=False)
    
    print(f"[validate] Unique sessions: {len(sessions):,}")
    print(f"[validate] Top 10 sessions by viewer count:")
    print(sessions.head(10).to_string())
    
    # Step 2: Select top N sessions (simulating session-first selection)
    selected_sessions = sessions.head(sample_sessions).copy()
    selected_session_keys = set(
        (row["streamer"], row["start"], row["stop"])
        for _, row in selected_sessions.iterrows()
    )
    
    print(f"\n[validate] Selected {len(selected_sessions):,} sessions")
    print(f"[validate] Selected sessions viewer count range: {selected_sessions['viewer_count'].min()} - {selected_sessions['viewer_count'].max()}")
    
    # Step 3: Find users who participated in selected sessions
    selected_users_df = df[df.apply(
        lambda row: (row["streamer"], row["start"], row["stop"]) in selected_session_keys,
        axis=1
    )]
    selected_users = set(selected_users_df["user"].unique())
    
    print(f"[validate] Users in selected sessions: {len(selected_users):,}")
    
    # Step 4: Get ALL interactions for selected users (simulating full history preservation)
    all_user_interactions = df[df["user"].isin(selected_users)].copy()
    
    print(f"[validate] All interactions for selected users: {len(all_user_interactions):,}")
    
    # Step 5: Identify sessions NOT selected but included via user interactions
    all_sessions_in_data = set(
        (row["streamer"], row["start"], row["stop"])
        for _, row in all_user_interactions.iterrows()
    )
    unselected_sessions = all_sessions_in_data - selected_session_keys
    
    print(f"\n[validate] ===== INCONSISTENCY ANALYSIS =====")
    print(f"[validate] Selected sessions: {len(selected_session_keys):,}")
    print(f"[validate] Sessions in final data (via user interactions): {len(all_sessions_in_data):,}")
    print(f"[validate] Unselected sessions included: {len(unselected_sessions):,}")
    print(f"[validate] Inconsistency ratio: {len(unselected_sessions) / len(all_sessions_in_data) * 100:.1f}%")
    
    # Step 6: Analyze unselected sessions
    unselected_sessions_df = all_user_interactions[
        all_user_interactions.apply(
            lambda row: (row["streamer"], row["start"], row["stop"]) in unselected_sessions,
            axis=1
        )
    ]
    
    unselected_stats = unselected_sessions_df.groupby(["streamer", "start", "stop"]).agg({
        "user": "count"
    }).reset_index()
    unselected_stats.columns = ["streamer", "start", "stop", "viewer_count"]
    
    print(f"\n[validate] Unselected sessions statistics:")
    print(f"[validate]   Count: {len(unselected_stats):,}")
    print(f"[validate]   Viewer count (from sampled users):")
    print(f"[validate]     Min: {unselected_stats['viewer_count'].min()}")
    print(f"[validate]     Max: {unselected_stats['viewer_count'].max()}")
    print(f"[validate]     Mean: {unselected_stats['viewer_count'].mean():.1f}")
    print(f"[validate]     Median: {unselected_stats['viewer_count'].median():.1f}")
    
    # Step 7: Compare with original session statistics
    original_unselected = sessions[sessions.apply(
        lambda row: (row["streamer"], row["start"], row["stop"]) in unselected_sessions,
        axis=1
    )]
    
    if len(original_unselected) > 0:
        print(f"\n[validate] Original viewer counts for unselected sessions:")
        print(f"[validate]   Mean: {original_unselected['viewer_count'].mean():.1f}")
        print(f"[validate]   Median: {original_unselected['viewer_count'].median():.1f}")
        print(f"[validate]   Min: {original_unselected['viewer_count'].min()}")
        print(f"[validate]   Max: {original_unselected['viewer_count'].max()}")
        
        # Calculate discrepancy
        merged = unselected_stats.merge(
            original_unselected[["streamer", "start", "stop", "viewer_count"]],
            on=["streamer", "start", "stop"],
            suffixes=("_sampled", "_original")
        )
        merged["discrepancy"] = merged["viewer_count_sampled"] / merged["viewer_count_original"]
        
        print(f"\n[validate] Viewer count discrepancy (sampled / original):")
        print(f"[validate]   Mean: {merged['discrepancy'].mean():.3f}")
        print(f"[validate]   Median: {merged['discrepancy'].median():.3f}")
        print(f"[validate]   Min: {merged['discrepancy'].min():.3f}")
        print(f"[validate]   Max: {merged['discrepancy'].max():.3f}")
    
    return {
        "total_interactions": len(df),
        "total_sessions": len(sessions),
        "selected_sessions": len(selected_sessions),
        "selected_users": len(selected_users),
        "all_user_interactions": len(all_user_interactions),
        "unselected_sessions_included": len(unselected_sessions),
        "inconsistency_ratio": len(unselected_sessions) / len(all_sessions_in_data) if len(all_sessions_in_data) > 0 else 0,
        "unselected_session_stats": {
            "count": len(unselected_stats),
            "viewer_count_min": int(unselected_stats["viewer_count"].min()) if len(unselected_stats) > 0 else 0,
            "viewer_count_max": int(unselected_stats["viewer_count"].max()) if len(unselected_stats) > 0 else 0,
            "viewer_count_mean": float(unselected_stats["viewer_count"].mean()) if len(unselected_stats) > 0 else 0,
            "viewer_count_median": float(unselected_stats["viewer_count"].median()) if len(unselected_stats) > 0 else 0,
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Validate session-first sampling approach")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/processed/full_resample_v2/interactions.parquet",
        help="Path to sample dataset (parquet or CSV)",
    )
    parser.add_argument(
        "--sample_sessions",
        type=int,
        default=100,
        help="Number of sessions to select for simulation",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="docs/analysis/session_approach_validation.json",
        help="Output JSON file for results",
    )
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"[validate] Error: Dataset not found: {dataset_path}")
        return
    
    print(f"[validate] Loading dataset: {dataset_path}")
    df = load_sample_data(dataset_path, max_rows=100_000)
    
    results = analyze_session_first_approach(df, sample_sessions=args.sample_sessions)
    
    # Save results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[validate] Results saved to: {out_path}")
    print(f"\n[validate] ===== SUMMARY =====")
    print(f"[validate] Inconsistency ratio: {results['inconsistency_ratio']*100:.1f}%")
    print(f"[validate] Unselected sessions included: {results['unselected_sessions_included']:,}")
    print(f"[validate] This means {results['inconsistency_ratio']*100:.1f}% of sessions in final data")
    print(f"[validate] were not explicitly selected but included via user interactions.")


if __name__ == "__main__":
    main()

