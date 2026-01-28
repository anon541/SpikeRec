#!/usr/bin/env python
"""Validate that computed viewer trends match the original data.

This script:
1. Loads the computed viewer trends.
2. Samples a few (streamer, timestamp) pairs.
3. Computes viewer counts directly from raw data for those pairs.
4. Compares the results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd


def validate_viewer_trends(
    viewer_trends_path: Path,
    raw_data_path: Path,
    sample_size: int = 100,
) -> None:
    """Validate viewer trends against raw data."""
    print(f"[validate] Loading viewer trends from: {viewer_trends_path}")
    
    # Load viewer trends
    df_trends = pd.read_parquet(viewer_trends_path)
    print(f"[validate] Loaded {len(df_trends):,} viewer trend records")
    
    # Sample some (streamer, timestamp) pairs for validation
    sample_df = df_trends.sample(min(sample_size, len(df_trends)), random_state=42)
    print(f"[validate] Sampling {len(sample_df):,} pairs for validation")
    
    # Connect to DuckDB
    con = duckdb.connect()
    
    # Load raw data
    print(f"[validate] Loading raw data from: {raw_data_path}")
    con.execute(f"""
        CREATE TABLE raw_input AS
        SELECT 
            column0 AS user,
            column1 AS stream,
            column2 AS streamer,
            column3 AS start,
            column4 AS stop
        FROM read_csv_auto('{raw_data_path}', header=false, parallel=true)
    """)
    
    raw_count = con.execute("SELECT COUNT(*) FROM raw_input").fetchone()[0]
    print(f"[validate] Loaded {raw_count:,} rows from raw data")
    
    # Validate each sampled pair
    print(f"\n[validate] Validating sampled pairs...")
    mismatches = []
    matches = 0
    
    for idx, row in sample_df.iterrows():
        streamer = row["streamer"]
        timestamp = row["timestamp"]
        expected_count = row["viewer_count"]
        
        # Compute actual viewer count from raw data
        actual_count = con.execute("""
            SELECT COUNT(DISTINCT user) AS viewer_count
            FROM raw_input
            WHERE streamer = ? AND start = ?
        """, [streamer, timestamp]).fetchone()[0]
        
        if actual_count != expected_count:
            mismatches.append({
                "streamer": streamer,
                "timestamp": timestamp,
                "expected": expected_count,
                "actual": actual_count,
            })
        else:
            matches += 1
        
        if (matches + len(mismatches)) % 20 == 0:
            print(f"[validate]   Processed {matches + len(mismatches)}/{len(sample_df)} pairs...")
    
    # Print results
    print(f"\n[validate] ========================================")
    print(f"[validate] Validation Results:")
    print(f"[validate]   Total sampled: {len(sample_df):,}")
    print(f"[validate]   Matches: {matches:,} ({100.0 * matches / len(sample_df):.1f}%)")
    print(f"[validate]   Mismatches: {len(mismatches):,} ({100.0 * len(mismatches) / len(sample_df):.1f}%)")
    
    if mismatches:
        print(f"\n[validate] Mismatches (first 10):")
        for i, mm in enumerate(mismatches[:10]):
            print(f"[validate]   {i+1}. streamer={mm['streamer']}, timestamp={mm['timestamp']}")
            print(f"[validate]      Expected: {mm['expected']}, Actual: {mm['actual']}")
    else:
        print(f"\n[validate] ✓ All sampled pairs match!")
    
    print(f"[validate] ========================================")
    
    return len(mismatches) == 0


def main():
    parser = argparse.ArgumentParser(description="Validate viewer trends against raw data")
    parser.add_argument(
        "--viewer_trends",
        type=str,
        required=True,
        help="Path to streamer_viewer_trends.parquet",
    )
    parser.add_argument(
        "--raw_data",
        type=str,
        required=True,
        help="Path to raw data (full.csv or sample)",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100,
        help="Number of pairs to sample for validation",
    )
    args = parser.parse_args()
    
    viewer_trends_path = Path(args.viewer_trends)
    if not viewer_trends_path.exists():
        raise FileNotFoundError(f"Viewer trends not found: {viewer_trends_path}")
    
    raw_data_path = Path(args.raw_data)
    if not raw_data_path.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_data_path}")
    
    is_valid = validate_viewer_trends(
        viewer_trends_path,
        raw_data_path,
        sample_size=args.sample_size,
    )
    
    if is_valid:
        print("\n[validate] ✓ Validation passed!")
    else:
        print("\n[validate] ✗ Validation failed - mismatches found!")


if __name__ == "__main__":
    main()

