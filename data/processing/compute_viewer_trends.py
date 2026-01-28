#!/usr/bin/env python
"""Compute streamer viewer trends from raw data for a given dataset.

This script:
1. Loads a dataset (e.g., 100k.csv or full_resample_v2) to get the list of streamers.
2. Computes viewer trends for those streamers from raw data (full.csv).
3. Stores the trends as streamer_viewer_trends.parquet.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb
import pandas as pd


def get_streamers_from_dataset(dataset_path: Path) -> set[str]:
    """Extract unique streamers from a dataset."""
    print(f"[viewer_trends] Loading streamers from: {dataset_path}")
    
    if dataset_path.suffix == ".parquet":
        # Read parquet file
        try:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(dataset_path)
            # Read all row groups to get all streamers
            table = pf.read()
            df = table.to_pandas()
        except Exception:
            df = pd.read_parquet(dataset_path)
    else:
        # Read CSV file
        df = pd.read_csv(
            dataset_path,
            header=None,
            names=["user", "stream", "streamer", "start", "stop"],
        )
    
    streamers = set(df["streamer"].unique())
    print(f"[viewer_trends] Found {len(streamers):,} unique streamers")
    return streamers


def compute_viewer_trends(
    raw_data_path: Path,
    streamers: set[str],
    output_path: Path,
    threads: int = 8,
) -> None:
    """Compute viewer trends for given streamers from raw data."""
    print(f"[viewer_trends] Computing viewer trends from: {raw_data_path}")
    print(f"[viewer_trends] Streamers to process: {len(streamers):,}")
    
    start_time = time.time()
    
    # Connect to DuckDB
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")
    try:
        con.execute("SET memory_limit='32GB'")
    except Exception:
        pass
    
    # Load raw data
    print(f"[viewer_trends] Loading raw data...")
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
    print(f"[viewer_trends] Loaded {raw_count:,} rows from raw data")
    
    # Filter to only streamers we care about
    streamer_list = "', '".join(sorted(streamers))
    print(f"[viewer_trends] Filtering to {len(streamers):,} streamers...")
    
    con.execute(f"""
        CREATE TABLE filtered_raw AS
        SELECT *
        FROM raw_input
        WHERE streamer IN ('{streamer_list}')
    """)
    
    filtered_count = con.execute("SELECT COUNT(*) FROM filtered_raw").fetchone()[0]
    print(f"[viewer_trends] Filtered to {filtered_count:,} rows ({100.0 * filtered_count / max(raw_count, 1):.1f}%)")
    
    # Compute viewer trends: for each (streamer, timestamp), count unique users
    print(f"[viewer_trends] Computing viewer trends...")
    con.execute("""
        CREATE TABLE viewer_trends AS
        SELECT
            streamer,
            start AS timestamp,
            COUNT(DISTINCT user) AS viewer_count
        FROM filtered_raw
        GROUP BY streamer, start
        ORDER BY streamer, timestamp
    """)
    
    trend_count = con.execute("SELECT COUNT(*) FROM viewer_trends").fetchone()[0]
    print(f"[viewer_trends] Computed {trend_count:,} (streamer, timestamp) pairs")
    
    # Write to parquet
    print(f"[viewer_trends] Writing to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY viewer_trends TO ? (FORMAT 'parquet')", [str(output_path)])
    
    elapsed = time.time() - start_time
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    
    print(f"[viewer_trends] ✓ Completed in {elapsed:.1f}s")
    print(f"[viewer_trends] Output size: {file_size_mb:.1f} MB")
    
    # Print some statistics
    stats = con.execute("""
        SELECT
            COUNT(DISTINCT streamer) AS num_streamers,
            COUNT(DISTINCT timestamp) AS num_timestamps,
            AVG(viewer_count) AS avg_viewers,
            QUANTILE(viewer_count, 0.5) AS median_viewers,
            QUANTILE(viewer_count, 0.9) AS p90_viewers,
            MAX(viewer_count) AS max_viewers
        FROM viewer_trends
    """).fetchone()
    
    print(f"[viewer_trends] Statistics:")
    print(f"[viewer_trends]   Streamers: {stats[0]:,}")
    print(f"[viewer_trends]   Timestamps: {stats[1]:,}")
    print(f"[viewer_trends]   Avg viewers: {stats[2]:.1f}")
    print(f"[viewer_trends]   Median viewers: {stats[3]:.1f}")
    print(f"[viewer_trends]   P90 viewers: {stats[4]:.1f}")
    print(f"[viewer_trends]   Max viewers: {stats[5]:,}")


def main():
    parser = argparse.ArgumentParser(description="Compute streamer viewer trends from raw data")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset (100k.csv or interactions.parquet) to extract streamers from",
    )
    parser.add_argument(
        "--raw_data",
        type=str,
        default="data/raw/full.csv",
        help="Path to raw data (full.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for streamer_viewer_trends.parquet",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Number of threads for DuckDB",
    )
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    raw_data_path = Path(args.raw_data)
    if not raw_data_path.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_data_path}")
    
    output_path = Path(args.output)
    
    # Step 1: Extract streamers from dataset
    streamers = get_streamers_from_dataset(dataset_path)
    
    # Step 2: Compute viewer trends from raw data
    compute_viewer_trends(raw_data_path, streamers, output_path, threads=args.threads)
    
    print(f"\n[viewer_trends] ========================================")
    print(f"[viewer_trends] Viewer trends saved to: {output_path}")
    print(f"[viewer_trends] ========================================")


if __name__ == "__main__":
    main()

