#!/usr/bin/env python
"""Validate resampled datasets by computing coverage and distribution stats.

This script is designed to run on the parquet/CSV outputs emitted by
``data/processing/full_resample.py`` without loading the full dataset into memory.
It emits a JSON report with row/user counts, live-slate and viewer-count summaries,
repeat-vs-new share, and viewer bucket coverage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import duckdb

CSV_SCHEMA = {
    "user": "VARCHAR",
    "stream": "VARCHAR",
    "streamer": "VARCHAR",
    "start": "BIGINT",
    "stop": "BIGINT",
    "live_streamers": "BIGINT",
    "viewer_count": "BIGINT",
    "viewer_bucket_id": "BIGINT",
    "viewer_bucket_label": "VARCHAR",
}

SLOTS_PER_DAY = 144  # 10-minute buckets


def _detect_dataset(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        parquet = path / "interactions.parquet"
        if parquet.exists():
            return parquet
        csv_path = path / "interactions.csv"
        if csv_path.exists():
            return csv_path
    raise FileNotFoundError(f"Could not find interactions.[parquet|csv] under {path}")


def _load_table(con: duckdb.DuckDBPyConnection, dataset_path: Path) -> None:
    if dataset_path.suffix.lower() == ".parquet":
        con.execute(f"CREATE OR REPLACE VIEW interactions AS SELECT * FROM read_parquet('{dataset_path}')")
        return
    if dataset_path.suffix.lower() == ".csv":
        column_spec = ", ".join(f"'{k}': '{v}'" for k, v in CSV_SCHEMA.items())
        con.execute(
            f"""
            CREATE OR REPLACE VIEW interactions AS
            SELECT * FROM read_csv_auto(
                ?, columns={{ {column_spec} }}, header=false, sample_size=-1
            )
            """,
            [str(dataset_path)],
        )
        return
    raise ValueError(f"Unsupported dataset format: {dataset_path}")


def _stats_row(con: duckdb.DuckDBPyConnection, sql: str) -> Tuple[Any, ...]:
    row = con.execute(sql).fetchone()
    return tuple(row) if row else tuple()


def _to_dict(keys: Tuple[str, ...], values: Tuple[Any, ...]) -> Dict[str, Any]:
    return {k: v for k, v in zip(keys, values)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate resampled dataset outputs")
    parser.add_argument("--dataset", required=True, help="Path to interactions.parquet or its parent directory")
    parser.add_argument("--out", required=True, help="Path to write JSON report")
    parser.add_argument("--threads", type=int, default=None, help="DuckDB threads (default: auto)")
    args = parser.parse_args()

    dataset_path = _detect_dataset(Path(args.dataset))
    con = duckdb.connect(database=":memory:")
    if args.threads:
        con.execute(f"PRAGMA threads={int(args.threads)}")

    _load_table(con, dataset_path)

    counts_keys = ("rows", "users", "streamers", "distinct_streams")
    counts_vals = _stats_row(
        con,
        """
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT "user") AS users,
            COUNT(DISTINCT streamer) AS streamers,
            COUNT(DISTINCT stream) AS distinct_streams
        FROM interactions
        """,
    )

    live_keys = (
        "avg",
        "p10",
        "p25",
        "p50",
        "p75",
        "p90",
        "p99",
        "min",
        "max",
    )
    live_vals = _stats_row(
        con,
        """
        SELECT
            AVG(live_streamers),
            QUANTILE(live_streamers, 0.1),
            QUANTILE(live_streamers, 0.25),
            QUANTILE(live_streamers, 0.5),
            QUANTILE(live_streamers, 0.75),
            QUANTILE(live_streamers, 0.9),
            QUANTILE(live_streamers, 0.99),
            MIN(live_streamers),
            MAX(live_streamers)
        FROM interactions
        """,
    )

    viewer_keys = (
        "avg",
        "p10",
        "p25",
        "p50",
        "p75",
        "p90",
        "p99",
        "max",
    )
    viewer_vals = _stats_row(
        con,
        """
        SELECT
            AVG(viewer_count),
            QUANTILE(viewer_count, 0.1),
            QUANTILE(viewer_count, 0.25),
            QUANTILE(viewer_count, 0.5),
            QUANTILE(viewer_count, 0.75),
            QUANTILE(viewer_count, 0.9),
            QUANTILE(viewer_count, 0.99),
            MAX(viewer_count)
        FROM interactions
        WHERE viewer_count IS NOT NULL
        """,
    )

    user_keys = ("p10", "p25", "p50", "p75", "p90", "p99", "max", "active_days_p50", "active_days_p90")
    user_vals = _stats_row(
        con,
        f"""
        WITH user_stats AS (
            SELECT
                "user" AS user_id,
                COUNT(*) AS interactions,
                COUNT(DISTINCT CAST(floor(start / {SLOTS_PER_DAY}) AS BIGINT)) AS active_days
            FROM interactions
            GROUP BY 1
        )
        SELECT
            QUANTILE(interactions, 0.1),
            QUANTILE(interactions, 0.25),
            QUANTILE(interactions, 0.5),
            QUANTILE(interactions, 0.75),
            QUANTILE(interactions, 0.9),
            QUANTILE(interactions, 0.99),
            MAX(interactions),
            QUANTILE(active_days, 0.5),
            QUANTILE(active_days, 0.9)
        FROM user_stats
        """,
    )

    repeat_vals = _stats_row(
        con,
        """
        WITH ranked AS (
            SELECT
                "user",
                streamer,
                start,
                ROW_NUMBER() OVER (PARTITION BY "user", streamer ORDER BY start) AS rnk
            FROM interactions
        )
        SELECT SUM(CASE WHEN rnk > 1 THEN 1 ELSE 0 END) AS repeats,
               COUNT(*) AS total
        FROM ranked
        """,
    )
    repeat_share = 0.0
    if repeat_vals and repeat_vals[1]:
        repeat_share = float(repeat_vals[0]) / float(repeat_vals[1])

    bucket_rows = con.execute(
        """
        SELECT
            COALESCE(CAST(viewer_bucket_id AS BIGINT), -1) AS bucket_id,
            COALESCE(viewer_bucket_label, 'NULL') AS bucket_label,
            COUNT(*) AS rows
        FROM interactions
        GROUP BY 1, 2
        ORDER BY bucket_id
        """
    ).fetchall()
    bucket_stats = [
        {"bucket_id": int(row[0]), "bucket_label": row[1], "rows": int(row[2])} for row in bucket_rows
    ]

    report: Dict[str, Any] = {
        "dataset_path": str(dataset_path),
        "counts": _to_dict(counts_keys, counts_vals),
        "live_streamers": _to_dict(live_keys, live_vals),
        "viewer_count": _to_dict(viewer_keys, viewer_vals),
        "user_stats": _to_dict(user_keys, user_vals),
        "repeat_share": repeat_share,
        "viewer_buckets": bucket_stats,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[validate] wrote report -> {out_path}")


if __name__ == "__main__":
    main()
