"""CLI for exploratory statistics on the full Twitch logs.

The raw file (~17 GB) is scanned via DuckDB so we avoid loading the
entire dataset into memory. The script computes per-user, per-timestep,
and per-streamer summary metrics and writes a JSON report for planning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import duckdb

CSV_COLUMNS = {
    "user": "VARCHAR",
    "stream": "VARCHAR",
    "streamer": "VARCHAR",
    "start": "BIGINT",
    "stop": "BIGINT",
}


def _default_threads() -> int:
    try:
        conn = duckdb.connect(database=":memory:")
        return max(1, int(conn.execute("PRAGMA threads").fetchone()[0]))
    except Exception:
        return 4


def _as_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return value


def _collect_columns(con: duckdb.DuckDBPyConnection) -> set[str]:
    rows = con.execute("DESCRIBE SELECT * FROM interactions").fetchall()
    return {row[0] for row in rows}


def _ensure_views(con: duckdb.DuckDBPyConnection, source: str) -> set[str]:
    con.execute(f"CREATE OR REPLACE VIEW interactions AS SELECT * FROM {source}")
    con.execute(
        """
        CREATE OR REPLACE VIEW timeline AS
        SELECT start AS bucket,
               COUNT(DISTINCT streamer) AS live_streamers
        FROM interactions
        GROUP BY 1
        """
    )
    return _collect_columns(con)


def _per_user_stats(con: duckdb.DuckDBPyConnection) -> Dict[str, Any]:
    con.execute(
        """
        CREATE OR REPLACE VIEW user_stats AS
        WITH base AS (
            SELECT
                user,
                COUNT(*) AS interactions,
                COUNT(DISTINCT CAST(floor(start / 144) AS BIGINT)) AS active_days,
                AVG(t.live_streamers) AS avg_live_slate
            FROM interactions i
            LEFT JOIN timeline t ON i.start = t.bucket
            GROUP BY user
        ),
        repeat_map AS (
            SELECT
                user,
                streamer,
                start,
                CASE WHEN ROW_NUMBER() OVER (PARTITION BY user, streamer ORDER BY start) > 1 THEN 1 ELSE 0 END AS is_repeat
            FROM interactions
        ),
        repeat_stats AS (
            SELECT user, AVG(is_repeat)::DOUBLE AS repeat_ratio
            FROM repeat_map
            GROUP BY user
        )
        SELECT
            b.user,
            interactions,
            active_days,
            avg_live_slate,
            COALESCE(r.repeat_ratio, 0) AS repeat_ratio
        FROM base b
        LEFT JOIN repeat_stats r USING(user)
        """
    )
    row = con.execute(
        """
        SELECT
            COUNT(*) AS num_users,
            AVG(interactions) AS avg_interactions,
            MIN(interactions) AS min_interactions,
            MAX(interactions) AS max_interactions,
            QUANTILE(interactions, 0.5) AS median_interactions,
            QUANTILE(interactions, 0.9) AS p90_interactions,
            QUANTILE(interactions, 0.99) AS p99_interactions,
            AVG(active_days) AS avg_active_days,
            QUANTILE(active_days, 0.5) AS median_active_days,
            AVG(avg_live_slate) AS avg_live_slate,
            QUANTILE(avg_live_slate, 0.5) AS median_live_slate,
            AVG(repeat_ratio) AS avg_repeat_ratio
        FROM user_stats
        """
    ).fetchone()
    columns = [
        "num_users",
        "avg_interactions",
        "min_interactions",
        "max_interactions",
        "median_interactions",
        "p90_interactions",
        "p99_interactions",
        "avg_active_days",
        "median_active_days",
        "avg_live_slate",
        "median_live_slate",
        "avg_repeat_ratio",
    ]
    summary = {col: _as_float(val) for col, val in zip(columns, row)}
    return {"summary": summary}


def _per_timestep_stats(con: duckdb.DuckDBPyConnection) -> Dict[str, Any]:
    row = con.execute(
        """
        SELECT
            COUNT(*) AS num_timesteps,
            AVG(live_streamers) AS avg_live_streamers,
            MIN(live_streamers) AS min_live_streamers,
            MAX(live_streamers) AS max_live_streamers,
            QUANTILE(live_streamers, 0.5) AS median_live_streamers,
            QUANTILE(live_streamers, 0.9) AS p90_live_streamers
        FROM timeline
        """
    ).fetchone()
    columns = [
        "num_timesteps",
        "avg_live_streamers",
        "min_live_streamers",
        "max_live_streamers",
        "median_live_streamers",
        "p90_live_streamers",
    ]
    summary = {col: _as_float(val) for col, val in zip(columns, row)}
    return {"summary": summary}


def _per_streamer_stats(con: duckdb.DuckDBPyConnection, has_viewer_count: bool) -> Dict[str, Any]:
    viewer_expr = "AVG(viewer_count)::DOUBLE" if has_viewer_count else "NULL"
    con.execute(
        f"""
        CREATE OR REPLACE VIEW streamer_stats AS
        SELECT
            streamer,
            COUNT(*) AS interactions,
            COUNT(DISTINCT user) AS unique_users,
            {viewer_expr} AS avg_viewer_count
        FROM interactions
        GROUP BY streamer
        """
    )
    row = con.execute(
        """
        SELECT
            COUNT(*) AS num_streamers,
            AVG(interactions) AS avg_interactions,
            QUANTILE(interactions, 0.5) AS median_interactions,
            AVG(unique_users) AS avg_unique_users,
            QUANTILE(unique_users, 0.5) AS median_unique_users,
            AVG(avg_viewer_count) AS avg_viewer_count
        FROM streamer_stats
        """
    ).fetchone()
    columns = [
        "num_streamers",
        "avg_interactions",
        "median_interactions",
        "avg_unique_users",
        "median_unique_users",
        "avg_viewer_count",
    ]
    summary = {col: _as_float(val) for col, val in zip(columns, row)}
    return {"summary": summary}


def run_stats(args: argparse.Namespace) -> Dict[str, Any]:
    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(path)
    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA threads={args.threads}")
    if path.suffix.lower() == ".parquet":
        source = f"read_parquet('{path.as_posix()}')"
    else:
        column_spec = ", ".join(f"'{k}': '{v}'" for k, v in CSV_COLUMNS.items())
        source = (
            "read_csv_auto("
            f"'{path.as_posix()}', "
            f"columns={{ {column_spec} }}, "
            "header=false, "
            "sample_size=-1"
            ")"
        )
    if args.limit:
        source = f"(SELECT * FROM {source} LIMIT {int(args.limit)})"
    columns = _ensure_views(con, source)
    result: Dict[str, Any] = {
        "input": str(path),
        "threads": args.threads,
    }
    if args.limit:
        result["limit"] = int(args.limit)
    total_rows = con.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
    result["summary"] = {"total_interactions": int(total_rows)}
    result["per_user"] = _per_user_stats(con)
    result["per_timestep"] = _per_timestep_stats(con)
    result["per_streamer"] = _per_streamer_stats(con, has_viewer_count="viewer_count" in columns)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exploratory stats for Twitch full logs")
    parser.add_argument("--input", required=True, help="Path to full.csv (CSV or Parquet)")
    parser.add_argument("--out", required=True, help="Path to write JSON stats")
    parser.add_argument("--threads", type=int, default=_default_threads(), help="DuckDB threads to use")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for faster exploratory runs (use full data when omitted)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = run_stats(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[resample_stats] wrote {out_path}")


if __name__ == "__main__":
    main()
