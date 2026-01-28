#!/usr/bin/env python
"""Preprocess large interaction CSV using DuckDB.

Steps:
1. Load CSV into DuckDB table with duration column.
2. Deduplicate overlapping interactions (same user/start) by keeping the longest duration.
3. Drop users/streamers with fewer than --min_interactions entries.
4. Export cleaned data + metadata/debug samples.
"""

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Chunk-friendly preprocessing via DuckDB")
    parser.add_argument("--dataset", required=True, help="Input CSV path")
    parser.add_argument("--out_dir", required=True, help="Directory to store outputs")
    parser.add_argument("--min_interactions", type=int, default=5, help="Minimum interactions per user/streamer")
    parser.add_argument("--debug_samples", type=int, default=3, help="Number of users to dump for debugging")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for debug sampling")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "preprocess.duckdb"
    if db_path.exists():
        db_path.unlink()

    conn = duckdb.connect(str(db_path))
    print(f"[duckdb] loading {dataset_path}")
    conn.execute(
        """
        CREATE TABLE raw AS
        SELECT *, stop - start AS duration
        FROM read_csv_auto(?,
            columns={
                'user': 'VARCHAR',
                'stream': 'VARCHAR',
                'streamer': 'VARCHAR',
                'start': 'BIGINT',
                'stop': 'BIGINT'
            },
            header=false);
        """,
        [str(dataset_path)],
    )

    rows_raw = conn.execute("SELECT COUNT(*) FROM raw").fetchone()[0]
    print(f"[stats] raw rows={rows_raw}")

    conn.execute(
        """
        CREATE TABLE dedup AS
        SELECT user, stream, streamer, start, stop
        FROM (
            SELECT *,
                   ROW_NUMBER() OVER (PARTITION BY user, start ORDER BY duration DESC, streamer) AS rn
            FROM raw
        )
        WHERE rn = 1;
        """
    )
    rows_dedup = conn.execute("SELECT COUNT(*) FROM dedup").fetchone()[0]
    print(f"[stats] after dedupe rows={rows_dedup}")

    conn.execute(
        """
        CREATE TABLE user_keep AS
        SELECT user
        FROM (
            SELECT user, COUNT(*) AS cnt
            FROM dedup
            GROUP BY user
        )
        WHERE cnt >= ?;
        """,
        [args.min_interactions],
    )
    conn.execute(
        """
        CREATE TABLE streamer_keep AS
        SELECT streamer
        FROM (
            SELECT streamer, COUNT(*) AS cnt
            FROM dedup
            GROUP BY streamer
        )
        WHERE cnt >= ?;
        """,
        [args.min_interactions],
    )

    users_after = conn.execute("SELECT COUNT(*) FROM user_keep").fetchone()[0]
    streamers_after = conn.execute("SELECT COUNT(*) FROM streamer_keep").fetchone()[0]
    print(f"[filter] users_after={users_after}, streamers_after={streamers_after}")

    conn.execute(
        """
        CREATE TABLE filtered AS
        SELECT d.*
        FROM dedup d
        JOIN user_keep u ON d.user = u.user
        JOIN streamer_keep s ON d.streamer = s.streamer;
        """
    )
    rows_final = conn.execute("SELECT COUNT(*) FROM filtered").fetchone()[0]
    print(f"[stats] final rows={rows_final}")

    cleaned_parquet = out_dir / "cleaned.parquet"
    conn.execute("COPY filtered TO ? (FORMAT 'parquet')", [str(cleaned_parquet)])
    cleaned_csv = out_dir / "cleaned.csv"
    conn.execute("COPY filtered TO ? (FORMAT 'csv', HEADER false)", [str(cleaned_csv)])

    meta = {
        "dataset": str(dataset_path),
        "rows_raw": int(rows_raw),
        "rows_dedup": int(rows_dedup),
        "rows_final": int(rows_final),
        "users_before": int(conn.execute("SELECT COUNT(DISTINCT user) FROM raw").fetchone()[0]),
        "users_after": int(users_after),
        "streamers_before": int(conn.execute("SELECT COUNT(DISTINCT streamer) FROM raw").fetchone()[0]),
        "streamers_after": int(streamers_after),
        "min_interactions": args.min_interactions,
    }
    with (out_dir / "clean_meta.json").open("w") as fout:
        json.dump(meta, fout, indent=2)
    print(json.dumps(meta, indent=2))

    debug_log = out_dir / "debug_samples.txt"
    user_df = conn.execute("SELECT user FROM user_keep").fetchdf()
    if not user_df.empty:
        sample_df = user_df.sample(n=min(args.debug_samples, len(user_df)), random_state=args.seed)
        with debug_log.open("w") as fout:
            for user in sample_df["user"]:
                sample_rows = conn.execute(
                    "SELECT * FROM filtered WHERE user=? ORDER BY start LIMIT 5",
                    [user],
                ).fetchdf()
                fout.write(f"user={user}\n{sample_rows}\n\n")
        print(f"[debug] sample rows -> {debug_log}")
    else:
        print("[debug] no users to sample")

    conn.close()


if __name__ == "__main__":
    main()
