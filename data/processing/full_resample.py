#!/usr/bin/env python
"""Resample full Twitch logs into availability-aware datasets.

This CLI wraps a DuckDB pipeline that:
1. Loads the raw CSV (optionally limited for debugging).
2. Filters interactions by live-channel concurrency.
3. Keeps only engaged users (min interactions + active days) and optionally
   caps the number of users or total interactions.
4. Annotates each (streamer, start) pair with concurrent viewer counts and
   assigns viewer-count buckets.
5. Writes the resulting dataset + stats + viewer bucket metadata.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import duckdb

CSV_COLUMNS = {
    "user": "VARCHAR",
    "stream": "VARCHAR",
    "streamer": "VARCHAR",
    "start": "BIGINT",
    "stop": "BIGINT",
}

SLOTS_PER_DAY = 144  # 10-minute buckets

# Debug defaults tuned on the 500k-row smoke-test slice only. Update these via
# a config file once full-log stats (e.g., docs/analysis/full_stats_full.json)
# are available for data/raw/full.csv.
DEBUG_DEFAULTS = {
    "min_live": 25,
    "max_live": 150,
    "min_interactions": 20,
    "min_active_days": 7,
    "max_users": 120_000,
    "target_interactions": 3_000_000,
    "viewer_bucket_quantiles": "0.2,0.4,0.6,0.8",
    "viewer_count_mode": "unique",
    "viewer_bucket_edges": [],
    "user_bins": [],
}


def _default_threads() -> int:
    try:
        con = duckdb.connect(database=":memory:")
        return max(1, int(con.execute("PRAGMA threads").fetchone()[0]))
    except Exception:
        return 4


def _coerce_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected integer threshold, got {value!r}") from exc


def _canonicalize_quantiles(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def _coerce_bucket_edges(value: Any) -> List[int]:
    if value in (None, "", []):
        return []
    if isinstance(value, str):
        tokens = [tok.strip() for tok in value.split(",")]
    elif isinstance(value, (list, tuple)):
        tokens = list(value)
    else:
        tokens = [value]
    edges: List[int] = []
    for token in tokens:
        if token in ("", None):
            continue
        try:
            num = int(token)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid viewer bucket edge {token!r}") from exc
        if not edges or num > edges[-1]:
            edges.append(num)
    return edges


def _load_config_file(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path).expanduser()
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    suffix = cfg_path.suffix.lower()
    text = cfg_path.read_text(encoding="utf-8")
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("PyYAML is required to read YAML configs") from exc
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text or "{}")
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {cfg_path}")
    return data


def _resolve_thresholds(args: argparse.Namespace) -> Dict[str, Any]:
    config_values = _load_config_file(getattr(args, "config", None))
    resolved: Dict[str, Any] = {**DEBUG_DEFAULTS}
    resolved.update(config_values)
    override_fields = [
        "min_live",
        "max_live",
        "min_interactions",
        "min_active_days",
        "max_users",
        "target_interactions",
        "viewer_bucket_quantiles",
        "viewer_bucket_edges",
        "viewer_count_mode",
        "user_bins",
    ]
    for field in override_fields:
        cli_value = getattr(args, field, None)
        if cli_value is not None:
            resolved[field] = cli_value
    resolved["viewer_bucket_quantiles"] = _canonicalize_quantiles(
        resolved.get("viewer_bucket_quantiles")
    )
    resolved["viewer_bucket_edges"] = _coerce_bucket_edges(resolved.get("viewer_bucket_edges"))
    resolved["viewer_count_mode"] = (resolved.get("viewer_count_mode") or "unique").lower()
    required_int_keys = ["min_live", "max_live", "min_interactions", "min_active_days"]
    for key in required_int_keys:
        resolved[key] = int(resolved[key])
    optional_int_keys = ["max_users", "target_interactions"]
    for key in optional_int_keys:
        resolved[key] = _coerce_optional_int(resolved.get(key))
    # viewer_count_mode not coerced here; validated later
    return resolved


def _using_debug_defaults(resolved: Dict[str, Any]) -> bool:
    for key, default in DEBUG_DEFAULTS.items():
        if key not in resolved:
            return False
        if str(resolved[key]) != str(default):
            return False
    return True


def _warn_if_debug_defaults(input_path: Path, resolved: Dict[str, Any]) -> None:
    if input_path.name == "full.csv" and _using_debug_defaults(resolved):
        print(
            "[resample][warn] Running on data/raw/full.csv with debug thresholds tuned on the "
            "500k sample. Provide --config (JSON/YAML) derived from full-log stats before "
            "shipping dataset artifacts."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-log resampling pipeline")
    parser.add_argument("--input", default="data/raw/full.csv", help="Raw interaction CSV (headerless)")
    parser.add_argument("--out_dir", required=True, help="Directory to write dataset + stats")
    parser.add_argument("--threads", type=int, default=_default_threads(), help="DuckDB threads")
    parser.add_argument("--config", default=None, help="JSON/YAML config with sampling thresholds")
    parser.add_argument(
        "--min_live", type=int, default=None, help="Minimum live channels per timestep to keep (debug default 25)"
    )
    parser.add_argument(
        "--max_live", type=int, default=None, help="Maximum live channels per timestep to keep (debug default 150)"
    )
    parser.add_argument(
        "--min_interactions", type=int, default=None, help="Minimum interactions per user (debug default 20)"
    )
    parser.add_argument(
        "--min_active_days", type=int, default=None, help="Minimum active day buckets per user (debug default 7)"
    )
    parser.add_argument(
        "--max_users",
        type=int,
        default=None,
        help="Upper bound on kept users (0 = no cap, debug default 120000)",
    )
    parser.add_argument(
        "--target_interactions",
        type=int,
        default=None,
        help="Soft cap on cumulative interactions (0 disables, debug default 3000000)",
    )
    parser.add_argument(
        "--viewer_bucket_quantiles",
        default=None,
        help="Comma-separated quantiles for viewer-count buckets (debug default 0.2,0.4,0.6,0.8)",
    )
    parser.add_argument(
        "--viewer_bucket_edges",
        default=None,
        help="Comma-separated edges for fixed viewer-count buckets (overrides quantiles)",
    )
    parser.add_argument(
        "--viewer_count_mode",
        default=None,
        choices=["unique", "total"],
        help="How to count concurrent viewers (unique users vs. total interactions)",
    )
    parser.add_argument(
        "--user_bins",
        default=None,
        help=(
            "JSON list defining stratified user bins, e.g. "
            '[{"min_interactions":10,"max_interactions":20,"max_users":15000}, ...]'
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for stratified user sampling (when enabled)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for debugging runs")
    parser.add_argument(
        "--keep_db",
        action="store_true",
        help="Reuse an existing DuckDB file in out_dir instead of recreating it",
    )
    return parser.parse_args()


def _build_csv_source(path: Path, limit: int | None) -> str:
    column_spec = ", ".join(f"'{k}': '{v}'" for k, v in CSV_COLUMNS.items())
    # 최적화: parallel 옵션으로 병렬 CSV 읽기 활성화
    source = (
        "read_csv_auto("
        f"'{path.as_posix()}', "
        f"columns={{ {column_spec} }}, "
        "header=false, "
        "sample_size=-1, "
        "parallel=true"
        ")"
    )
    if limit:
        source = f"(SELECT * FROM {source} LIMIT {int(limit)})"
    return source


def _connect(db_path: Path, threads: int, keep_db: bool) -> duckdb.DuckDBPyConnection:
    if db_path.exists() and not keep_db:
        db_path.unlink()
    con = duckdb.connect(database=str(db_path))
    con.execute(f"PRAGMA threads={threads}")
    # 메모리 및 성능 최적화 설정 (대용량 파일 처리)
    try:
        # DuckDB 메모리 제한 설정 (시스템 메모리의 80% 사용, 최대 32GB)
        con.execute("SET memory_limit='32GB'")
    except Exception:
        pass  # DuckDB 버전에 따라 지원하지 않을 수 있음
    # 성능 최적화 설정
    try:
        # 버퍼 풀 크기 증가 (기본값보다 크게)
        con.execute("SET temp_directory='/tmp'")
    except Exception:
        pass
    try:
        # 통계 자동 수집 비활성화 (대용량 데이터에서는 느림)
        con.execute("PRAGMA enable_progress_bar=false")
    except Exception:
        pass
    try:
        # 최적화 레벨 설정
        con.execute("PRAGMA default_order='ASC'")
    except Exception:
        pass
    return con


def _parse_quantiles(raw: str) -> List[float]:
    values: List[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            val = float(token)
        except ValueError:
            continue
        if 0.0 < val < 1.0:
            values.append(val)
    values = sorted(set(values))
    return values


def _compute_bucket_edges(
    con: duckdb.DuckDBPyConnection, quantiles: Sequence[float]
) -> List[int]:
    if not quantiles:
        return []
    select_expr = ", ".join(f"QUANTILE(concurrent_viewers, {q}) AS q{idx}" for idx, q in enumerate(quantiles))
    row = con.execute(f"SELECT {select_expr} FROM viewer_counts").fetchone()
    if not row:
        return []
    edges: List[int] = []
    for val in row:
        if val is None:
            continue
        candidate = max(0, int(round(float(val))))
        if not edges or candidate > edges[-1]:
            edges.append(candidate)
    return edges


def _bucket_case_expr(edges: Sequence[int], field: str) -> tuple[str, str]:
    if not edges:
        return "NULL", "NULL"
    id_parts = ["CASE", f"WHEN {field} IS NULL THEN NULL"]
    label_parts = ["CASE", f"WHEN {field} IS NULL THEN NULL"]
    lower = 0
    for idx, edge in enumerate(edges):
        condition = f"WHEN {field} < {edge}"
        id_parts.append(f"{condition} THEN {idx}")
        label_parts.append(f"{condition} THEN '[{lower},{edge})'")
        lower = edge
    id_parts.append(f"ELSE {len(edges)} END")
    label_parts.append(f"ELSE '[{lower},+inf)' END")
    return " ".join(id_parts), " ".join(label_parts)


def _parse_user_bins(raw_bins: Any) -> List[Dict[str, Any]]:
    """Normalize user bin configs: [{'min':10,'max':50,'max_users':20000}, ...]."""
    if not raw_bins:
        return []
    if isinstance(raw_bins, str):
        try:
            raw_bins = json.loads(raw_bins)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid user_bins JSON: {raw_bins}") from exc
    bins: List[Dict[str, Any]] = []
    for entry in raw_bins:
        if not isinstance(entry, dict):
            continue
        min_int = int(entry.get("min_interactions", 0))
        max_int_val = entry.get("max_interactions")
        max_int = int(max_int_val) if max_int_val not in (None, "") else None
        max_users_val = entry.get("max_users")
        max_users = int(max_users_val) if max_users_val not in (None, "", 0) else None
        bins.append(
            {
                "min_interactions": min_int,
                "max_interactions": max_int,
                "max_users": max_users,
            }
        )
    return bins


def _summarize_row(row: Iterable) -> list[float | int | None]:
    if row is None:
        return []
    out = []
    for val in row:
        if val is None:
            out.append(None)
        elif isinstance(val, (int, float)):
            out.append(val if isinstance(val, int) else float(val))
        else:
            try:
                out.append(float(val))
            except Exception:
                out.append(val)
    return out


def _format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{seconds:.1f}초"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}분 {secs:.1f}초"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}시간 {mins}분 {secs:.1f}초"


def _estimate_file_size_mb(path: Path) -> float:
    """Estimate file size in MB."""
    try:
        return path.stat().st_size / (1024 * 1024)
    except Exception:
        return 0.0


def _print_step(step_name: str, start_time: float, prev_time: float | None = None) -> float:
    """Print step progress with elapsed time and estimated remaining time."""
    elapsed = time.time() - start_time
    elapsed_str = _format_time(elapsed)
    
    if prev_time is not None:
        step_elapsed = elapsed - prev_time
        step_str = _format_time(step_elapsed)
        print(f"[resample] ✓ {step_name} 완료 ({step_str} 소요, 누적: {elapsed_str})")
    else:
        print(f"[resample] → {step_name} 시작...")
    
    return elapsed


def main() -> None:
    overall_start = time.time()
    args = parse_args()
    thresholds = _resolve_thresholds(args)
    args.min_live = thresholds["min_live"]
    args.max_live = thresholds["max_live"]
    args.min_interactions = thresholds["min_interactions"]
    args.min_active_days = thresholds["min_active_days"]
    args.max_users = thresholds["max_users"]
    args.target_interactions = thresholds["target_interactions"]
    args.viewer_bucket_quantiles = thresholds["viewer_bucket_quantiles"]
    args.viewer_bucket_edges = thresholds.get("viewer_bucket_edges")
    args.viewer_count_mode = thresholds.get("viewer_count_mode", "unique")
    args.user_bins = _parse_user_bins(thresholds.get("user_bins"))

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    _warn_if_debug_defaults(input_path, thresholds)
    
    # 파일 크기 확인 및 예상 시간 출력
    file_size_mb = _estimate_file_size_mb(input_path)
    if file_size_mb > 0:
        print(f"[resample] 입력 파일 크기: {file_size_mb:.1f} MB")
        # 대략적인 추정: 1GB당 2-5분 (스레드 수와 하드웨어에 따라 다름)
        estimated_minutes = (file_size_mb / 1024) * 3 * (8 / max(args.threads, 1))
        print(f"[resample] 예상 처리 시간: 약 {estimated_minutes:.1f}분 (스레드={args.threads})")
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "full_resample.duckdb"
    
    step_time = overall_start
    step_time = _print_step("데이터베이스 연결", overall_start)
    con = _connect(db_path, args.threads, args.keep_db)
    if args.seed is not None:
        # DuckDB seed syntax varies by version; try SET seed first, ignore failures.
        try:
            con.execute(f"SET seed = {int(args.seed)}")
        except Exception:
            try:
                con.execute(f"PRAGMA seed={int(args.seed)}")
            except Exception:
                print("[resample][warn] Unable to set RNG seed; continuing without deterministic sampling")
    step_time = _print_step("데이터베이스 연결", overall_start, step_time)

    step_time = _print_step("CSV 파일 로드", overall_start, step_time)
    source_sql = _build_csv_source(input_path, args.limit)
    con.execute(f"CREATE OR REPLACE VIEW raw_input AS SELECT * FROM {source_sql}")
    step_time = _print_step("CSV 파일 로드", overall_start, step_time)
    
    # Raw input 행 수 확인 (진행률 추정용)
    raw_count_start = time.time()
    raw_count = int(con.execute("SELECT COUNT(*) FROM raw_input").fetchone()[0])
    print(f"[resample]   → 로드된 행 수: {raw_count:,}")
    step_time = _print_step("타임라인 테이블 생성 (live streamer 수 계산)", overall_start, step_time)
    
    # 최적화: VIEW 대신 TABLE로 materialize하여 재계산 방지
    # 기존 VIEW/TABLE이 있으면 먼저 삭제
    try:
        con.execute("DROP VIEW IF EXISTS timeline")
    except Exception:
        pass
    try:
        con.execute("DROP TABLE IF EXISTS timeline")
    except Exception:
        pass
    con.execute(
        """
        CREATE TABLE timeline AS
        SELECT start AS bucket,
               COUNT(DISTINCT streamer) AS live_streamers
        FROM raw_input
        GROUP BY 1
        """
    )
    step_time = _print_step("타임라인 테이블 생성 (live streamer 수 계산)", overall_start, step_time)
    
    # 타임라인 행 수 확인 (인덱스 생성 전에 COUNT 수행)
    timeline_count = int(con.execute("SELECT COUNT(*) FROM timeline").fetchone()[0])
    print(f"[resample]   → 타임스텝 수: {timeline_count:,}")
    
    # 인덱스 생성으로 조인 성능 향상 (COUNT 이후에 생성하여 성능 영향 최소화)
    try:
        con.execute("CREATE INDEX IF NOT EXISTS idx_timeline_bucket ON timeline(bucket)")
    except Exception:
        pass  # 인덱스가 이미 존재하거나 지원하지 않을 수 있음
    
    step_time = _print_step("라이브 필터링 테이블 생성 (live streamer 수 기준)", overall_start, step_time)
    # 최적화: VIEW 대신 TABLE로 materialize
    # 기존 VIEW/TABLE이 있으면 먼저 삭제
    try:
        con.execute("DROP VIEW IF EXISTS filtered")
    except Exception:
        pass
    try:
        con.execute("DROP TABLE IF EXISTS filtered")
    except Exception:
        pass
    con.execute(
        f"""
        CREATE TABLE filtered AS
        SELECT r.*, t.live_streamers
        FROM raw_input r
        JOIN timeline t ON r.start = t.bucket
        WHERE t.live_streamers BETWEEN {int(args.min_live)} AND {int(args.max_live)}
        """
    )
    step_time = _print_step("라이브 필터링 테이블 생성 (live streamer 수 기준)", overall_start, step_time)
    
    # 필터링된 행 수 확인 (인덱스 생성 전에 COUNT 수행)
    filtered_count = int(con.execute("SELECT COUNT(*) FROM filtered").fetchone()[0])
    print(f"[resample]   → 필터링 후 행 수: {filtered_count:,} ({100.0 * filtered_count / max(raw_count, 1):.1f}%)")
    
    # 인덱스 생성으로 후속 조인 성능 향상 (COUNT 이후에 생성하여 성능 영향 최소화)
    try:
        con.execute("CREATE INDEX IF NOT EXISTS idx_filtered_user ON filtered(user)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_filtered_start ON filtered(start)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_filtered_streamer_start ON filtered(streamer, start)")
    except Exception:
        pass  # 인덱스가 이미 존재하거나 지원하지 않을 수 있음

    max_users = args.max_users if args.max_users and args.max_users > 0 else None
    target_interactions = args.target_interactions if args.target_interactions and args.target_interactions > 0 else None
    max_users_sql = "NULL" if max_users is None else str(int(max_users))
    target_sql = "NULL" if target_interactions is None else str(int(target_interactions))
    slots = int(SLOTS_PER_DAY)
    min_interactions = int(args.min_interactions)
    min_active_days = int(args.min_active_days)

    step_time = _print_step("사용자 필터링 및 랭킹", overall_start, step_time)
    if args.user_bins:
        # Stratified caps by interaction bins; random order for diversity (seeded if provided)
        values_clause = ", ".join(
            f"({idx}, {b['min_interactions']}, "
            f"{'NULL' if b['max_interactions'] is None else b['max_interactions']}, "
            f"{'NULL' if b['max_users'] is None else b['max_users']})"
            for idx, b in enumerate(args.user_bins)
        )
        con.execute(
            f"""
            CREATE OR REPLACE TABLE user_bins AS
            SELECT * FROM (VALUES {values_clause}) AS t(bin_id, min_interactions, max_interactions, max_users);
            """
        )
        con.execute(
            f"""
            CREATE OR REPLACE TABLE keep_users AS
            WITH user_stats AS (
                SELECT
                    user,
                    COUNT(*) AS interactions,
                    COUNT(DISTINCT CAST(floor(start / {slots}) AS BIGINT)) AS active_days
                FROM filtered
                GROUP BY user
                HAVING COUNT(*) >= {min_interactions}
                   AND COUNT(DISTINCT CAST(floor(start / {slots}) AS BIGINT)) >= {min_active_days}
            ),
            tagged AS (
                SELECT
                    us.*,
                    ub.bin_id,
                    ub.max_users
                FROM user_stats us
                JOIN user_bins ub
                  ON us.interactions >= ub.min_interactions
                 AND (ub.max_interactions IS NULL OR us.interactions < ub.max_interactions)
            ),
            ranked AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (PARTITION BY bin_id ORDER BY random()) AS bin_rank
                FROM tagged
            ),
            capped AS (
                SELECT *,
                       SUM(interactions) OVER (ORDER BY bin_id, bin_rank) AS cum_interactions
                FROM ranked
                WHERE (max_users IS NULL OR bin_rank <= max_users)
            )
            SELECT user, interactions, active_days, bin_id AS rank_idx, cum_interactions
            FROM capped
            WHERE ({target_sql} IS NULL OR cum_interactions - interactions < {target_sql})
            """
        )
    else:
        con.execute(
            f"""
            CREATE OR REPLACE TABLE keep_users AS
            WITH user_stats AS (
                SELECT
                    user,
                    COUNT(*) AS interactions,
                    COUNT(DISTINCT CAST(floor(start / {slots}) AS BIGINT)) AS active_days
                FROM filtered
                GROUP BY user
                HAVING COUNT(*) >= {min_interactions}
                   AND COUNT(DISTINCT CAST(floor(start / {slots}) AS BIGINT)) >= {min_active_days}
            ),
            ranked AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (ORDER BY interactions DESC, user) AS rank_idx,
                    SUM(interactions) OVER (
                        ORDER BY interactions DESC, user
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) AS cum_interactions
                FROM user_stats
            )
            SELECT user, interactions, active_days, rank_idx, cum_interactions
            FROM ranked
            WHERE ({max_users_sql} IS NULL OR rank_idx <= {max_users_sql})
              AND ({target_sql} IS NULL OR cum_interactions - interactions < {target_sql})
            """
        )
    step_time = _print_step("사용자 필터링 및 랭킹", overall_start, step_time)
    
    # 선택된 사용자 수 확인
    kept_users_count = int(con.execute("SELECT COUNT(*) FROM keep_users").fetchone()[0])
    kept_interactions = int(con.execute("SELECT SUM(interactions) FROM keep_users").fetchone()[0] or 0)
    print(f"[resample]   → 선택된 사용자 수: {kept_users_count:,}, 총 상호작용: {kept_interactions:,}")
    
    step_time = _print_step("최종 테이블 생성 (사용자-상호작용 조인)", overall_start, step_time)
    # 최적화: VIEW 대신 TABLE로 materialize
    # 기존 VIEW/TABLE이 있으면 먼저 삭제
    try:
        con.execute("DROP VIEW IF EXISTS final_view")
    except Exception:
        pass
    try:
        con.execute("DROP TABLE IF EXISTS final_view")
    except Exception:
        pass
    con.execute(
        """
        CREATE TABLE final_view AS
        SELECT f.*, ku.interactions AS user_interactions
        FROM filtered f
        JOIN keep_users ku USING(user)
        """
    )
    # 인덱스 생성으로 후속 조인 및 정렬 성능 향상
    try:
        con.execute("CREATE INDEX IF NOT EXISTS idx_final_user_start ON final_view(user, start)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_final_streamer_start ON final_view(streamer, start)")
    except Exception:
        pass  # 인덱스가 이미 존재하거나 지원하지 않을 수 있음
    step_time = _print_step("최종 테이블 생성 (사용자-상호작용 조인)", overall_start, step_time)
    
    final_count = int(con.execute("SELECT COUNT(*) FROM final_view").fetchone()[0])
    print(f"[resample]   → 최종 행 수: {final_count:,}")
    
    step_time = _print_step("동시 시청자 수 계산", overall_start, step_time)
    viewer_count_expr = "COUNT(DISTINCT user)" if str(args.viewer_count_mode).lower() == "unique" else "COUNT(*)"
    # 최적화: VIEW 대신 TABLE로 materialize
    # 기존 VIEW/TABLE이 있으면 먼저 삭제
    try:
        con.execute("DROP VIEW IF EXISTS viewer_counts")
    except Exception:
        pass
    try:
        con.execute("DROP TABLE IF EXISTS viewer_counts")
    except Exception:
        pass
    con.execute(
        f"""
        CREATE TABLE viewer_counts AS
        SELECT streamer,
               start,
               {viewer_count_expr} AS concurrent_viewers
        FROM final_view
        GROUP BY 1,2
        """
    )
    step_time = _print_step("동시 시청자 수 계산", overall_start, step_time)
    
    viewer_counts_count = int(con.execute("SELECT COUNT(*) FROM viewer_counts").fetchone()[0])
    print(f"[resample]   → 고유 (streamer, start) 쌍: {viewer_counts_count:,}")
    
    # 인덱스 생성으로 후속 조인 성능 향상 (COUNT 이후에 생성하여 성능 영향 최소화)
    try:
        con.execute("CREATE INDEX IF NOT EXISTS idx_viewer_streamer_start ON viewer_counts(streamer, start)")
    except Exception:
        pass  # 인덱스가 이미 존재하거나 지원하지 않을 수 있음

    step_time = _print_step("뷰어 버킷 계산", overall_start, step_time)
    if args.viewer_bucket_edges:
        bucket_edges = [int(val) for val in args.viewer_bucket_edges]
        quantile_list: list[float] = []
    else:
        quantile_list = _parse_quantiles(args.viewer_bucket_quantiles)
        bucket_edges = _compute_bucket_edges(con, quantile_list)
    bucket_id_expr, bucket_label_expr = _bucket_case_expr(bucket_edges, "vc.concurrent_viewers")
    step_time = _print_step("뷰어 버킷 계산", overall_start, step_time)
    print(f"[resample]   → 버킷 경계: {bucket_edges}")

    step_time = _print_step("Parquet 파일 쓰기", overall_start, step_time)
    # 최적화: ORDER BY는 대용량 데이터에서 매우 느리므로 제거
    # 필요시 나중에 정렬하거나 인덱스를 활용하여 정렬 비용 최소화
    dataset_query = f"""
    SELECT
        f.user,
        f.stream,
        f.streamer,
        f.start,
        f.stop,
        f.live_streamers,
        vc.concurrent_viewers AS viewer_count,
        {bucket_id_expr} AS viewer_bucket_id,
        {bucket_label_expr} AS viewer_bucket_label
    FROM final_view f
    LEFT JOIN viewer_counts vc
           ON f.streamer = vc.streamer
          AND f.start = vc.start
    """

    parquet_path = out_dir / "interactions.parquet"
    csv_path = out_dir / "interactions.csv"
    con.execute(f"COPY ({dataset_query}) TO ? (FORMAT 'parquet')", [str(parquet_path)])
    parquet_size_mb = _estimate_file_size_mb(parquet_path)
    step_time = _print_step("Parquet 파일 쓰기", overall_start, step_time)
    print(f"[resample]   → 파일 크기: {parquet_size_mb:.1f} MB")
    
    step_time = _print_step("CSV 파일 쓰기", overall_start, step_time)
    con.execute(f"COPY ({dataset_query}) TO ? (FORMAT 'csv', HEADER false)", [str(csv_path)])
    csv_size_mb = _estimate_file_size_mb(csv_path)
    step_time = _print_step("CSV 파일 쓰기", overall_start, step_time)
    print(f"[resample]   → 파일 크기: {csv_size_mb:.1f} MB")

    step_time = _print_step("통계 계산", overall_start, step_time)
    stats = {
        "input": str(input_path),
        "limit": args.limit,
        "min_live": args.min_live,
        "max_live": args.max_live,
        "min_interactions": args.min_interactions,
        "min_active_days": args.min_active_days,
        "max_users": max_users,
        "target_interactions": target_interactions,
        "viewer_bucket_quantiles": quantile_list,
        "viewer_bucket_edges": bucket_edges,
        "viewer_count_mode": args.viewer_count_mode,
        "user_bins": args.user_bins,
    }
    stats["rows_raw"] = raw_count
    stats["rows_filtered_live"] = filtered_count
    stats["rows_final"] = final_count
    stats["num_users_final"] = int(con.execute("SELECT COUNT(DISTINCT user) FROM final_view").fetchone()[0])
    stats["live_slate_summary"] = dict(
        zip(
            [
                "avg_live_streamers",
                "p50_live_streamers",
                "p90_live_streamers",
                "min_live_streamers",
                "max_live_streamers",
            ],
            _summarize_row(
                con.execute(
                    """
                    SELECT
                        AVG(live_streamers),
                        QUANTILE(live_streamers, 0.5),
                        QUANTILE(live_streamers, 0.9),
                        MIN(live_streamers),
                        MAX(live_streamers)
                    FROM final_view
                    """
                ).fetchone()
            ),
        )
    )
    stats["viewer_count_summary"] = dict(
        zip(
            ["avg_viewers", "p50_viewers", "p90_viewers", "max_viewers"],
            _summarize_row(
                con.execute(
                    """
                    SELECT
                        AVG(concurrent_viewers),
                        QUANTILE(concurrent_viewers, 0.5),
                        QUANTILE(concurrent_viewers, 0.9),
                        MAX(concurrent_viewers)
                    FROM viewer_counts
                    """
                ).fetchone()
            ),
        )
    )
    step_time = _print_step("통계 계산", overall_start, step_time)
    
    step_time = _print_step("메타데이터 파일 쓰기", overall_start, step_time)
    stats_path = out_dir / "stats.json"
    with stats_path.open("w", encoding="utf-8") as fout:
        json.dump(stats, fout, indent=2)
    buckets_path = out_dir / "viewer_buckets.json"
    bucket_desc = (
        "Viewer buckets derived from fixed edges"
        if args.viewer_bucket_edges
        else "Viewer buckets derived from concurrent viewer quantiles"
    )
    with buckets_path.open("w", encoding="utf-8") as fout:
        json.dump(
            {
                "quantiles": quantile_list,
                "edges": bucket_edges,
                "description": bucket_desc,
            },
            fout,
            indent=2,
        )
    step_time = _print_step("메타데이터 파일 쓰기", overall_start, step_time)
    
    # 최종 요약
    total_time = time.time() - overall_start
    print(f"\n[resample] ========================================")
    print(f"[resample] 전체 처리 완료!")
    print(f"[resample] 총 소요 시간: {_format_time(total_time)}")
    print(f"[resample] 처리 속도: {raw_count / max(total_time, 0.001):,.0f} 행/초")
    print(f"[resample] ========================================")


if __name__ == "__main__":
    main()
