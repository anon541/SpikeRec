import os
import torch
import pickle
import numpy as np
import pandas as pd
import math
import time
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, Counter
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False

from data.processing.sampling import *

import torch.utils.data as data
from torch.utils.data import DataLoader

VIEWER_SPIKE_SCALE = 1000


def _dataset_cache_key(dataset_path: Path) -> str:
    if dataset_path.is_dir():
        return dataset_path.name
    stem = dataset_path.stem
    if stem == "interactions":
        parent = dataset_path.parent.name or "dataset"
        return f"{parent}_{stem}"
    return stem


def _sequence_cache_key(args) -> str:
    dataset_key = _dataset_cache_key(Path(args.dataset))
    seq_len = getattr(args, "seq_len", None)
    if seq_len:
        return f"{dataset_key}_s{int(seq_len)}"
    return dataset_key


class ViewerTrendLookup:
    """Lookup viewer trends by (streamer, timestamp) and compute spike features."""
    
    def __init__(self, viewer_trends_path: Optional[Path] = None, 
                 sample_streamers: Optional[set] = None, max_rows: Optional[int] = None):
        self.trends = None
        self.streamer_to_id = None
        self.id_to_streamer = None
        self.trend_cache = {}  # Cache for computed spike features
        
        # 규모별 파라미터
        self.streamer_avg_viewers = {}  # streamer -> avg_viewer (라이브 시점만)
        self.size_thresholds = {'q33': 1.90, 'q67': 4.59}  # 분석 결과 기반
        
        # Baseline 및 percentile 계산용
        self.streamer_baselines = {}  # streamer -> baseline (과거 평균)
        self.streamer_percentiles = {}  # streamer -> percentile [0, 1] (Mean popularity rank)
        
        # New Percentile Features Cache
        self.ts_quantiles = {}        # timestamp -> [p0, ... p100] viewer counts
        self.streamer_quantiles = {}  # streamer -> [p0, ... p100] viewer counts
        
        if viewer_trends_path and viewer_trends_path.exists():
            self._load_trends(viewer_trends_path, sample_streamers=sample_streamers, max_rows=max_rows)
            self._compute_streamer_avg_viewers()
            self._compute_percentile_stats() # New: Compute detailed percentile stats

            # Percentile 캐시 로드 시도
            cache_path = Path(viewer_trends_path).parent / "streamer_percentiles.pkl"
            if cache_path.exists():
                try:
                    load_start = time.perf_counter()
                    with open(cache_path, 'rb') as f:
                        self.streamer_percentiles = pickle.load(f)
                    load_time = time.perf_counter() - load_start
                    print(f"[viewer_trends] [timing] Loaded percentile cache from {cache_path}: {load_time:.3f}s ({len(self.streamer_percentiles):,} streamers)")
                except Exception as e:
                    print(f"[viewer_trends] [warn] Failed to load percentile cache: {e}, computing...")
                    self._compute_streamer_percentiles()
                    # 저장 시도
                    try:
                        with open(cache_path, 'wb') as f:
                            pickle.dump(self.streamer_percentiles, f)
                        print(f"[viewer_trends] Saved percentile cache to {cache_path}")
                    except Exception:
                        pass
            else:
                self._compute_streamer_percentiles()
                # 저장 시도
                try:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(cache_path, 'wb') as f:
                        pickle.dump(self.streamer_percentiles, f)
                    print(f"[viewer_trends] Saved percentile cache to {cache_path}")
                except Exception as e:
                    print(f"[viewer_trends] [warn] Failed to save percentile cache: {e}")
    
    def _load_trends(self, path: Path, sample_streamers: Optional[set] = None, max_rows: Optional[int] = None):
        """Load viewer trends from parquet file.
        
        Args:
            path: Path to viewer trends parquet file
            sample_streamers: If provided, only load trends for these streamers (for dry-run)
            max_rows: If provided, limit number of rows to load (for dry-run)
        """
        print(f"[viewer_trends] Loading viewer trends from: {path}")
        
        # For dry-run or sampling, use efficient partial reading
        if sample_streamers is not None or max_rows is not None:
            try:
                import pyarrow.parquet as pq
                parquet_file = pq.ParquetFile(path)
                
                if sample_streamers is not None:
                    # Filter by streamers - read only first row group for speed
                    print(f"[viewer_trends] Filtering to {len(sample_streamers):,} streamers for dry-run...")
                    try:
                        import pyarrow.parquet as pq
                        parquet_file = pq.ParquetFile(path)
                        # Read only first row group for dry-run (much faster)
                        table = parquet_file.read_row_groups([0], columns=['streamer', 'timestamp', 'viewer_count'])
                        df = table.to_pandas()
                        # Filter by streamers
                        df = df[df['streamer'].isin(sample_streamers)]
                        if max_rows is not None and len(df) > max_rows:
                            df = df.head(max_rows)
                        print(f"[viewer_trends] Loaded {len(df):,} rows from first row group (dry-run)")
                    except Exception as e:
                        print(f"[viewer_trends] PyArrow read failed: {e}, using pandas fallback...")
                        # Fallback: read full file but limit early
                        df = pd.read_parquet(path, nrows=max_rows if max_rows else None)
                        df = df[df['streamer'].isin(sample_streamers)]
                        if max_rows is not None and len(df) > max_rows:
                            df = df.head(max_rows)
                elif max_rows is not None:
                    # Just limit rows
                    print(f"[viewer_trends] Limiting to {max_rows:,} rows for dry-run...")
                    table = parquet_file.read_row_groups([0])
                    if table.num_rows > max_rows:
                        table = table.slice(0, max_rows)
                    df = table.to_pandas()
                else:
                    df = pd.read_parquet(path)
            except Exception as e:
                print(f"[viewer_trends] Partial read failed: {e}, falling back to full load")
                df = pd.read_parquet(path)
                if sample_streamers is not None:
                    df = df[df['streamer'].isin(sample_streamers)]
                if max_rows is not None and len(df) > max_rows:
                    df = df.head(max_rows)
        else:
            df = pd.read_parquet(path)
        
        print(f"[viewer_trends] Loaded {len(df):,} viewer trend records")
        
        # Create lookup: (streamer, timestamp) -> viewer_count
        self.trends = df.set_index(['streamer', 'timestamp'])['viewer_count'].to_dict()
        print(f"[viewer_trends] Created lookup table with {len(self.trends):,} entries")
    
    def _compute_streamer_avg_viewers(self):
        """Streamer별 평균 viewer count 계산 (라이브 시점만)"""
        if self.trends is None:
            return
        
        print("[viewer_trends] Computing streamer average viewer counts...")
        streamer_counts = defaultdict(list)
        
        # 라이브 시점(viewer_count > 0)만 수집
        for (streamer, timestamp), count in self.trends.items():
            if count > 0:
                streamer_counts[streamer].append(count)
        
        # 평균 계산
        for streamer, counts in streamer_counts.items():
            if len(counts) > 0:
                self.streamer_avg_viewers[streamer] = np.mean(counts)
        
        print(f"[viewer_trends] Computed averages for {len(self.streamer_avg_viewers):,} streamers")
    
    def _compute_streamer_percentiles(self):
        """Streamer별 popularity percentile 계산 (O(n log n) 최적화)"""
        if not self.streamer_avg_viewers:
            return
        
        # This computes the static popularity rank of a streamer
        start_time = time.perf_counter()
        print("[viewer_trends] Computing streamer popularity percentiles...")
        
        # Streamer와 avg_viewer를 함께 저장
        streamer_avg_list = [(streamer, avg) for streamer, avg in self.streamer_avg_viewers.items()]
        if len(streamer_avg_list) == 0:
            return
        
        # avg_viewer 기준으로 정렬 (O(n log n))
        sorted_streamer_avg = sorted(streamer_avg_list, key=lambda x: x[1])
        n = len(sorted_streamer_avg)
        
        # 정렬된 순서로 percentile 할당 (O(n))
        for rank, (streamer, avg_viewer) in enumerate(sorted_streamer_avg):
            percentile = rank / n if n > 0 else 0.5
            self.streamer_percentiles[streamer] = percentile
            
        print(f"[viewer_trends] Computed percentiles for {len(self.streamer_percentiles):,} streamers (total: {time.perf_counter() - start_time:.3f}s)")

    def _compute_percentile_stats(self):
        """Global & Self Percentile 계산을 위한 통계량 캐싱 (Quantiles)"""
        print("[viewer_trends] Computing global and self percentile stats...")
        start_time = time.perf_counter()
        
        # 1. Collect counts efficiently
        ts_counts = defaultdict(list)  # timestamp -> [counts]
        streamer_counts = defaultdict(list) # streamer -> [counts]
        
        for (streamer, ts), count in self.trends.items():
            if count > 0:
                ts_counts[ts].append(count)
                streamer_counts[streamer].append(count)
        
        step1_time = time.perf_counter()
        
        # 2. Compute Quantiles (0~100%)
        # 메모리 절약을 위해 1% 단위(101개 포인트)만 저장
        quantiles = np.linspace(0, 100, 101)
        
        self.ts_quantiles = {}
        for ts, counts in ts_counts.items():
            if counts:
                self.ts_quantiles[ts] = np.percentile(counts, quantiles)
                
        self.streamer_quantiles = {}
        for s, counts in streamer_counts.items():
            if counts:
                self.streamer_quantiles[s] = np.percentile(counts, quantiles)
        
        step2_time = time.perf_counter()
        print(f"[viewer_trends] [timing] Collect counts: {step1_time - start_time:.3f}s")
        print(f"[viewer_trends] [timing] Compute quantiles: {step2_time - step1_time:.3f}s")
        print(f"[viewer_trends] Computed stats for {len(self.ts_quantiles):,} timestamps and {len(self.streamer_quantiles):,} streamers")

    def _get_percentile_rank(self, value, quantiles):
        """Quantile 배열에서 값의 백분위수(0.0~1.0) 추정 (Linear Interpolation)"""
        if quantiles is None:
            return 0.5
        # np.searchsorted로 위치 찾기
        # quantiles는 오름차순 정렬되어 있음 (0% -> 100%)
        if value <= quantiles[0]: return 0.0
        if value >= quantiles[-1]: return 1.0
        
        idx = np.searchsorted(quantiles, value)
        # value는 quantiles[idx-1] < value <= quantiles[idx]
        
        lower = quantiles[idx-1]
        upper = quantiles[idx]
        
        # 해당 구간 내에서의 비율
        if upper == lower:
            fraction = 0.0
        else:
            fraction = (value - lower) / (upper - lower)
            
        # idx는 1~100 사이. (idx-1)은 0~99.
        # 각 구간은 1% (0.01) 크기.
        # percentile = (구간 시작 % + 구간 내 비율 * 1%)
        rank = (idx - 1 + fraction) * 0.01
        return rank

    def get_global_rank_percentile(self, timestamp, count):
        q = self.ts_quantiles.get(timestamp)
        return self._get_percentile_rank(count, q)

    def get_self_rank_percentile(self, streamer, count):
        q = self.streamer_quantiles.get(streamer)
        return self._get_percentile_rank(count, q)
    
    def get_baseline(self, streamer: str, timestamp: int, window_size: int = 20) -> float:
        """과거 window의 평균 viewer count를 baseline으로 계산 (최적화)"""
        if self.trends is None:
            return 0.0
        
        # 최적화: trends dict에서 직접 조회 (get_viewer_count 호출 최소화)
        baseline_counts = []
        start_t = max(0, timestamp - window_size)
        for t in range(start_t, timestamp):
            count = self.trends.get((streamer, t), 0)
            if count > 0:
                baseline_counts.append(count)
        
        if len(baseline_counts) == 0:
            # Baseline이 없으면 streamer 평균 사용
            return self.streamer_avg_viewers.get(streamer, 0.0)
        
        return np.mean(baseline_counts)
    
    def get_baseline_batch(self, streamer_timestamp_pairs: list, window_size: int = 20) -> list:
        """Batch로 baseline 계산 (완전 벡터화 - Python 루프 제거)"""
        if self.trends is None:
            return [0.0] * len(streamer_timestamp_pairs)
        
        n = len(streamer_timestamp_pairs)
        if n == 0:
            return []
        
        baselines = np.zeros(n, dtype=np.float32)
        
        # 모든 필요한 키를 한 번에 수집 (벡터화)
        all_keys_list = []
        key_to_pair_indices = {}  # 각 키가 어떤 pair들에 속하는지
        
        for pair_idx, (streamer, timestamp) in enumerate(streamer_timestamp_pairs):
            start_t = max(0, timestamp - window_size)
            for t in range(start_t, timestamp):
                key = (streamer, t)
                all_keys_list.append(key)
                if key not in key_to_pair_indices:
                    key_to_pair_indices[key] = []
                key_to_pair_indices[key].append(pair_idx)
        
        # 한 번에 모든 조회 수행 (벡터화)
        counts_cache = {}
        for key in all_keys_list:
            if key not in counts_cache:  # 중복 제거
                counts_cache[key] = self.trends.get(key, 0)
        
        # 각 pair별로 키를 그룹화하고 벡터화된 계산
        # pair_idx별로 필요한 키들을 수집
        pair_key_counts = {}  # pair_idx -> [counts]
        for key, pair_indices in key_to_pair_indices.items():
            count = counts_cache[key]
            for pair_idx in pair_indices:
                if pair_idx not in pair_key_counts:
                    pair_key_counts[pair_idx] = []
                pair_key_counts[pair_idx].append(count)
        
        # 벡터화된 baseline 계산 (NumPy 연산만 사용)
        for pair_idx in range(n):
            if pair_idx in pair_key_counts:
                counts = np.array(pair_key_counts[pair_idx], dtype=np.float32)
                positive_counts = counts[counts > 0]
                if len(positive_counts) > 0:
                    baselines[pair_idx] = np.mean(positive_counts)
                else:
                    # Fallback to streamer average
                    streamer = streamer_timestamp_pairs[pair_idx][0]
                    baselines[pair_idx] = self.streamer_avg_viewers.get(streamer, 0.0)
            else:
                # No keys for this pair (empty window)
                streamer = streamer_timestamp_pairs[pair_idx][0]
                baselines[pair_idx] = self.streamer_avg_viewers.get(streamer, 0.0)
        
        return baselines.tolist()
    
    def get_viewer_count_batch(self, streamer_timestamp_pairs: list) -> list:
        """Batch로 viewer count 조회 (최적화)"""
        if self.trends is None:
            return [0] * len(streamer_timestamp_pairs)
        
        # List comprehension이 이미 최적화되어 있지만, 명시적으로 처리
        # trends dict 조회는 O(1)이므로 배치로 처리해도 개별 조회와 비슷하지만
        # 한 번에 처리하면 약간의 오버헤드 감소
        keys = [(streamer, ts) for streamer, ts in streamer_timestamp_pairs]
        return [self.trends.get(key, 0) for key in keys]
    
    def get_percentile(self, streamer: str) -> float:
        """Streamer의 popularity percentile 반환 [0, 1]"""
        return self.streamer_percentiles.get(streamer, 0.5)
    
    def get_window_rank_percentile(self, streamer: str, timestamp: int, current_count: int, window_size: int = 20) -> float:
        """
        Get the percentile rank of the current count within the streamer's recent window.
        Returns 0.0 ~ 1.0.
        """
        if self.trends is None:
            return 0.5
            
        # Get window counts
        window_counts = []
        for t in range(max(0, timestamp - window_size), timestamp): # Exclude current
            val = self.trends.get((streamer, t), 0)
            if val > 0:
                window_counts.append(val)
        
        if not window_counts:
            return 0.5 # No history
            
        # Calculate rank
        # Count how many in window are <= current
        # Use np.searchsorted style logic but simple
        window_counts.sort()
        rank = 0
        for x in window_counts:
            if x < current_count:
                rank += 1
            else:
                break
        
        # Normalize to 0~1
        return rank / len(window_counts)

    def _get_window_size_by_size(self, avg_viewer: float) -> int:
        """규모별 window size 결정"""
        q33 = self.size_thresholds['q33']
        q67 = self.size_thresholds['q67']
        
        if avg_viewer < q33:
            return 5  # 소규모: 빠른 변화 감지
        elif avg_viewer < q67:
            return 10  # 중간: 기본값
        else:
            return 20  # 대규모: 안정적인 기준
    
    def _scale_z_score_by_size(self, z_score: float, avg_viewer: float) -> float:
        """규모별 z-score 스케일링"""
        q33 = self.size_thresholds['q33']
        q67 = self.size_thresholds['q67']
        
        if avg_viewer < q33:
            return z_score  # 소규모: 원본
        elif avg_viewer < q67:
            return z_score * 0.9  # 중간: 약간 감소
        else:
            return z_score * 1.1  # 대규모: 약간 증가
    
    def set_streamer_mapping(self, streamer_to_id: dict, id_to_streamer: dict):
        """Set mapping between streamer names and IDs."""
        self.streamer_to_id = streamer_to_id
        self.id_to_streamer = id_to_streamer
    
    def get_viewer_count(self, streamer: str, timestamp: int) -> int:
        """Get viewer count for (streamer, timestamp)."""
        if self.trends is None:
            return 0
        return self.trends.get((streamer, timestamp), 0)
    
    def compute_spike_features(self, streamer: str, timestamp: int, window_size: Optional[int] = None, use_percentile: bool = False, use_percentile_3d: bool = False) -> Tuple:
        """
        Compute spike features.
        
        Args:
            streamer: Streamer name
            timestamp: Current timestamp
            window_size: Number of recent timestamps to use for statistics (None이면 규모별 자동 결정)
            use_percentile: If True, returns 4 features (Global, Self, Ratio, Conf).
            use_percentile_3d: If True, returns 3 features (Global, Self, WindowRank).
        
        Returns:
            Tuple of features.
        """
        if self.trends is None:
            return (0.0, 0.0, 0.0) if use_percentile_3d else ((0.0, 0.0, 0.0, 0.0) if use_percentile else (0.0, 0.0, 0.0))
        
        current_count = self.get_viewer_count(streamer, timestamp)
        if current_count == 0:
            return (0.0, 0.0, 0.0) if use_percentile_3d else ((0.0, 0.0, 0.0, 0.0) if use_percentile else (0.0, 0.0, 0.0))

        # 규모별 window size 결정
        if window_size is None:
            avg_viewer = self.streamer_avg_viewers.get(streamer, 0.0)
            window_size = self._get_window_size_by_size(avg_viewer)
        
        # Get viewer counts for recent timestamps
        recent_counts = []
        for t in range(max(0, timestamp - window_size), timestamp + 1):
            count = self.get_viewer_count(streamer, t)
            if count > 0:
                recent_counts.append(count)
        
        if len(recent_counts) == 0:
            return (0.0, 0.0, 0.0) if use_percentile_3d else ((0.0, 0.0, 0.0, 0.0) if use_percentile else (0.0, 0.0, 0.0))

        if use_percentile_3d:
            # New 3-Feature System (All 0~1 Rank)
            global_rank = self.get_global_rank_percentile(timestamp, current_count)
            self_rank = self.get_self_rank_percentile(streamer, current_count)
            window_rank = self.get_window_rank_percentile(streamer, timestamp, current_count, window_size)
            return global_rank, self_rank, window_rank

        # Common Statistics
        mean_count = np.mean(recent_counts)
        max_count = max(recent_counts) if recent_counts else 1.0
        
        # 3. viewer_ratio_recent: ratio of current to recent average
        viewer_ratio_recent = current_count / max(mean_count, 1.0)
        
        # 4. viewer_confidence (Peak Ratio): current / max in window
        viewer_confidence = current_count / max(max_count, 1.0)

        if use_percentile:
            # New Feature Set (4D)
            # 1. Global Rank Percentile
            global_rank = self.get_global_rank_percentile(timestamp, current_count)
            
            # 2. Self History Percentile
            self_rank = self.get_self_rank_percentile(streamer, current_count)
            
            return global_rank, self_rank, viewer_ratio_recent, viewer_confidence
        else:
            # Legacy Feature Set (3D)
            std_count = np.std(recent_counts) if len(recent_counts) > 1 else 1.0
            viewer_z = (current_count - mean_count) / max(std_count, 1.0) if std_count > 0 else 0.0
            
            # 규모별 z-score 스케일링 적용
            avg_viewer = self.streamer_avg_viewers.get(streamer, 0.0)
            viewer_z = self._scale_z_score_by_size(viewer_z, avg_viewer)
            
            return viewer_z, viewer_ratio_recent, viewer_confidence
    
    def compute_spike_features_batch(
        self,
        streamer_timestamp_pairs: list,
        num_workers: int = 4,
        use_percentile: bool = False,
        use_percentile_3d: bool = False,
        use_hybrid: bool = False,
        window_size: Optional[int] = None,
    ) -> np.ndarray:
        """Compute spike features for multiple (streamer, timestamp) pairs efficiently.

        Args:
            streamer_timestamp_pairs: List of (streamer, timestamp) tuples
            num_workers: Number of parallel workers (default: 4)
            use_percentile: Use percentile feature set (4D)
            use_percentile_3d: Use 3D percentile set (p_glob, p_self, p_window)
            use_hybrid: Use hybrid features (log_count + percentile stats)
            window_size: Optional fixed window size for spike stats

        Returns:
            np.ndarray of shape (N, 4) aligned with model spike feature layout
        """
        if self.trends is None or len(streamer_timestamp_pairs) == 0:
            return np.zeros((len(streamer_timestamp_pairs), 4), dtype=np.float32)

        def _feature_for(streamer: str, timestamp: int) -> np.ndarray:
            if use_percentile_3d:
                g_rank, s_rank, w_rank = self.compute_spike_features(
                    streamer, timestamp, window_size=window_size, use_percentile_3d=True
                )
                return np.array([g_rank, s_rank, w_rank, 0.0], dtype=np.float32)
            if use_percentile:
                g_rank, s_rank, ratio, conf = self.compute_spike_features(
                    streamer, timestamp, window_size=window_size, use_percentile=True
                )
                return np.array([g_rank, s_rank, ratio, conf], dtype=np.float32)
            if use_hybrid:
                _, s_rank, ratio, conf = self.compute_spike_features(
                    streamer, timestamp, window_size=window_size, use_percentile=True
                )
                viewer_count = self.get_viewer_count(streamer, timestamp)
                log_count = math.log1p(max(viewer_count, 0))
                return np.array([log_count, s_rank, ratio, conf], dtype=np.float32)

            viewer_z, viewer_ratio, viewer_conf = self.compute_spike_features(
                streamer, timestamp, window_size=window_size
            )
            viewer_count = self.get_viewer_count(streamer, timestamp)
            log_count = math.log1p(max(viewer_count, 0))
            return np.array([viewer_z, viewer_ratio, log_count, viewer_conf], dtype=np.float32)

        # Small batch: sequential to avoid overhead.
        if len(streamer_timestamp_pairs) < 100:
            results = np.zeros((len(streamer_timestamp_pairs), 4), dtype=np.float32)
            for idx, (streamer, timestamp) in enumerate(streamer_timestamp_pairs):
                results[idx] = _feature_for(streamer, timestamp)
            return results

        # Large batch: parallel computation.
        results = np.zeros((len(streamer_timestamp_pairs), 4), dtype=np.float32)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {
                executor.submit(_feature_for, streamer, timestamp): idx
                for idx, (streamer, timestamp) in enumerate(streamer_timestamp_pairs)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = np.zeros(4, dtype=np.float32)

        return results


def save_sequences_to_parquet(datalist, output_path: Path):
    """Save sequence datalist to Parquet format for fast loading.
    
    Args:
        datalist: List of sequences (each sequence is a list of tensors)
        output_path: Path to save parquet file
    """
    if not HAS_PARQUET:
        print(f"[cache] WARNING: pyarrow not available, skipping Parquet save")
        return
    
    print(f"[cache] Converting {len(datalist)} sequences to Parquet...")
    
    # Convert tensors to lists for storage
    records = []
    for seq in datalist:
        # Each seq is a list of tensors: [bpad, positions, inputs_ts, items, users, targets, targets_ts, ...]
        record = {f'field_{i}': tensor.tolist() if isinstance(tensor, torch.Tensor) else tensor 
                  for i, tensor in enumerate(seq)}
        records.append(record)
    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    
    # Save as Parquet with compression
    df.to_parquet(output_path, compression='snappy', index=False)
    print(f"[cache] Saved to {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")


def load_sequences_from_parquet(input_path: Path) -> list:
    """Load sequence datalist from Parquet format.
    
    Args:
        input_path: Path to parquet file
        
    Returns:
        List of sequences (each sequence is a list of tensors)
    """
    if not HAS_PARQUET:
        raise RuntimeError("pyarrow not available, cannot load Parquet files")
    
    print(f"[cache] Loading from Parquet: {input_path}")
    df = pd.read_parquet(input_path)
    
    # Convert back to list of tensor lists
    datalist = []
    for idx, row in df.iterrows():
        # Create tensors from values (handles both list and numpy array)
        seq = []
        for i in range(len(row)):
            val = row[f'field_{i}']
            # Convert to list first to avoid numpy read-only warning
            if hasattr(val, 'tolist'):
                val = val.tolist()
            seq.append(torch.LongTensor(val))
        datalist.append(seq)
    
    return datalist


def load_data(args):
    dataset_path = Path(getattr(args, "dataset", "dataset"))
    if dataset_path.is_dir():
        # Check for new resampled dataset format (parquet first, then csv)
        parquet_file = dataset_path / "interactions.parquet"
        csv_file = dataset_path / "interactions.csv"
        legacy_file = dataset_path / "100k.csv"
        if parquet_file.exists():
            infile = parquet_file
        elif csv_file.exists():
            infile = csv_file
        elif legacy_file.exists():
            infile = legacy_file
        else:
            raise FileNotFoundError(f"Dataset not found in {dataset_path}. Expected interactions.parquet, interactions.csv, or 100k.csv")
    else:
        infile = dataset_path
    if not infile.exists():
        raise FileNotFoundError(f"Dataset not found: {infile}")
    
    # Load viewer trends if available
    viewer_trends_path = None
    if dataset_path.is_dir():
        # Check for viewer trends in dataset directory
        trends_file = dataset_path / "streamer_viewer_trends.parquet"
        if trends_file.exists():
            viewer_trends_path = trends_file
        else:
            # Check in parent directory (for 100k.csv case)
            parent_trends = dataset_path.parent / "100k_viewer_trends" / "streamer_viewer_trends.parquet"
            if parent_trends.exists():
                viewer_trends_path = parent_trends
    
    # Also check explicit viewer_trends_path argument
    if hasattr(args, 'viewer_trends_path') and args.viewer_trends_path:
        viewer_trends_path = Path(args.viewer_trends_path)
    
    args.viewer_trends = ViewerTrendLookup(viewer_trends_path) if viewer_trends_path else None

    base_cols = ["user", "stream", "streamer", "start", "stop"]
    optional_cols = [
        "median_peak",
        "tier",
        "viewer_bucket_id",
        "viewer_bucket_label",
        "viewer_count",
        "mean_all",
        "std_all",
        "median_all",
        "q75",
        "q90",
        "viewer_z",
        "viewer_ratio_recent",
        "viewer_confidence",
    ]
    cols = base_cols.copy()
    cols.extend([c for c in optional_cols if c])
    nrows = None
    try:
        nrows = int(getattr(args, "debug_nrows", 0) or 0)
        if nrows <= 0:
            nrows = None
    except Exception:
        nrows = None

    suffix = infile.suffix.lower()
    if suffix == ".parquet":
        # Optimize: for debug/test, read only a sample of rows
        if nrows is not None and nrows > 0:
            # Use pyarrow to read only first nrows efficiently
            try:
                import pyarrow.parquet as pq
                parquet_file = pq.ParquetFile(infile)
                total_rows = parquet_file.metadata.num_rows
                # Read first row group and slice
                first_rg = parquet_file.read_row_groups([0], columns=None)
                if first_rg.num_rows >= nrows:
                    table = first_rg.slice(0, nrows)
                else:
                    # Need more rows: read multiple row groups
                    num_rgs = parquet_file.num_row_groups
                    rows_read = 0
                    tables = []
                    for rg_idx in range(min(num_rgs, 10)):  # Limit to first 10 row groups
                        rg_table = parquet_file.read_row_groups([rg_idx], columns=None)
                        remaining = nrows - rows_read
                        if rg_table.num_rows <= remaining:
                            tables.append(rg_table)
                            rows_read += rg_table.num_rows
                        else:
                            tables.append(rg_table.slice(0, remaining))
                            rows_read = nrows
                        if rows_read >= nrows:
                            break
                    import pyarrow as pa
                    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
                data_fu = table.to_pandas()
                print(f"[debug] read_parquet nrows={nrows} (from {total_rows:,} total)")
            except Exception as e:
                # Fallback: try CSV if parquet is corrupted
                print(f"[debug] pyarrow read failed: {e}, trying CSV fallback...")
                csv_file = infile.parent / "interactions.csv"
                if csv_file.exists():
                    data_fu = pd.read_csv(csv_file, header=None, names=base_cols + optional_cols, nrows=nrows)
                    print(f"[debug] read_csv nrows={nrows} (CSV fallback)")
                else:
                    raise ValueError(f"Parquet read failed and CSV not available: {e}")
        else:
            # Full load - try parquet first, fallback to CSV if corrupted
            try:
                data_fu = pd.read_parquet(infile)
            except Exception as e:
                print(f"[warning] Parquet read failed: {e}, trying CSV fallback...")
                csv_file = infile.parent / "interactions.csv"
                if csv_file.exists():
                    data_fu = pd.read_csv(csv_file, header=None, names=base_cols + optional_cols)
                    print("[info] Loaded from CSV instead of parquet")
                else:
                    raise ValueError(f"Parquet read failed and CSV not available: {e}")
        
        missing = [c for c in base_cols if c not in data_fu.columns]
        if missing:
            raise ValueError(f"Parquet file missing columns: {missing}")
        data_fu = data_fu[[c for c in data_fu.columns if c in (base_cols + optional_cols)]]
    else:
        data_fu = pd.read_csv(infile, header=None, names=base_cols, nrows=nrows)
        if nrows is not None:
            print(f"[debug] read_csv nrows={nrows}")
    viewer_mode = str(getattr(args, "viewer_feat_mode", "off") or "off")
    use_bucket = viewer_mode in ("bucket", "spike") and "viewer_bucket_id" in data_fu.columns
    use_spike = viewer_mode == "spike"

    if use_bucket:
        data_fu["viewer_bucket_id"] = data_fu["viewer_bucket_id"].fillna(0).astype(int)
        args.has_viewer_bucket = True
        args.num_viewer_buckets = int(data_fu["viewer_bucket_id"].max())
    else:
        args.has_viewer_bucket = False
        args.num_viewer_buckets = 0

    args.viewer_feat_mode = viewer_mode
    args.viewer_spike_scale = VIEWER_SPIKE_SCALE
    args.bucket_input_idx = None
    args.bucket_target_idx = None
    args.viewer_z_idx = None
    args.viewer_ratio_idx = None
    args.viewer_conf_idx = None
    args.streamer_tier = None
    args.ts_bucket = {}

    if use_spike:
        # Check if viewer trends are available
        viewer_trends = getattr(args, 'viewer_trends', None)
        is_dry_run = getattr(args, 'dry_run', False)
        has_trends = (viewer_trends is not None and 
                     hasattr(viewer_trends, 'trends') and 
                     viewer_trends.trends is not None and 
                     len(viewer_trends.trends) > 0)
        if not has_trends:
            # In dry-run, allow empty trends (will be populated later)
            if is_dry_run:
                print("[viewer_trends] Dry-run: viewer trends will be populated after data loading")
            else:
                # Fallback: require columns in data
                required_cols = ["viewer_z", "viewer_ratio_recent", "viewer_confidence"]
                missing = [c for c in required_cols if c not in data_fu.columns]
                if missing:
                    raise ValueError(f"viewer_feat_mode=spike requires either viewer_trends.parquet or columns {missing}")
        else:
            print("[viewer_trends] Using viewer trends for spike features (no columns required)")
    
    # Add one for padding
    data_fu.user = pd.factorize(data_fu.user)[0]+1
    data_fu['streamer_raw'] = data_fu.streamer
    data_fu.streamer = pd.factorize(data_fu.streamer)[0]+1
    if "tier" in data_fu.columns:
        tier_map = data_fu[["streamer", "tier"]].drop_duplicates("streamer")
        tier_lookup = {}
        for _, row in tier_map.iterrows():
            tier_val = row["tier"]
            if pd.isna(tier_val):
                tier_val = "unknown"
            tier_lookup[int(row["streamer"])] = str(tier_val)
        args.streamer_tier = tier_lookup
    print("Num users: ", data_fu.user.nunique())
    print("Num streamers: ", data_fu.streamer.nunique())
    print("Num interactions: ", len(data_fu))
    print("Estimated watch time: ", (data_fu['stop']-data_fu['start']).sum() * 10 / 60.0)
    # optional: keep only top-N users by interaction count for faster debug
    try:
        maxu = int(getattr(args, 'debug_max_users', 0) or 0)
    except Exception:
        maxu = 0
    if maxu and maxu > 0:
        vc = data_fu['user'].value_counts()
        keep_users = set(vc.head(maxu).index.tolist())
        before = len(data_fu)
        data_fu = data_fu[data_fu['user'].isin(keep_users)].copy()
        after = len(data_fu)
        print(f"[debug] filtered top-{maxu} users: {before} -> {after} rows")
    
    args.M = data_fu.user.max()+1 # users
    args.N = data_fu.streamer.max()+2 # items
    
    data_temp = data_fu.drop_duplicates(subset=['streamer','streamer_raw'])
    umap      = dict(zip(data_temp.streamer_raw.tolist(),data_temp.streamer.tolist()))
    id_to_streamer = {v: k for k, v in umap.items()}
    args.id_to_streamer = id_to_streamer  # Store for use in models
    
    # Set streamer mapping for viewer trends lookup
    if args.viewer_trends is not None:
        args.viewer_trends.set_streamer_mapping(umap, id_to_streamer)
    
    # Splitting and caching
    max_step = max(data_fu.start.max(),data_fu.stop.max())
    print("Num timesteps: ", max_step)
    args.max_step = max_step
    args.pivot_1  = max_step-500
    args.pivot_2  = max_step-250
    
    # Skip availability caching for dry-run or small debug samples (too slow)
    skip_av_cache = getattr(args, 'dry_run', False) or (nrows is not None and nrows > 0 and nrows < 10000)
    
    # Availability 캐시 파일 경로 (한 번 계산 후 재사용)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_key = _dataset_cache_key(Path(args.dataset))
    avail_cache_path = cache_dir / f"avail_cache_{dataset_key}_bucket{int(args.has_viewer_bucket)}_{max_step}_v2.pt"
    
    if not skip_av_cache:
        # 1) 캐시 로드 시도
        if avail_cache_path.exists():
            print(f"[loader] Loading availability cache from {avail_cache_path}")
            cache_obj = torch.load(avail_cache_path, map_location=args.device)
            args.ts = cache_obj['ts']
            args.max_avail = cache_obj['max_avail']
            args.av_tens = cache_obj['av_tens'].to(args.device)
            if args.has_viewer_bucket:
                args.ts_bucket = cache_obj.get('ts_bucket', {})
                args.av_bucket_tens = cache_obj.get('av_bucket_tens', None)
                if args.av_bucket_tens is not None:
                    args.av_bucket_tens = args.av_bucket_tens.to(args.device)
            print(f"[loader] Availability matrix loaded: shape={args.av_tens.shape}")
            return data_fu
        
        print("caching availability (incremental sweep, cacheable)")
        ts = {}
        ts_bucket = {}
        max_avail = 0
        
        # 이벤트 기반 sweep으로 시간축 반복 계산을 최소화
        start_events = defaultdict(list)
        stop_events = defaultdict(list)
        if args.has_viewer_bucket:
            for row in data_fu[['streamer','start','stop','viewer_bucket_id']].itertuples():
                start_events[row.start].append((row.streamer, int(row.viewer_bucket_id)))
                stop_time = row.stop if row.stop > row.start else row.start + 1
                stop_events[stop_time].append((row.streamer, int(row.viewer_bucket_id)))
        else:
            for row in data_fu[['streamer','start','stop']].itertuples():
                start_events[row.start].append(row.streamer)
                stop_time = row.stop if row.stop > row.start else row.start + 1
                stop_events[stop_time].append(row.streamer)
        
        active_counts = Counter()
        active_bucket = {}  # streamer -> latest bucket
        
        # max_av 계산을 위해 매 timestep에서 active set 길이를 기록
        for s in range(max_step+1):
            if s in start_events:
                if args.has_viewer_bucket:
                    for stream_id, bucket_id in start_events[s]:
                        active_counts[stream_id] += 1
                        active_bucket[stream_id] = bucket_id
                else:
                    for stream_id in start_events[s]:
                        active_counts[stream_id] += 1
            if s in stop_events:
                if args.has_viewer_bucket:
                    for stream_id, _ in stop_events[s]:
                        if active_counts[stream_id] > 0:
                            active_counts[stream_id] -= 1
                            if active_counts[stream_id] == 0:
                                active_bucket.pop(stream_id, None)
                else:
                    for stream_id in stop_events[s]:
                        if active_counts[stream_id] > 0:
                            active_counts[stream_id] -= 1
            # 현재 활성 스트리머 수
            active_streams = [k for k,v in active_counts.items() if v > 0]
            ts[s] = active_streams
            if args.has_viewer_bucket:
                ts_bucket[s] = [active_bucket.get(stream_id, 0) for stream_id in active_streams]
            max_avail = max(max_avail, len(active_streams))
        args.max_avail = max_avail
        args.ts = ts
        if args.has_viewer_bucket:
            args.ts_bucket = ts_bucket
        print("max_avail: ", max_avail)
        
        # Compute availability matrix of size (num_timesteps x max_available)
        print("[loader] Building availability matrix (fast path)...")
        av_tens = torch.zeros(max_step+1, max_avail, dtype=torch.long)
        av_bucket_tens = torch.zeros(max_step+1, max_avail, dtype=torch.long) if args.has_viewer_bucket else None
        
        # NumPy/torch batch fill to reduce Python overhead
        for k, v in ts.items():
            if len(v) == 0:
                continue
            v_tensor = torch.as_tensor(v, dtype=torch.long)
            av_tens[k, :len(v)] = v_tensor
            if args.has_viewer_bucket:
                bucket_list = ts_bucket.get(k, [])
                if bucket_list:
                    av_bucket_tens[k, :len(bucket_list)] = torch.as_tensor(bucket_list, dtype=torch.long)
        
        print(f"[loader] Moving availability matrix to device: {args.device}")
        args.av_tens = av_tens.to(args.device)
        if args.has_viewer_bucket and av_bucket_tens is not None:
            args.av_bucket_tens = av_bucket_tens.to(args.device)
            args.ts_bucket = ts_bucket
        print(f"[loader] Availability matrix ready: shape={args.av_tens.shape}")
        
        # 캐시 저장 (CPU 텐서로 저장하여 재사용)
        try:
            cpu_cache = {
                'ts': ts,
                'ts_bucket': ts_bucket if args.has_viewer_bucket else {},
                'max_avail': max_avail,
                'av_tens': av_tens.cpu(),
                'av_bucket_tens': av_bucket_tens.cpu() if av_bucket_tens is not None else None,
            }
            torch.save(cpu_cache, avail_cache_path)
            print(f"[loader] Availability cache saved to {avail_cache_path}")
        except Exception as e:
            print(f"[loader] Warning: failed to save availability cache: {e}")
    else:
        # Dry-run: set minimal availability info to avoid KeyError in negative sampling
        # Create minimal ts dict with all unique streamers for each timestep
        unique_streamers = data_fu['streamer'].unique().tolist()
        args.max_avail = len(unique_streamers)
        args.ts = {}
        args.ts_bucket = {}
        
        # For dry-run, use all streamers as available at all timesteps (simplified)
        # This avoids KeyError in negative sampling while keeping it fast
        for s in range(max_step + 1):
            args.ts[s] = unique_streamers.copy()
            if args.has_viewer_bucket:
                # Use default bucket (0) for all items in dry-run
                args.ts_bucket[s] = [0] * len(unique_streamers)
        base_av = torch.as_tensor(unique_streamers, dtype=torch.long)
        args.av_tens = base_av.repeat(max_step + 1, 1).to(args.device)
        if args.has_viewer_bucket:
            args.av_bucket_tens = torch.zeros(max_step + 1, len(unique_streamers), dtype=torch.long).to(args.device)
        else:
            args.av_bucket_tens = None
        print(f"[dry-run] Created minimal availability info: {len(unique_streamers)} streamers available at all timesteps")
    return data_fu


def get_dataloaders(data_fu, args):
    print("[loader] Creating data loaders...")
    if args.debug:
        mu = 1000
    else:
        mu = int(10e9)
 
    # Check for Parquet cache first (faster), fallback to Pickle
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_key = _sequence_cache_key(args)
    
    # Parquet cache paths (new format)
    cache_tr_pq = cache_dir / f"{dataset_key}_tr.parquet"
    cache_va_pq = cache_dir / f"{dataset_key}_val.parquet"
    cache_te_pq = cache_dir / f"{dataset_key}_te.parquet"
    
    # Pickle cache paths (legacy)
    cache_tr = cache_dir / f"{dataset_key}_tr.p"
    cache_te = cache_dir / f"{dataset_key}_te.p"
    cache_va = cache_dir / f"{dataset_key}_val.p"
    
    # Check which cache format is available
    has_parquet_cache = all([cache_tr_pq.exists(), cache_va_pq.exists(), cache_te_pq.exists()])
    has_pickle_cache = all([cache_tr.exists(), cache_va.exists(), cache_te.exists()])
    
    val_only_mode = getattr(args, 'val_only', False)
    
    if has_parquet_cache and args.caching:
        print("[loader] Loading cached sequences from Parquet (fast)...")
        print("[loader] Loading val sequences...")
        datalist_va = load_sequences_from_parquet(cache_va_pq)
        print(f"[loader] Loaded val: {len(datalist_va)} sequences")
        
        if not val_only_mode:
            print("[loader] Loading train sequences...")
            datalist_tr = load_sequences_from_parquet(cache_tr_pq)
            print(f"[loader] Loaded train: {len(datalist_tr)} sequences")
            print("[loader] Loading test sequences...")
            datalist_te = load_sequences_from_parquet(cache_te_pq)
            print(f"[loader] Loaded test: {len(datalist_te)} sequences")
        else:
            print("[loader] Val-only mode: skipping train/test loading")
            datalist_tr = []
            datalist_te = []
            
    elif has_pickle_cache and args.caching:
        print("[loader] Loading cached sequences from Pickle...")
        print("[loader] Loading val sequences...")
        with open(cache_va, "rb") as f:
            datalist_va = pickle.load(f)
        print(f"[loader] Loaded val: {len(datalist_va)} sequences")
        
        if not val_only_mode:
            print("[loader] Loading train sequences...")
            with open(cache_tr, "rb") as f:
                datalist_tr = pickle.load(f)
            print(f"[loader] Loaded train: {len(datalist_tr)} sequences")
            print("[loader] Loading test sequences...")
            with open(cache_te, "rb") as f:
                datalist_te = pickle.load(f)
            print(f"[loader] Loaded test: {len(datalist_te)} sequences")
        else:
            print("[loader] Val-only mode: skipping train/test loading")
            datalist_tr = []
            datalist_te = []
    elif args.caching:
        print("[loader] Cache not found. Generating sequences (this may take 5-10 minutes)...")
        
        if not val_only_mode:
            print("[loader] Generating train sequences...")
            datalist_tr = get_sequences(data_fu,0,args.pivot_1,args,mu)
        else:
            datalist_tr = []
        
        print("[loader] Generating val sequences...")
        datalist_va = get_sequences(data_fu,args.pivot_1,args.pivot_2,args,mu)
        
        if not val_only_mode:
            print("[loader] Generating test sequences...")
            datalist_te = get_sequences(data_fu,args.pivot_2,args.max_step,args,mu)
        else:
            datalist_te = []

        print("[loader] Saving sequences to cache (Parquet + Pickle)...")
        
        # Save as Parquet (new, fast)
        if not val_only_mode:
            save_sequences_to_parquet(datalist_tr, cache_tr_pq)
        save_sequences_to_parquet(datalist_va, cache_va_pq)
        if not val_only_mode:
            save_sequences_to_parquet(datalist_te, cache_te_pq)
        print("[loader] Parquet cache saved.")
        
        # Also save as Pickle (legacy compatibility)
        if not val_only_mode:
            with open(cache_tr, "wb") as f:
                pickle.dump(datalist_tr, f)
        with open(cache_va, "wb") as f:
            pickle.dump(datalist_va, f)
        if not val_only_mode:
            with open(cache_te, "wb") as f:
                pickle.dump(datalist_te, f)
        print("[loader] Pickle cache saved (legacy).")
    else:
        # caching 비활성화: 즉시 시퀀스 생성
        print("[loader] Caching disabled. Generating sequences...")
        
        if not val_only_mode:
            print("[loader] Generating train sequences...")
            datalist_tr = get_sequences(data_fu,0,args.pivot_1,args,mu)
        else:
            datalist_tr = []
        
        print("[loader] Generating val sequences...")
        datalist_va = get_sequences(data_fu,args.pivot_1,args.pivot_2,args,mu)
        
        if not val_only_mode:
            print("[loader] Generating test sequences...")
            datalist_te = get_sequences(data_fu,args.pivot_2,args.max_step,args,mu)
        else:
            datalist_te = []

    print(f"[loader] Creating DataLoader objects (batch_size={args.batch_size})...")
    
    if not val_only_mode:
        train_loader = DataLoader(datalist_tr,batch_size=args.batch_size,
                                  collate_fn=lambda x: custom_collate(x,args))
        test_loader  = DataLoader(datalist_te,batch_size=args.batch_size,
                                  collate_fn=lambda x: custom_collate(x,args))
    else:
        train_loader = None
        test_loader = None
    
    val_loader = DataLoader(datalist_va,batch_size=args.batch_size,
                            collate_fn=lambda x: custom_collate(x,args))
    
    if val_only_mode:
        print(f"[loader] DataLoaders ready (VAL ONLY): val={len(val_loader)} batches")
    else:
        print(f"[loader] DataLoaders ready: train={len(train_loader)} batches, val={len(val_loader)} batches, test={len(test_loader)} batches")

    return train_loader, val_loader, test_loader


def custom_collate(batch,args):
    # returns a [batch x seq x feats] tensor
    # feats: [padded_positions,positions,inputs_ts,items,users,targets,targets_ts]

    bs = len(batch)
    feat_len = len(batch[0])
    batch_seq = torch.zeros(bs,args.seq_len, feat_len, dtype=torch.long)
    for ib,b in enumerate(batch):
        for ifeat,feat in enumerate(b):
            batch_seq[ib,b[0],ifeat] = feat
    return batch_seq

class SequenceDataset(data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    ## get sequence lengths
    lengths = torch.tensor([ t.shape[0] for t in batch ]).to(device)
    ## padd
    batch = [ torch.Tensor(t).to(device) for t in batch ]
    batch = torch.nn.utils.rnn.pad_sequence(batch)
    ## compute mask
    mask = (batch != 0).to(device)
    return batch, lengths, mask


def get_sequences(_data, _p1, _p2, args, max_u=int(10e9)):
    data_list = []

    _data = _data[_data.stop<_p2].copy()
    
    grouped = _data.groupby('user')
    use_bucket = bool(getattr(args, 'has_viewer_bucket', False))
    use_spike = getattr(args, 'viewer_feat_mode', 'off') == 'spike'
    scale = getattr(args, 'viewer_spike_scale', VIEWER_SPIKE_SCALE)
    feature_indices_set = False

    for user_id, group in tqdm(grouped):
        group = group.sort_values('start')
        group = group.tail(args.seq_len+1)
        if len(group)<2: continue

        group = group.reset_index(drop=True) 
        
        # Get last interaction
        last_el = group.tail(1)
        yt = last_el.start.values[0]
        group.drop(last_el.index,inplace=True)

        # avoid including train in test/validation
        if yt < _p1 or yt >= _p2: continue

        padlen = args.seq_len - len(group)

        # sequence input features
        positions  = torch.LongTensor(group.index.values)
        inputs_ts  = torch.LongTensor(group.start.values)
        items      = torch.LongTensor(group['streamer'].values)
        users      = torch.LongTensor(group.user.values)
        bpad       = torch.LongTensor(group.index.values + padlen)

        # sequence output features
        targets    = torch.LongTensor(items[1:].tolist() + [last_el.streamer.values[0]])
        targets_ts = torch.LongTensor(inputs_ts[1:].tolist() + [last_el.start.values[0]])

        entry = [bpad,positions,inputs_ts,items,users,targets,targets_ts]
        idx_ptr = len(entry)

        if use_bucket and 'viewer_bucket_id' in group.columns:
            bucket_inputs = torch.LongTensor(group.viewer_bucket_id.fillna(0).astype(int).values)
            target_bucket_val = int(last_el.viewer_bucket_id.fillna(0).astype(int).values[0])
            bucket_targets = torch.LongTensor(bucket_inputs[1:].tolist() + [target_bucket_val])
            entry.extend([bucket_inputs, bucket_targets])
            if not feature_indices_set:
                args.bucket_input_idx = idx_ptr
                args.bucket_target_idx = idx_ptr + 1
            idx_ptr += 2

        if use_spike:
            def to_scaled(series):
                vals = pd.Series(series).fillna(0.0).astype(float)
                return torch.LongTensor(np.round(vals.to_numpy() * scale).astype(int))

            # Use viewer trends if available, otherwise use columns from data
            viewer_trends = getattr(args, 'viewer_trends', None)
            id_to_streamer = getattr(args, 'id_to_streamer', None) if viewer_trends else None
            
            if viewer_trends is not None and id_to_streamer is not None:
                # Compute spike features from viewer trends
                def compute_spike_feat(streamer_id, ts):
                    streamer_name = id_to_streamer.get(int(streamer_id), None)
                    if streamer_name:
                        return viewer_trends.compute_spike_features(streamer_name, int(ts))
                    return 0.0, 0.0, 0.0
                
                # For input sequence
                viewer_z_vals = []
                ratio_vals = []
                conf_vals = []
                for sid, ts in zip(group['streamer'].values, group['start'].values):
                    z, r, c = compute_spike_feat(sid, ts)
                    viewer_z_vals.append(z)
                    ratio_vals.append(r)
                    conf_vals.append(c)
                
                viewer_z_inputs = to_scaled(viewer_z_vals)
                ratio_inputs = to_scaled(ratio_vals)
                conf_inputs = to_scaled(conf_vals)
                
                # For target
                target_sid = last_el['streamer'].values[0]
                target_ts = last_el['start'].values[0]
                viewer_z_target_val, ratio_target_val, conf_target_val = compute_spike_feat(target_sid, target_ts)
                viewer_z_target = int(round(viewer_z_target_val * scale))
                ratio_target = int(round(ratio_target_val * scale))
                conf_target = int(round(conf_target_val * scale))
            else:
                # Fallback to columns in data (for backward compatibility)
                viewer_z_inputs = to_scaled(group['viewer_z'] if 'viewer_z' in group.columns else 0.0)
                viewer_z_target_val = float(last_el['viewer_z'].iloc[0]) if 'viewer_z' in last_el.columns else 0.0
                if math.isnan(viewer_z_target_val):
                    viewer_z_target_val = 0.0
                viewer_z_target = int(round(viewer_z_target_val * scale))

                ratio_inputs = to_scaled(group['viewer_ratio_recent'] if 'viewer_ratio_recent' in group.columns else 0.0)
                ratio_target_val = float(last_el['viewer_ratio_recent'].iloc[0]) if 'viewer_ratio_recent' in last_el.columns else 0.0
                if math.isnan(ratio_target_val):
                    ratio_target_val = 0.0
                ratio_target = int(round(ratio_target_val * scale))

                conf_inputs = to_scaled(group['viewer_confidence'] if 'viewer_confidence' in group.columns else 0.0)
                conf_target_val = float(last_el['viewer_confidence'].iloc[0]) if 'viewer_confidence' in last_el.columns else 0.0
                if math.isnan(conf_target_val):
                    conf_target_val = 0.0
                conf_target = int(round(conf_target_val * scale))

            viewer_z_targets = torch.LongTensor(viewer_z_inputs[1:].tolist() + [viewer_z_target])
            ratio_targets = torch.LongTensor(ratio_inputs[1:].tolist() + [ratio_target])
            conf_targets = torch.LongTensor(conf_inputs[1:].tolist() + [conf_target])

            entry.extend([viewer_z_inputs, viewer_z_targets,
                          ratio_inputs, ratio_targets,
                          conf_inputs, conf_targets])
            if not feature_indices_set:
                args.viewer_z_idx = idx_ptr
                args.viewer_z_target_idx = idx_ptr + 1
                args.viewer_ratio_idx = idx_ptr + 2
                args.viewer_ratio_target_idx = idx_ptr + 3
                args.viewer_conf_idx = idx_ptr + 4
                args.viewer_conf_target_idx = idx_ptr + 5
            idx_ptr += 6

        feature_indices_set = True
        data_list.append(entry)

        # stop if user limit is reached
        if len(data_list)>max_u: break

    return SequenceDataset(data_list)
