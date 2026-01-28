#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

mkdir -p data/raw

# Generate a tiny synthetic interaction log.
# Format: CSV without header, columns: user, stream, streamer, start, stop
python - <<'PY'
import os
import random

random.seed(42)

out_path = os.path.join("data", "raw", "smoke.csv")

num_users = 5
num_streamers = 12

rows = []

# Ensure timestamps span train/val/test pivots used by loader: pivot_1=max_step-500, pivot_2=max_step-250.
# So we want max_step >= 1000.
max_step = 1000

stream_id = 0
for u in range(1, num_users + 1):
    t = 0
    for _ in range(60):
        streamer = f"s{random.randint(1, num_streamers)}"
        start = t
        dur = random.randint(1, 3)
        stop = min(max_step, start + dur)
        # 'stream' can be any identifier; keep it simple.
        rows.append((f"u{u}", f"st{stream_id}", streamer, start, stop))
        stream_id += 1
        t += random.randint(5, 25)
        if t > max_step:
            break

with open(out_path, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(",".join(map(str, r)) + "\n")

print(f"[smoke] wrote {len(rows)} rows -> {out_path}")
PY

python -m scripts.train \
  --config experiments/method/1_sasrec_baseline.yaml \
  --model MinimalSpikeHeadMLP \
  --dataset data/raw/smoke.csv \
  --device cpu \
  --dry-run

echo "SMOKE_TEST_PASSED"
