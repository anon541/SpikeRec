# SpikeRec: Accurate Live-Streaming Recommendation under Item-Level Signal Loss from User Sampling

Code release for **SpikeRec** experiments (paper source is managed separately).

SpikeRec targets **live-streaming recommendation under dynamic availability**: given a user’s interaction history and the set of channels that are currently live, recommend the next channel while remaining responsive to short-lived popularity surges (“spikes”) and robust to **item-level signal loss caused by user sampling**.

## Repository Structure

| Path | Description |
| --- | --- |
| `data/` | Data loading and preprocessing modules. (Raw datasets are not included.) |
| `models/` | PyTorch implementations of SpikeRec and baselines (e.g., SASRec, LiveRec, GRU4Rec, Caser, BERT4Rec). |
| `scripts/` | Training (`train.py`) and evaluation (`eval.py`) entrypoints. |
| `experiments/` | Configuration files (YAML/JSON) for reproducing experiments. |
| `run_smoke_test.sh` | Quick integration test script to verify the environment. |

## Getting Started

### 1. Environment Setup

    conda create -n spikerec python=3.10 -y
    conda activate spikerec
    pip install -r requirements.txt

### 2. Data Preparation

This repo expects an interaction log with the following base schema:

- CSV **without header**, 5 columns: `user`, `stream`, `streamer`, `start`, `stop`
- Place the file under `data/raw/` and point `--dataset` to the path.
- See `data/loader.py` for loader details.

#### Datasets

- **Twitch-100k**: Provided by the LiveRec dataset release (see the LiveRec repository for access).
- **Chzzk**: Collected separately for this work; **will be released publicly after paper acceptance** (e.g., via a public Drive link).

### 3. Smoke Test

    ./run_smoke_test.sh

PASS keyword: `SMOKE_TEST_PASSED`

### 4. Training

    # Train GRU4Rec baseline
    python -m scripts.train --config experiments/method/3_gru4rec_baseline.yaml --model GRU4Rec

    # Train GRU4Rec + SpikeRec
    python -m scripts.train --config experiments/method/3_gru4rec_spike.yaml --model GRU4RecSpike

    # Train Caser baseline
    python -m scripts.train --config experiments/method/4_caser_baseline.yaml --model Caser

    # Train Caser + SpikeRec
    python -m scripts.train --config experiments/method/4_caser_spike.yaml --model CaserSpike

    # Train BERT4Rec baseline
    python -m scripts.train --config experiments/method/5_bert4rec_baseline.yaml --model BERT4Rec

    # Train BERT4Rec + SpikeRec
    python -m scripts.train --config experiments/method/5_bert4rec_spike.yaml --model BERT4RecSpike

- Checkpoints: `checkpoints/`

### 5. Evaluation

    python -m scripts.eval \
      --config experiments/method/3_gru4rec_baseline.yaml \
      --model GRU4Rec \
      --checkpoint checkpoints/gru4rec_baseline.pt \
      --eval_split test

## Reproducibility

See `REPRODUCIBILITY.md` for a minimal runnable path (no private data required) and notes on full reproduction once datasets are available.

## Citation

    @inproceedings{spikerec_kdd26,
      title={SpikeRec: Accurate Live-Streaming Recommendation under Item-Level Signal Loss from User Sampling},
      author={Anonymous},
      booktitle={Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '26)},
      year={2026}
    }
