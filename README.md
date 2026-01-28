# Live Streaming Recommendation with Availability Constraints

Code release for the LiveRec/SpikeRec-style experiments (paper source is managed separately).

We propose a new recommendation framework that explicitly models **channel availability** and leverages **concurrent viewer counts** to improve next-channel prediction in live streaming platforms (e.g., Twitch).

## Repository Structure

| Path | Description |
| --- | --- |
| `data/` | Data loading and preprocessing modules. (Note: Raw datasets are not included due to size/privacy) |
| `models/` | PyTorch implementations of our proposed models and baselines (SASRec, LiveRec). |
| `scripts/` | Training (`train.py`) and evaluation (`eval.py`) entrypoints. |
| `experiments/` | Configuration files (YAML/JSON) for reproducing experiments. |
| `run_smoke_test.sh` | A quick integration test script to verify the environment. |

## Getting Started

### 1. Environment Setup

We recommend using Conda to manage the environment.

   ```bash
conda create -n liverec python=3.10 -y
conda activate liverec
pip install -r requirements.txt
```

### 2. Data Preparation

Due to privacy and policy constraints, the raw Twitch dataset used in the paper cannot be shared directly. 
The anonymized dataset release is planned separately. This repo expects an interaction log with the following base columns:

- CSV without header, 5 columns: `user`, `stream`, `streamer`, `start`, `stop`
- Place your interaction data under `data/raw/` and point `--dataset` to the file path.

- Place your interaction data in `data/raw/`.
- See `data/loader.py` for details on the expected schema.

### 3. Running a Smoke Test

To verify that the environment is set up correctly and the code runs without errors, execute the smoke test script:

   ```bash
./run_smoke_test.sh
```

This script runs a minimal 1-epoch training loop on a small subset to ensure all modules are wired correctly.
PASS keyword: `SMOKE_TEST_PASSED`

### 4. Training

To train a model, use `scripts/train.py` with a configuration file:

   ```bash
# Train GRU4Rec baseline
   python -m scripts.train --config experiments/method/3_gru4rec_baseline.yaml --model GRU4Rec

# Train GRU4Rec spike (trend-only)
python -m scripts.train --config experiments/method/3_gru4rec_spike.yaml --model GRU4RecSpike

# Train Caser baseline
python -m scripts.train --config experiments/method/4_caser_baseline.yaml --model Caser

# Train Caser spike (trend-only)
python -m scripts.train --config experiments/method/4_caser_spike.yaml --model CaserSpike

# Train BERT4Rec baseline
python -m scripts.train --config experiments/method/5_bert4rec_baseline.yaml --model BERT4Rec

# Train BERT4Rec spike (trend-only)
python -m scripts.train --config experiments/method/5_bert4rec_spike.yaml --model BERT4RecSpike

```

- **Checkpoints**: Saved to `checkpoints/` by default.
- **Logs**: Printed to stdout by default. (Optional integrations may be added.)

### 5. Evaluation

To evaluate a trained checkpoint:

   ```bash
   python -m scripts.eval \
     --config experiments/method/3_gru4rec_baseline.yaml \
     --model GRU4Rec \
     --checkpoint checkpoints/gru4rec_baseline.pt \
     --eval_split test

   python -m scripts.eval \
     --config experiments/method/4_caser_baseline.yaml \
     --model Caser \
     --checkpoint checkpoints/caser_baseline.pt \
     --eval_split test

   python -m scripts.eval \
     --config experiments/method/5_bert4rec_baseline.yaml \
     --model BERT4Rec \
     --checkpoint checkpoints/bert4rec_baseline.pt \
     --eval_split test

   ```

## Reproducibility

See `REPRODUCIBILITY.md` for a minimal runnable path (no private data required) and notes on full reproduction once the anonymized dataset is available.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{your_paper_2025,
  title={...},
  author={...},
  booktitle={...},
  year={2025}
}
```
