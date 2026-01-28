# Reproducibility

This repository is a code-only release. The paper source and private/raw datasets are managed separately.

## What You Can Reproduce Today

1) Environment setup and import sanity
2) End-to-end training pipeline wiring via a tiny synthetic dataset

Run:

```bash
./run_smoke_test.sh
```

Expected output contains: `SMOKE_TEST_PASSED`

## Full Reproduction (After Anonymized Data Release)

Once the anonymized dataset is available:

1) Place the interaction log under `data/raw/`.
2) Point the training/eval commands to the dataset path via `--dataset` or `experiments/*.yaml`.

Base interaction schema (CSV without header):

- `user` (anonymized user id)
- `stream` (session id or stream instance id)
- `streamer` (anonymized streamer/channel id)
- `start` (integer timestep)
- `stop` (integer timestep)

Example (baseline training):

```bash
python -m scripts.train --config experiments/method/1_sasrec_baseline.yaml --model MinimalSpikeHeadMLP
```

## Notes

- Checkpoints are written to `checkpoints/` by default (gitignored).
- Intermediate caches are written to `cache/` by default (gitignored).
