# models/

PyTorch modules for Twitch recommendation experiments.
- `baselines/`: SASRec and other reference architectures.
- `concurrent/`: future concurrent-viewer-aware families (placeholder).
Each submodule should expose a factory function consumed by `scripts/train.py`.
