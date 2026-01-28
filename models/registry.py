import importlib
import inspect
import os
from pathlib import Path
from typing import Dict, Type

import torch.nn as nn

_BASELINE_MODULES = [
    "models.baselines.liverec",
    "models.baselines.gru4rec",
    "models.baselines.gru4rec_spike",
    "models.baselines.caser",
    "models.baselines.caser_spike",
    "models.baselines.bert4rec",
    "models.baselines.bert4rec_spike",
    "models.experimental.minimal_spike_head",
]


def _discover_model_classes() -> Dict[str, Type[nn.Module]]:
    """Return all nn.Module subclasses defined inside baseline modules."""
    classes: Dict[str, Type[nn.Module]] = {}
    for module_name in _BASELINE_MODULES:
        module = importlib.import_module(module_name)
        for name, cls in inspect.getmembers(module, inspect.isclass):
            if not issubclass(cls, nn.Module):
                continue
            if cls.__module__ != module.__name__:
                continue
            classes[name] = cls
    return classes


def _ensure_unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    base = path.with_suffix("")
    ext = path.suffix
    for idx in range(1, 1000):
        candidate = Path(f"{base}-{idx}{ext}")
        if not candidate.exists():
            return candidate
    from datetime import datetime
    stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return Path(f"{base}-{stamp}{ext}")


def get_model_type(args):
    """
    Resolve (checkpoint_path, model_cls) based on args.model while keeping
    backward compatibility with the previous models.py helper.
    """
    candidate_classes = _discover_model_classes()
    model_name = getattr(args, "model", None)
    if not model_name:
        raise ValueError("args.model must be provided")
    if model_name not in candidate_classes:
        available = ", ".join(sorted(candidate_classes))
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
    checkpoint_name = str(getattr(args, "mto", model_name)) + ".pt"
    ckpt_dir = Path(getattr(args, "model_path", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    target_path = _ensure_unique_path(ckpt_dir / checkpoint_name)
    return str(target_path), candidate_classes[model_name]
