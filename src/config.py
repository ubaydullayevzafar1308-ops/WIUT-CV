"""Project paths, parameters, seeds and device selection."""
from __future__ import annotations

import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# The evaluation machine has no internet: stop Ultralytics from probing DNS and sending events.
os.environ.setdefault("YOLO_OFFLINE", "1")

ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH = ROOT / "configs" / "params.yaml"


@lru_cache(maxsize=None)
def load_params(path: Path = PARAMS_PATH) -> dict[str, Any]:
    """Load the parameter file once per process."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Resolve a repository-relative path to an absolute one."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def set_seeds(seed: int) -> None:
    """Fix every random number generator the pipeline touches."""
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device() -> str:
    """Pick the fastest available torch device: cuda, then mps, then cpu."""
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
