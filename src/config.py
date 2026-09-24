"""Project paths, parameters, seeds and device selection."""
from __future__ import annotations

import copy
import logging
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
DEVICE_ENV = "WIUT_DEVICE"   # forces the device, e.g. WIUT_DEVICE=cpu to test the CPU fallback on a GPU machine

log = logging.getLogger(__name__)


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
    """Pick the fastest available torch device: cuda, then mps, then cpu (``WIUT_DEVICE`` overrides)."""
    import torch

    if forced := os.environ.get(DEVICE_ENV):
        return forced
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_description(device: str) -> str:
    """Human-readable device, torch and CUDA versions for the log."""
    import torch

    cuda = f"CUDA {torch.version.cuda}" if torch.version.cuda else "no CUDA build"
    if device.startswith("cuda"):
        name = torch.cuda.get_device_name(device)
        return f"{device} ({name}), torch {torch.__version__}, {cuda}, cuDNN {torch.backends.cudnn.version()}"
    available = "CUDA available" if torch.cuda.is_available() else "CUDA not available"
    return f"{device}, torch {torch.__version__}, {cuda}, {available}"


def device_kind(device: str) -> str:
    return device.split(":")[0]


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        out[key] = deep_merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


@lru_cache(maxsize=None)
def _runtime_params(device: str) -> dict[str, Any]:
    params = load_params()
    profile = params["device_profiles"].get(device_kind(device), {})
    log.info("device: %s; profile: %s", device_description(device), device_kind(device) if profile else "default")
    merged = deep_merge(params, profile)
    merged["device"] = device
    return merged


def runtime_params() -> dict[str, Any]:
    """Parameters for this machine: params.yaml with the device profile applied (logged once)."""
    return _runtime_params(select_device())
