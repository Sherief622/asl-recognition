"""Path resolution, class lists, seeding, and shared runtime helpers."""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

# Repo root is two levels above this file (src/aslrec/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]

# The 28 classes of the original Kapil Londhe dataset, in ImageFolder
# (alphabetical) order — this ordering is what the historical checkpoints
# were trained against and must not change.
ORIGINAL_CLASSES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
    "Nothing", "O", "P", "Q", "R", "S", "Space", "T", "U", "V", "W", "X",
    "Y", "Z",
]
LETTER_CLASSES = [c for c in ORIGINAL_CLASSES if c not in ("Nothing", "Space")]


def load_config(config_path: Path | None = None) -> dict:
    """Load config.yaml and resolve all paths relative to the repo root."""
    path = config_path or REPO_ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {
        key: (REPO_ROOT / value).resolve() for key, value in cfg["paths"].items()
    }
    return cfg


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def require_cuda() -> None:
    """Fail fast with an actionable message when CUDA is expected but absent."""
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available. An RTX 5070 (Blackwell, sm_120) needs "
            "torch >= 2.7 built for cu128 — install with:\n"
            "  uv pip install -r requirements.txt\n"
            "or pass --device cpu to run (much slower)."
        )


def save_results(name: str, payload: dict, results_dir: Path) -> Path:
    """Write a stage's metrics JSON — the single source of truth for README numbers."""
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"{name}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return out


def worker_count() -> int:
    # DataLoader workers; leave headroom for the main process.
    return max(2, min(8, (os.cpu_count() or 4) - 2))
