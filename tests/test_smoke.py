"""Smoke tests. Run directly (no pytest needed):

    .venv/Scripts/python tests/test_smoke.py

The checkpoint-load test is the port-fidelity gate: the historical weights
trained by the original notebook must load into the ported architectures
with strict=True, proving the classes are byte-for-byte compatible.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd
import torch

from aslrec.config import ORIGINAL_CLASSES, load_config
from aslrec.data.splits import check_split_invariants, stratified_split
from aslrec.data.transforms import build_eval_transforms, build_train_transforms
from aslrec.models import ASL_CNN, SLRVGG8


def test_forward_shapes():
    for cls, n in ((SLRVGG8, 28), (ASL_CNN, 28), (SLRVGG8, 36), (ASL_CNN, 11)):
        model = cls(num_classes=n)
        out = model(torch.randn(2, 3, 224, 224))
        assert out.shape == (2, n), f"{cls.__name__}: {out.shape}"
    print("ok: forward shapes")


def test_checkpoint_loads():
    cfg = load_config()
    ckpt_dir = cfg["paths"]["historical_checkpoints"]
    gates = [
        ("slrvgg8_best.pth", SLRVGG8, 28),
        ("asl_cnn_best.pth", ASL_CNN, 28),
    ]
    for name, cls, n in gates:
        path = ckpt_dir / name
        assert path.exists(), f"missing historical checkpoint: {path}"
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(num_classes=n)
        model.load_state_dict(state, strict=True)
        print(f"ok: {name} loads strict=True into {cls.__name__}({n})")


def test_transforms():
    from PIL import Image

    img = Image.new("RGB", (400, 400), color=(120, 90, 60))
    for tf in (
        build_train_transforms(legacy=True),
        build_train_transforms(legacy=False),
        build_eval_transforms(),
    ):
        out = tf(img)
        assert out.shape == (3, 224, 224), out.shape
        assert 0.0 <= out.min() and out.max() <= 1.0
    print("ok: transforms")


def test_split_invariants():
    frame = pd.DataFrame({
        "path": [f"img_{i}.jpg" for i in range(300)],
        "label": ["A"] * 150 + ["B"] * 150,
        "signer": None,
    })
    split = stratified_split(frame, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=42)
    report = check_split_invariants(split)
    assert report["n_total"] == 300
    assert set(report["by_split"]) == {"train", "val", "test"}
    assert sum(report["by_split"].values()) == 300
    # per-class stratification
    for label in ("A", "B"):
        sub = split[split["label"] == label]
        assert abs(len(sub[sub["split"] == "train"]) - 105) <= 1
    print("ok: split invariants")


def test_class_order_matches_imagefolder():
    """ORIGINAL_CLASSES must equal torchvision ImageFolder's alphabetical
    ordering of the on-disk class dirs — the checkpoint's output indexing."""
    cfg = load_config()
    test_dir = cfg["paths"]["asl_original"] / "Test"
    if not test_dir.exists():
        print("skip: original dataset not present")
        return
    on_disk = sorted(d.name for d in test_dir.iterdir() if d.is_dir())
    assert on_disk == ORIGINAL_CLASSES, f"class order mismatch: {on_disk}"
    print("ok: class order matches ImageFolder")


if __name__ == "__main__":
    test_forward_shapes()
    test_checkpoint_loads()
    test_transforms()
    test_split_invariants()
    test_class_order_matches_imagefolder()
    print("\nAll smoke tests passed.")
