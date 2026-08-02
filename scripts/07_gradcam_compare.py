"""Stage 07 — what the models look at.

What this stage shows: Grad-CAM attention grids for both CNN architectures
in three settings: the baseline (leaky-trained) models on original test
images, the same models on ASL-HG images, and the honest ASL-HG-trained
models on ASL-HG images. Comparing rows reveals whether a model attends to
hand geometry or to signer/background shortcuts.
"""

import random

import torch
from _common import LETTER_CLASSES, ORIGINAL_CLASSES, base_parser, start
from aslrec.config import get_device, save_results
from aslrec.data.datasets import load_manifest
from aslrec.data.splits import scan_class_folders
from aslrec.data.transforms import build_eval_transforms
from aslrec.gradcam import cam_grid
from aslrec.models import build_model

N_IMAGES = 6


def sample_original(cfg, rng) -> list[str]:
    frame = scan_class_folders(cfg["paths"]["asl_original"] / "Test")
    frame = frame[frame["label"].isin(LETTER_CLASSES)]
    picks = frame.groupby("label").head(1)["path"].tolist()
    return rng.sample(picks, min(N_IMAGES, len(picks)))


def sample_aslhg(cfg, rng) -> list[str]:
    for variant in ("raw", "processed"):
        manifest = cfg["paths"]["manifests"] / f"aslhg_{variant}.csv"
        if manifest.exists():
            frame = load_manifest(manifest, split="test")
            frame = frame[frame["label"].astype(str).str.upper().isin(LETTER_CLASSES)]
            picks = frame.groupby("label").head(1)["path"].tolist()
            return rng.sample(picks, min(N_IMAGES, len(picks)))
    return []


def load_ckpt(cfg, model_name, ckpt_path, n_classes, device):
    model = build_model(model_name, num_classes=n_classes).to(device)
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True), strict=True
    )
    return model


def main():
    parser = base_parser(__doc__)
    args, cfg = start(parser, "Stage 07: Grad-CAM comparison")
    device = get_device(args.device)
    rng = random.Random(args.seed)
    preprocess = build_eval_transforms()
    figures = cfg["paths"]["figures"]

    original_imgs = sample_original(cfg, rng)
    aslhg_imgs = sample_aslhg(cfg, rng)

    made = []
    for name in ("slrvgg8", "asl_cnn"):
        baseline = load_ckpt(
            cfg, name,
            cfg["paths"]["historical_checkpoints"] / f"{name}_best.pth",
            len(ORIGINAL_CLASSES), device,
        )
        made.append(cam_grid(
            baseline, original_imgs, preprocess, ORIGINAL_CLASSES, device,
            figures / f"07_{name}_baseline_on_original.png",
            f"{name} (baseline) on original test images",
        ))
        if aslhg_imgs:
            made.append(cam_grid(
                baseline, aslhg_imgs, preprocess, ORIGINAL_CLASSES, device,
                figures / f"07_{name}_baseline_on_aslhg.png",
                f"{name} (baseline) on ASL-HG images",
            ))
        honest_ckpts = [
            cfg["paths"]["checkpoints"] / f"{name}_aslhg_{v}_best.pth"
            for v in ("raw", "processed")
        ]
        for ckpt in [c for c in honest_ckpts if c.exists()]:
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            head_key = "classifier.5.weight" if name == "slrvgg8" else "fc2.weight"
            n = state[head_key].shape[0]
            honest = load_ckpt(cfg, name, ckpt, n, device)
            classes = sorted(
                set(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"))
            )[:n] if n == 36 else LETTER_CLASSES
            if aslhg_imgs:
                made.append(cam_grid(
                    honest, aslhg_imgs, preprocess, classes, device,
                    figures / f"07_{name}_honest_on_aslhg.png",
                    f"{name} (ASL-HG trained) on ASL-HG images",
                ))

    payload = {"figures": [str(p) for p in made], "n_images_per_grid": N_IMAGES}
    out = save_results("07_gradcam", payload, cfg["paths"]["results"])
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
