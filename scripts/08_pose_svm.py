"""Stage 08 — pose-estimation + SVM baseline (previously never run).

What this stage shows: MediaPipe Hands 21-landmark (x,y) features feeding a
linear SVM, trained and evaluated on the original dataset AND on ASL-HG's
signer split, plus a cross-dataset transfer (original-trained SVM applied
to ASL-HG letters). Landmark features factor out appearance (skin tone,
background, lighting), so this model family is the natural control for the
CNNs' cross-dataset collapse. Detection-failure rates are reported rather
than silently dropped ('Nothing'/'Space' images have no hand to find).
"""

import numpy as np
from _common import LETTER_CLASSES, base_parser, start
from aslrec.config import save_results
from aslrec.data.datasets import load_manifest
from aslrec.data.splits import scan_class_folders
from aslrec.engine.evaluate import summarize
from aslrec.pose.extract import extract_features
from aslrec.pose.svm import plot_c_selection, train_svm

SMOKE_PER_CLASS = 8


def original_frames(cfg, smoke: bool):
    frames = {}
    for split, folder in (("train", "Train"), ("val", "Val"), ("test", "Test")):
        frame = scan_class_folders(cfg["paths"]["asl_original"] / folder)
        if smoke:
            frame = frame.groupby("label").head(SMOKE_PER_CLASS).reset_index(drop=True)
        frames[split] = frame
    return frames


def aslhg_frames(cfg, smoke: bool):
    for variant in ("raw", "processed"):
        path = cfg["paths"]["manifests"] / f"aslhg_{variant}.csv"
        if path.exists():
            manifest = load_manifest(path)
            manifest["label"] = manifest["label"].astype(str).str.upper()
            frames = {}
            for split in ("train", "val", "test"):
                frame = manifest[manifest["split"] == split].reset_index(drop=True)
                if smoke:
                    frame = frame.groupby("label").head(SMOKE_PER_CLASS).reset_index(drop=True)
                frames[split] = frame
            return variant, frames
    return None, None


def run_experiment(tag, frames, cfg, smoke) -> tuple[dict, object, dict]:
    cache_dir = cfg["paths"]["aslhg"].parent / "keypoint_cache"
    suffix = "_smoke" if smoke else ""
    feats = {
        split: extract_features(frame, cache_dir / f"{tag}_{split}{suffix}.npz")
        for split, frame in frames.items()
    }
    for split, f in feats.items():
        print(f"  [{tag}] {split}: {len(f['y'])} ok, "
              f"{f['n_failed']} no-hand ({f['failure_rate']:.1%})")

    model, best_c, val_accs = train_svm(
        feats["train"]["X"], feats["train"]["y"],
        feats["val"]["X"], feats["val"]["y"],
    )
    plot_c_selection(
        val_accs,
        cfg["paths"]["figures"] / f"08_svm_c_selection_{tag}.png",
        f"SVM C selection ({tag})",
    )
    classes = sorted(np.unique(feats["train"]["y"]).tolist())
    preds = model.predict(feats["test"]["X"])
    label_idx = {c: i for i, c in enumerate(classes)}
    stats = summarize(
        np.array([label_idx[p] for p in preds]),
        np.array([label_idx[t] for t in feats["test"]["y"]]),
        classes,
    )
    print(f"  [{tag}] test accuracy {stats['accuracy']:.4f} (C={best_c})")
    result = {
        "best_C": best_c,
        "val_accuracies": val_accs,
        "test": stats,
        "detection_failure_rates": {
            split: f["failure_rate"] for split, f in feats.items()
        },
    }
    return result, model, feats


def main():
    parser = base_parser(__doc__)
    args, cfg = start(parser, "Stage 08: pose-estimation + linear SVM baseline")

    payload = {"smoke": args.smoke, "features": "21 MediaPipe landmarks x (x,y) = 42"}

    print("\nOriginal dataset:")
    orig_result, orig_model, _ = run_experiment(
        "original", original_frames(cfg, args.smoke), cfg, args.smoke
    )
    payload["original"] = orig_result

    variant, frames = aslhg_frames(cfg, args.smoke)
    if frames is not None:
        print(f"\nASL-HG ({variant}):")
        aslhg_result, _, aslhg_feats = run_experiment(
            f"aslhg_{variant}", frames, cfg, args.smoke
        )
        payload[f"aslhg_{variant}"] = aslhg_result

        # cross-dataset control: the original-trained SVM on ASL-HG letters
        X, y = aslhg_feats["test"]["X"], aslhg_feats["test"]["y"]
        keep = [i for i, label in enumerate(y) if label in LETTER_CLASSES]
        if keep:
            preds = orig_model.predict(X[keep])
            acc = float(np.mean([p == y[k] for p, k in zip(preds, keep)]))
            payload["cross_dataset_original_svm_on_aslhg_letters"] = {
                "accuracy": acc, "n_samples": len(keep),
            }
            print(f"\ncross-dataset SVM (original -> ASL-HG letters): {acc:.4f}")
    else:
        print("no ASL-HG manifest found — run stage 00 first for the full comparison")

    out = save_results("08_pose_svm", payload, cfg["paths"]["results"])
    print(f"\nresults -> {out}")


if __name__ == "__main__":
    main()
