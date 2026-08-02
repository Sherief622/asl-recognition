"""Stage 04 — cross-dataset evaluation: the collapse.

What this stage shows: the models that scored (near-)perfectly on the leaky
split evaluated zero-shot on ASL-HG — 10 different signers, natural indoor
and outdoor backgrounds. Evaluation is restricted to the A-Z intersection
and reported under two protocols: masked (argmax over the 26 letter logits)
and unmasked (argmax over all 28 outputs; Space/Nothing predictions count
as errors). The gap between 100% and these numbers is the leak, quantified.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from _common import LETTER_CLASSES, ORIGINAL_CLASSES, base_parser, smoke_cap, start
from aslrec.config import get_device, save_results
from aslrec.data.datasets import ManifestDataset, load_manifest, make_loader
from aslrec.data.transforms import build_eval_transforms
from aslrec.engine.evaluate import cross_dataset_eval
from aslrec.models import build_model

CHECKPOINTS = {
    "slrvgg8": "slrvgg8_best.pth",
    "asl_cnn": "asl_cnn_best.pth",
}


def letters_dataset(cfg, variant: str):
    manifest_path = cfg["paths"]["manifests"] / f"aslhg_{variant}.csv"
    if not manifest_path.exists():
        return None
    frame = load_manifest(manifest_path)
    frame = frame[frame["label"].str.upper().isin(LETTER_CLASSES)].copy()
    frame["label"] = frame["label"].str.upper()
    classes = sorted(frame["label"].unique())
    return ManifestDataset(frame, classes, build_eval_transforms()), classes


def plot_per_class(per_class: dict, title: str, out_path):
    names = sorted(per_class)
    vals = [per_class[n] if per_class[n] is not None else 0 for n in names]
    plt.figure(figsize=(11, 4))
    plt.bar(names, vals, edgecolor="black")
    plt.ylim(0, 1)
    plt.title(title)
    plt.ylabel("masked accuracy")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = base_parser(__doc__)
    args, cfg = start(parser, "Stage 04: zero-shot cross-dataset evaluation on ASL-HG")
    device = get_device(args.device)

    payload = {"smoke": args.smoke, "shared_classes": "A-Z intersection"}
    any_variant = False
    for variant in ("raw", "processed"):
        built = letters_dataset(cfg, variant)
        if built is None:
            continue
        any_variant = True
        dataset, target_classes = built
        loader = make_loader(smoke_cap(dataset, args), device=device)
        payload[variant] = {}
        for name, ckpt_file in CHECKPOINTS.items():
            model = build_model(name, num_classes=len(ORIGINAL_CLASSES)).to(device)
            model.load_state_dict(
                torch.load(
                    cfg["paths"]["historical_checkpoints"] / ckpt_file,
                    map_location=device, weights_only=True,
                ),
                strict=True,
            )
            stats = cross_dataset_eval(
                model, loader, device, ORIGINAL_CLASSES, target_classes
            )
            payload[variant][name] = stats
            print(
                f"[{variant}] {name}: masked {stats['masked_accuracy']:.4f} | "
                f"unmasked {stats['unmasked_accuracy']:.4f} "
                f"({stats['n_samples']} images)"
            )
            plot_per_class(
                stats["per_class_masked_accuracy"],
                f"{name} zero-shot on ASL-HG ({variant}) — per-class masked accuracy",
                cfg["paths"]["figures"] / f"04_{name}_{variant}_per_class.png",
            )

    if not any_variant:
        raise SystemExit("No ASL-HG manifests found — run scripts/00_prepare_data.py first.")

    out = save_results("04_cross_dataset_eval", payload, cfg["paths"]["results"])
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
