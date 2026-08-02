"""Stage 06 — transfer learning to our self-collected digits dataset.

What this stage shows: the 990-image ASL digits dataset we photographed
ourselves (3 signers, varied backgrounds/devices) used to test transfer
learning — with the protocol fixed. The notebook early-stopped on the test
set; here model selection uses a real validation split carved from the
training half (stage 00), and Digits_Test is evaluated exactly once per
run. Both backbones are compared: the original leaky-trained SLRVGG8 and
the honest ASL-HG-trained one.
"""

import torch
import torch.nn as nn
from _common import ORIGINAL_CLASSES, base_parser, smoke_cap, start
from aslrec.config import get_device, save_results
from aslrec.data.datasets import ManifestDataset, load_manifest, make_loader
from aslrec.data.transforms import build_eval_transforms, build_train_transforms
from aslrec.engine.evaluate import (
    collect_predictions,
    plot_confusion,
    plot_history,
    summarize,
)
from aslrec.engine.train import train_model
from aslrec.models import SLRVGG8


def load_backbone(cfg, which: str, device):
    """Return an SLRVGG8 with pretrained features and the source class count."""
    if which == "baseline":
        ckpt = cfg["paths"]["historical_checkpoints"] / "slrvgg8_best.pth"
        n_src = len(ORIGINAL_CLASSES)
    else:  # aslhg
        candidates = [
            cfg["paths"]["checkpoints"] / f"slrvgg8_aslhg_{v}_best.pth"
            for v in ("raw", "processed")
        ]
        candidates = [c for c in candidates if c.exists()]
        if not candidates:
            return None, None
        ckpt = candidates[0]
        # infer class count from the checkpoint's final layer
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        n_src = state["classifier.5.weight"].shape[0]
    model = SLRVGG8(num_classes=n_src)
    model.load_state_dict(
        torch.load(ckpt, map_location="cpu", weights_only=True), strict=True
    )
    return model.to(device), str(ckpt)


def main():
    parser = base_parser(__doc__)
    parser.add_argument(
        "--backbones", nargs="+", default=["baseline", "aslhg"],
        choices=["baseline", "aslhg"],
    )
    parser.add_argument("--epochs", type=int, default=20)
    args, cfg = start(parser, "Stage 06: digits transfer learning (protocol fixed)")
    device = get_device(args.device)

    manifest = load_manifest(cfg["paths"]["manifests"] / "digits.csv")
    manifest["label"] = manifest["label"].astype(str)
    classes = sorted(manifest["label"].unique())
    print(f"digits: {len(manifest)} images, {len(classes)} classes "
          f"{dict(manifest['split'].value_counts())}")

    def ds(split, transform):
        return ManifestDataset(manifest[manifest["split"] == split], classes, transform)

    train_loader = make_loader(
        smoke_cap(ds("train", build_train_transforms(legacy=False)), args),
        shuffle=True, device=device,
    )
    val_loader = make_loader(smoke_cap(ds("val", build_eval_transforms()), args), device=device)
    test_loader = make_loader(smoke_cap(ds("test", build_eval_transforms()), args), device=device)

    payload = {
        "smoke": args.smoke,
        "protocol": "model selection on val manifest; test evaluated once",
        "invalid_notebook_number": {
            "accuracy": 0.9114,
            "note": "the coursework's 91.14% used the test set for early "
                    "stopping and model selection — recorded here for the "
                    "report, not comparable to the corrected numbers",
        },
    }

    for which in args.backbones:
        backbone, ckpt_used = load_backbone(cfg, which, device)
        if backbone is None:
            print(f"skipping backbone '{which}' — no ASL-HG checkpoint yet (run stage 05)")
            continue
        print(f"\n--- backbone: {which} ({ckpt_used}) ---")

        # replace the head, freeze the feature extractor (notebook recipe)
        in_features = backbone.classifier[2].in_features
        backbone.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, len(classes)),
        ).to(device)
        for p in backbone.features.parameters():
            p.requires_grad = False

        ckpt = cfg["paths"]["checkpoints"] / f"slrvgg8_digits_{which}_best.pth"
        result = train_model(
            backbone, train_loader, val_loader, device, ckpt,
            epochs=1 if args.smoke else args.epochs,
            trainable_params=backbone.classifier.parameters(),
        )
        plot_history(
            result.to_dict()["history"],
            f"digits transfer ({which} backbone)",
            cfg["paths"]["figures"] / f"06_digits_{which}_history.png",
        )

        preds, labels, _ = collect_predictions(backbone, test_loader, device)
        stats = summarize(preds, labels, classes)
        print(f"digits test accuracy ({which} backbone): {stats['accuracy']:.4f}")
        plot_confusion(
            preds, labels, classes,
            f"Digits transfer — {which} backbone",
            cfg["paths"]["figures"] / f"06_digits_{which}_confusion.png",
        )
        payload[which] = {"checkpoint": ckpt_used, "train": result.to_dict(), "test": stats}

    out = save_results("06_transfer_digits", payload, cfg["paths"]["results"])
    print(f"\nresults -> {out}")


if __name__ == "__main__":
    main()
