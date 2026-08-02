"""Stage 05 — the honest models.

What this stage shows: both CNN architectures trained from scratch on
ASL-HG with the corrected augmentation stack (no horizontal flip) and the
manifest split (signer-grouped when signer ids were recoverable). The test
number here is the honest counterpart to the baseline's 100%. Each model is
also back-evaluated zero-shot on the original dataset's letters to measure
generalization in the reverse direction.
"""

import torch
from _common import LETTER_CLASSES, ORIGINAL_CLASSES, base_parser, smoke_cap, start
from aslrec.config import get_device, save_results
from aslrec.data.datasets import (
    ManifestDataset,
    image_folder,
    load_manifest,
    make_loader,
)
from aslrec.data.transforms import build_eval_transforms, build_train_transforms
from aslrec.engine.evaluate import (
    collect_predictions,
    cross_dataset_eval,
    plot_confusion,
    plot_history,
    summarize,
)
from aslrec.engine.train import train_model
from aslrec.models import build_model


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--variant", choices=["raw", "processed"], default="raw")
    parser.add_argument("--models", nargs="+", default=["slrvgg8", "asl_cnn"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--out-suffix", default="",
        help="suffix for results/checkpoint names (e.g. diagnostic runs)",
    )
    args, cfg = start(parser, "Stage 05: honest training on ASL-HG")
    device = get_device(args.device)

    manifest_path = cfg["paths"]["manifests"] / f"aslhg_{args.variant}.csv"
    if not manifest_path.exists():
        raise SystemExit(f"missing manifest {manifest_path} — run stage 00 first")
    frame = load_manifest(manifest_path)
    frame["label"] = frame["label"].astype(str).str.upper()
    classes = sorted(frame["label"].unique())
    print(f"{len(frame)} images, {len(classes)} classes")

    def ds(split, transform):
        return ManifestDataset(
            frame[frame["split"] == split], classes, transform
        )

    train_ds = smoke_cap(ds("train", build_train_transforms(legacy=False)), args, 400)
    val_ds = smoke_cap(ds("val", build_eval_transforms()), args)
    test_ds = smoke_cap(ds("test", build_eval_transforms()), args)

    payload = {"smoke": args.smoke, "variant": args.variant, "classes": classes,
               "split_strategy": "see results/00_data_inventory.json"}

    for name in args.models:
        print(f"\n--- training {name} ---")
        model = build_model(name, num_classes=len(classes)).to(device)
        ckpt = cfg["paths"]["checkpoints"] / f"{name}_aslhg_{args.variant}{args.out_suffix}_best.pth"
        result = train_model(
            model,
            make_loader(train_ds, shuffle=True, device=device),
            make_loader(val_ds, device=device),
            device,
            ckpt,
            epochs=1 if args.smoke else args.epochs,
            lr=args.lr,
            patience=args.patience,
        )
        plot_history(
            result.to_dict()["history"],
            f"{name} on ASL-HG ({args.variant})",
            cfg["paths"]["figures"] / f"05_{name}_{args.variant}_history.png",
        )

        preds, labels, _ = collect_predictions(
            model, make_loader(test_ds, device=device), device
        )
        stats = summarize(preds, labels, classes)
        print(f"{name}: ASL-HG test accuracy {stats['accuracy']:.4f}")
        plot_confusion(
            preds, labels, classes,
            f"{name} on ASL-HG test ({args.variant})",
            cfg["paths"]["figures"] / f"05_{name}_{args.variant}_confusion.png",
        )

        # reverse direction: does the honest model generalize back to the
        # original single-signer studio data?
        back = None
        original_test = cfg["paths"]["asl_original"] / "Test"
        if original_test.exists():
            back_ds = smoke_cap(
                image_folder(original_test, build_eval_transforms()), args
            )
            back = cross_dataset_eval(
                model, make_loader(back_ds, device=device), device,
                source_classes=classes, target_classes=ORIGINAL_CLASSES,
            )
            print(f"{name}: back-eval on original letters "
                  f"masked {back['masked_accuracy']:.4f}")

        payload[name] = {
            "train": result.to_dict(),
            "test": stats,
            "back_eval_original_letters": back,
        }

    payload["lr"] = args.lr
    out = save_results(
        f"05_train_aslhg_{args.variant}{args.out_suffix}", payload, cfg["paths"]["results"]
    )
    print(f"\nresults -> {out}")


if __name__ == "__main__":
    main()
