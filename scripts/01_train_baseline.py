"""Stage 01 (optional) — reproduce the flawed baseline training.

What this stage shows: the exact coursework recipe (legacy augmentation
including the chirality-breaking horizontal flip, Adam 1e-3, plateau LR,
early stopping) trained on the frozen leaky split. The historical
checkpoints already encode this run, so by default the stage refuses to
burn GPU-hours when they exist — pass --force to retrain from scratch.
"""

from _common import ORIGINAL_CLASSES, base_parser, smoke_cap, start
from aslrec.config import get_device, save_results
from aslrec.data.datasets import image_folder, make_loader
from aslrec.data.transforms import build_eval_transforms, build_train_transforms
from aslrec.engine.evaluate import plot_history
from aslrec.engine.train import train_model
from aslrec.models import build_model


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--model", choices=["slrvgg8", "asl_cnn"], default="slrvgg8")
    parser.add_argument("--force", action="store_true", help="retrain even if the historical checkpoint exists")
    parser.add_argument("--epochs", type=int, default=25)
    args, cfg = start(parser, f"Stage 01: baseline training ({args_model_hint()})")

    historical = cfg["paths"]["historical_checkpoints"] / f"{ckpt_name(args.model)}"
    if historical.exists() and not args.force and not args.smoke:
        print(f"historical checkpoint exists: {historical}")
        print("nothing to do — stage 02 evaluates it. Use --force to retrain.")
        return

    device = get_device(args.device)
    root = cfg["paths"]["asl_original"]
    train_ds = image_folder(root / "Train", build_train_transforms(legacy=True))
    val_ds = image_folder(root / "Val", build_eval_transforms())
    train_ds, val_ds = smoke_cap(train_ds, args), smoke_cap(val_ds, args)

    model = build_model(args.model, num_classes=len(ORIGINAL_CLASSES)).to(device)
    ckpt = cfg["paths"]["checkpoints"] / f"{args.model}_baseline_repro.pth"
    result = train_model(
        model,
        make_loader(train_ds, shuffle=True, device=device),
        make_loader(val_ds, device=device),
        device,
        ckpt,
        epochs=1 if args.smoke else args.epochs,
    )

    plot_history(
        result.to_dict()["history"],
        f"{args.model} baseline (legacy recipe)",
        cfg["paths"]["figures"] / f"01_{args.model}_baseline_history.png",
    )
    payload = {"model": args.model, "recipe": "legacy (with horizontal flip)",
               "smoke": args.smoke, **result.to_dict()}
    out = save_results(f"01_train_baseline_{args.model}", payload, cfg["paths"]["results"])
    print(f"results -> {out}")


def args_model_hint() -> str:
    import sys

    return "asl_cnn" if "asl_cnn" in sys.argv else "slrvgg8"


def ckpt_name(model: str) -> str:
    return {"slrvgg8": "slrvgg8_best.pth", "asl_cnn": "asl_cnn_best.pth"}[model]


if __name__ == "__main__":
    main()
