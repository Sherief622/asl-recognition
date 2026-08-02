"""Stage 02 — reproduce the suspicious perfect score.

What this stage shows: the historical checkpoints, loaded into this repo's
ported architectures, evaluated on the frozen test split. SLRVGG8 comes out
at (or within noise of) 100.00% across 33k images — the number that
triggered the investigation in stage 03. This stage doubles as the port
fidelity gate: reproducing the notebook's numbers proves the port is exact.
"""

import torch
from _common import ORIGINAL_CLASSES, base_parser, smoke_cap, start
from aslrec.config import get_device, save_results
from aslrec.data.datasets import image_folder, make_loader
from aslrec.data.transforms import build_eval_transforms
from aslrec.engine.evaluate import collect_predictions, plot_confusion, summarize
from aslrec.models import build_model

CHECKPOINTS = {
    "slrvgg8": "slrvgg8_best.pth",
    "asl_cnn": "asl_cnn_best.pth",
}


def main():
    parser = base_parser(__doc__)
    args, cfg = start(parser, "Stage 02: reproduce the 100% baseline evaluation")
    device = get_device(args.device)

    test_ds = image_folder(
        cfg["paths"]["asl_original"] / "Test", build_eval_transforms()
    )
    assert [c for c in test_ds.classes] == ORIGINAL_CLASSES
    loader = make_loader(smoke_cap(test_ds, args), device=device)

    payload = {"dataset": "original (frozen leaky split)", "smoke": args.smoke}
    for name, ckpt_file in CHECKPOINTS.items():
        ckpt = cfg["paths"]["historical_checkpoints"] / ckpt_file
        model = build_model(name, num_classes=len(ORIGINAL_CLASSES)).to(device)
        model.load_state_dict(
            torch.load(ckpt, map_location=device, weights_only=True), strict=True
        )
        preds, labels, _ = collect_predictions(model, loader, device)
        stats = summarize(preds, labels, ORIGINAL_CLASSES)
        payload[name] = stats
        print(f"{name}: accuracy {stats['accuracy']:.6f} on {stats['n_samples']} images")
        plot_confusion(
            preds, labels, ORIGINAL_CLASSES,
            f"{name} on original test split (historical checkpoint)",
            cfg["paths"]["figures"] / f"02_{name}_confusion.png",
        )

    out = save_results("02_eval_baseline", payload, cfg["paths"]["results"])
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
