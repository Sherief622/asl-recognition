"""Evaluation: accuracy, per-class report, confusion matrices, cross-dataset
masked/unmasked protocols."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from ..data.datasets import RemappedLogits


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    preds, labels_all, logits_all = [], [], []
    for imgs, labels in tqdm(loader, desc="Eval", leave=False):
        imgs = imgs.to(device)
        logits = model(imgs)
        preds.extend(logits.argmax(dim=1).cpu().numpy())
        logits_all.append(logits.cpu())
        labels_all.extend(labels.numpy())
    return np.array(preds), np.array(labels_all), torch.cat(logits_all)


def summarize(preds, labels, class_names) -> dict:
    present = sorted(set(labels.tolist()) | set(preds.tolist()))
    report = classification_report(
        labels,
        preds,
        labels=present,
        target_names=[class_names[i] for i in present],
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float((preds == labels).mean()),
        "n_samples": int(len(labels)),
        "macro_f1": report["macro avg"]["f1-score"],
        "per_class": {
            name: {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"],
                "support": report[name]["support"],
            }
            for name in report
            if name not in ("accuracy", "macro avg", "weighted avg")
        },
    }


def plot_confusion(preds, labels, class_names, title: str, out_path: Path):
    cm = confusion_matrix(labels, preds, labels=range(len(class_names)))
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=90)
    plt.yticks(ticks, class_names)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


@torch.no_grad()
def cross_dataset_eval(model, loader, device, source_classes, target_classes) -> dict:
    """Evaluate a model on a dataset with a different class list.

    Assumes loader yields labels indexed into `target_classes`, and only for
    classes present in the source/target intersection. Reports both the
    masked protocol (argmax over shared logits only) and the unmasked
    protocol (any out-of-set argmax counts as an error).
    """
    mapper = RemappedLogits(source_classes, target_classes)
    model.eval()
    masked_correct = unmasked_correct = total = 0
    per_class = {c: {"n": 0, "masked_correct": 0} for c in mapper.shared}

    for imgs, labels in tqdm(loader, desc="XEval", leave=False):
        # drop samples whose true class has no counterpart in the source
        # model's label space (e.g. 'Nothing'/'Space' vs ASL-HG)
        target_names = [target_classes[i] for i in labels.tolist()]
        keep = [i for i, n in enumerate(target_names) if n in mapper.target_to_shared]
        if not keep:
            continue
        imgs = imgs[keep].to(device)
        target_names = [target_names[i] for i in keep]
        logits = model(imgs).cpu()
        shared_labels = torch.tensor(
            [mapper.target_to_shared[name] for name in target_names]
        )
        masked = mapper.masked_predict(logits)
        unmasked = mapper.unmasked_predict(logits)
        masked_correct += (masked == shared_labels).sum().item()
        unmasked_correct += (unmasked == shared_labels).sum().item()
        total += len(shared_labels)
        for name, m_pred, lab in zip(target_names, masked.tolist(), shared_labels.tolist()):
            per_class[name]["n"] += 1
            per_class[name]["masked_correct"] += int(m_pred == lab)

    return {
        "shared_classes": mapper.shared,
        "n_samples": total,
        "masked_accuracy": masked_correct / total,
        "unmasked_accuracy": unmasked_correct / total,
        "per_class_masked_accuracy": {
            c: (v["masked_correct"] / v["n"] if v["n"] else None)
            for c, v in per_class.items()
        },
    }


def plot_history(history: dict, title_prefix: str, out_path: Path):
    """Loss and accuracy curves side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title(f"{title_prefix} loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title(f"{title_prefix} accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
