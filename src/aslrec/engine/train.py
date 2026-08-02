"""Generic training loop ported from the notebook.

Same recipe as the coursework: Adam, CrossEntropyLoss, ReduceLROnPlateau on
validation loss, early stopping on validation accuracy, best-checkpoint
saving. Generalized so every stage (baseline reproduction, ASL-HG retrain,
transfer-learning head training) uses the identical engine.
"""

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


@dataclass
class TrainResult:
    best_val_acc: float = 0.0
    best_epoch: int = 0
    epochs_ran: int = 0
    train_losses: list = field(default_factory=list)
    val_losses: list = field(default_factory=list)
    train_accs: list = field(default_factory=list)
    val_accs: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "best_val_acc": self.best_val_acc,
            "best_epoch": self.best_epoch,
            "epochs_ran": self.epochs_ran,
            "history": {
                "train_loss": self.train_losses,
                "val_loss": self.val_losses,
                "train_acc": self.train_accs,
                "val_acc": self.val_accs,
            },
        }


def run_epoch(model, loader, criterion, device, optimizer=None, desc="Train"):
    """One pass over loader. Trains when an optimizer is given, else evaluates."""
    training = optimizer is not None
    model.train() if training else model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for imgs, labels in tqdm(loader, desc=desc, leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += imgs.size(0)
    return running_loss / total, correct / total


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    checkpoint_path: Path,
    epochs: int = 25,
    lr: float = 1e-3,
    patience: int = 3,
    trainable_params=None,
) -> TrainResult:
    """Train with early stopping; the best (val accuracy) weights are saved to
    checkpoint_path and re-loaded into the model before returning."""
    criterion = nn.CrossEntropyLoss()
    params = trainable_params if trainable_params is not None else model.parameters()
    optimizer = optim.Adam(params, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=3, factor=0.5
    )

    result = TrainResult()
    epochs_no_improve = 0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer, desc=f"Train {epoch}"
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, device, desc=f"Val {epoch}"
        )
        scheduler.step(val_loss)

        result.train_losses.append(train_loss)
        result.val_losses.append(val_loss)
        result.train_accs.append(train_acc)
        result.val_accs.append(val_acc)
        result.epochs_ran = epoch

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        if val_acc > result.best_val_acc:
            result.best_val_acc = val_acc
            result.best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            print("  saved new best model")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  early stop: no improvement in {patience} epochs")
                break

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    return result
