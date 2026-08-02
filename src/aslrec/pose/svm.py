"""Linear SVM over hand-keypoint features.

The notebook's C-value selection plot had two bugs (plt.bar given a list of
(C, accuracy) tuples, and a colorbar on a bar chart, both of which raise);
this version separates values properly and drops the colorbar.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

C_GRID = (0.1, 1.0, 10.0)


def train_svm(X_train, y_train, X_val, y_val, c_grid=C_GRID):
    """Fit a linear SVM per C, select on validation accuracy.

    LinearSVC (liblinear) rather than SVC(kernel='linear'): identical model
    family, but it scales to the ~100k-sample training sets here, where
    libsvm's quadratic solver would not finish in reasonable time.
    """
    best_val_acc, best_model, best_c = 0.0, None, None
    val_accuracies = []
    for C in c_grid:
        model = LinearSVC(C=C)
        model.fit(X_train, y_train)
        val_acc = model.score(X_val, y_val)
        val_accuracies.append((C, val_acc))
        print(f"  C={C}: val acc {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc, best_model, best_c = val_acc, model, C
    return best_model, best_c, val_accuracies


def plot_c_selection(val_accuracies, out_path: Path, title: str):
    cs = [str(c) for c, _ in val_accuracies]
    accs = [acc for _, acc in val_accuracies]
    plt.figure(figsize=(6, 4))
    plt.bar(cs, accs, edgecolor="black")
    plt.title(title)
    plt.xlabel("C")
    plt.ylabel("validation accuracy")
    plt.ylim(0, 1)
    for x, acc in zip(cs, accs):
        plt.text(x, acc + 0.01, f"{acc:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path
