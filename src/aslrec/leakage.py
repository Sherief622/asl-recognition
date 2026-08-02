"""Near-duplicate leakage analysis via perceptual hashing.

The tooling behind Act II of the report: for every test image, find the
closest training image by pHash Hamming distance. If a large fraction of the
test set sits within a few bits of a training image, the random split is
measuring memorization, not generalization.
"""

import re
from pathlib import Path

import imagehash
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

HASH_SIZE = 8  # 64-bit pHash


def compute_hashes(paths: list[str], cache_file: Path | None = None) -> pd.DataFrame:
    """pHash every image; returns dataframe(path, hash_int). Cached to CSV."""
    if cache_file is not None and cache_file.exists():
        cached = pd.read_csv(cache_file, dtype={"hash_int": np.uint64})
        if set(cached["path"]) == set(paths):
            return cached
    rows = []
    for p in tqdm(paths, desc="pHash", unit="img"):
        with Image.open(p) as img:
            h = imagehash.phash(img, hash_size=HASH_SIZE)
        # pack the 64-bit boolean hash into a python int for fast xor
        rows.append({"path": p, "hash_int": int(str(h), 16)})
    frame = pd.DataFrame(rows)
    frame["hash_int"] = frame["hash_int"].astype(np.uint64)
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cache_file, index=False)
    return frame


def _popcount_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamming distances between two uint64 hash vectors (all pairs)."""
    x = a[:, None] ^ b[None, :]
    # vectorized popcount over uint64
    count = np.zeros(x.shape, dtype=np.uint8)
    for _ in range(8):
        count += (x & 1).astype(np.uint8)
        x >>= 1
        count += (x & 1).astype(np.uint8)
        x >>= 1
        count += (x & 1).astype(np.uint8)
        x >>= 1
        count += (x & 1).astype(np.uint8)
        x >>= 1
    return count


def nearest_train_neighbours(
    test_frame: pd.DataFrame,
    train_frame: pd.DataFrame,
    test_labels: pd.Series,
    train_labels: pd.Series,
) -> pd.DataFrame:
    """For each test image, distance to & path of its nearest training image.

    Compared within the same class (the relevant leak: same-class duplicate
    frames landing on both sides of the split).
    """
    results = []
    for label in tqdm(sorted(test_labels.unique()), desc="NN search"):
        test_sub = test_frame[test_labels == label]
        train_sub = train_frame[train_labels == label]
        if test_sub.empty or train_sub.empty:
            continue
        t = test_sub["hash_int"].to_numpy(dtype=np.uint64)
        tr = train_sub["hash_int"].to_numpy(dtype=np.uint64)
        dists = _popcount_matrix(t, tr)
        nn_idx = dists.argmin(axis=1)
        nn_dist = dists[np.arange(len(t)), nn_idx]
        for row, d, j in zip(test_sub.itertuples(), nn_dist, nn_idx):
            results.append({
                "test_path": row.path,
                "label": label,
                "nn_train_path": train_sub.iloc[int(j)]["path"],
                "hamming": int(d),
            })
    return pd.DataFrame(results)


def duplicate_rate_table(nn_frame: pd.DataFrame, thresholds=(0, 2, 4, 8)) -> dict:
    total = len(nn_frame)
    return {
        "n_test": total,
        "rates": {
            f"<={t} bits": float((nn_frame["hamming"] <= t).mean())
            for t in thresholds
        },
        "median_hamming": float(nn_frame["hamming"].median()),
        "mean_hamming": float(nn_frame["hamming"].mean()),
    }


def plot_distance_histogram(nn_frame: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(8, 4.5))
    plt.hist(nn_frame["hamming"], bins=range(0, 34), edgecolor="black")
    plt.title("Nearest-training-image pHash distance for every test image")
    plt.xlabel("Hamming distance (bits, 64-bit pHash)")
    plt.ylabel("test images")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_duplicate_pairs(nn_frame: pd.DataFrame, out_path: Path, n_pairs: int = 8):
    """Side-by-side grid of the closest test/train pairs across distinct classes."""
    best = (
        nn_frame.sort_values("hamming")
        .drop_duplicates("label")
        .head(n_pairs)
        .reset_index(drop=True)
    )
    fig, axes = plt.subplots(2, len(best), figsize=(2.2 * len(best), 5))
    for i, row in best.iterrows():
        for ax, path, title in (
            (axes[0][i], row["test_path"], f"test '{row['label']}'"),
            (axes[1][i], row["nn_train_path"], f"train (d={row['hamming']})"),
        ):
            with Image.open(path) as img:
                ax.imshow(img)
            ax.set_title(title, fontsize=8)
            ax.axis("off")
    fig.suptitle("Test images (top) and their nearest training images (bottom)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def sequential_frame_evidence(nn_frame: pd.DataFrame, n_examples: int = 10) -> list[dict]:
    """Show that many nearest pairs are numerically adjacent frame indices —
    direct evidence the dataset is correlated frames scattered by the split."""
    def frame_no(path: str) -> int | None:
        m = re.search(r"(\d+)\.[A-Za-z]+$", Path(path).name)
        return int(m.group(1)) if m else None

    rows = []
    for row in nn_frame.sort_values("hamming").itertuples():
        a, b = frame_no(row.test_path), frame_no(row.nn_train_path)
        if a is None or b is None:
            continue
        rows.append({
            "label": row.label,
            "test_frame": a,
            "train_frame": b,
            "frame_gap": abs(a - b),
            "hamming": row.hamming,
        })
        if len(rows) >= n_examples:
            break
    return rows
