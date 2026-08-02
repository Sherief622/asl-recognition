"""Stage 03 — investigate the perfect score.

What this stage shows: for every image in the frozen test split, the pHash
distance to its nearest same-class training image. Near-zero distances at
scale mean train and test contain near-duplicate frames of the same capture
session — the random file-level split measured memorization, not
generalization. Produces the duplicate-rate table, distance histogram,
side-by-side pair figure, and sequential-frame-number evidence.
"""

import pandas as pd
from _common import base_parser, start
from aslrec.config import save_results
from aslrec.data.splits import scan_class_folders
from aslrec.leakage import (
    compute_hashes,
    duplicate_rate_table,
    nearest_train_neighbours,
    plot_distance_histogram,
    plot_duplicate_pairs,
    sequential_frame_evidence,
)


def main():
    parser = base_parser(__doc__)
    args, cfg = start(parser, "Stage 03: near-duplicate leakage investigation")
    root = cfg["paths"]["asl_original"]
    cache_dir = cfg["paths"]["aslhg"].parent / "phash_cache"

    train_frame = scan_class_folders(root / "Train")
    test_frame = scan_class_folders(root / "Test")
    if args.smoke:
        train_frame = train_frame.groupby("label").head(40).reset_index(drop=True)
        test_frame = test_frame.groupby("label").head(10).reset_index(drop=True)

    suffix = "_smoke" if args.smoke else ""
    train_hashes = compute_hashes(
        list(train_frame["path"]), cache_dir / f"train{suffix}.csv"
    )
    test_hashes = compute_hashes(
        list(test_frame["path"]), cache_dir / f"test{suffix}.csv"
    )
    # align hash rows with label rows by path
    train_hashes = train_hashes.merge(train_frame, on="path")
    test_hashes = test_hashes.merge(test_frame, on="path")

    nn_frame = nearest_train_neighbours(
        test_hashes, train_hashes, test_hashes["label"], train_hashes["label"]
    )

    # sanity checks on the metric itself
    same = nn_frame["hamming"].min()
    assert same <= 4, "expected at least one near-duplicate pair in this dataset"

    rates = duplicate_rate_table(nn_frame)
    print("\nNearest-training-image distance for test images:")
    for k, v in rates["rates"].items():
        print(f"  {k}: {v:.2%} of test set")
    print(f"  median distance: {rates['median_hamming']} bits")

    seq = sequential_frame_evidence(nn_frame)
    if seq:
        print("\nClosest pairs are adjacent capture frames (test vs train file number):")
        for row in seq[:5]:
            print(f"  class {row['label']}: frame {row['test_frame']} vs "
                  f"{row['train_frame']} (gap {row['frame_gap']}, "
                  f"{row['hamming']} bits)")

    figures = cfg["paths"]["figures"]
    plot_distance_histogram(nn_frame, figures / "03_phash_distance_histogram.png")
    plot_duplicate_pairs(nn_frame, figures / "03_duplicate_pairs.png")
    nn_frame.to_csv(cache_dir / f"nearest_neighbours{suffix}.csv", index=False)

    payload = {
        "smoke": args.smoke,
        "n_train": int(len(train_frame)),
        "n_test": int(len(test_frame)),
        "duplicate_rates": rates,
        "per_class_median_hamming": {
            label: float(sub["hamming"].median())
            for label, sub in nn_frame.groupby("label")
        },
        "sequential_frame_examples": seq,
    }
    out = save_results("03_leakage", payload, cfg["paths"]["results"])
    print(f"\nresults -> {out}")


if __name__ == "__main__":
    main()
