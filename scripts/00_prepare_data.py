"""Stage 00 — data readiness.

What this stage shows: an inventory of the frozen original dataset (the
exact leaky split that produced the 100% result), a proper train/val
manifest for the self-collected digits dataset, and acquisition plus
manifest-building for ASL-HG, the multi-signer replacement dataset.
"""

import re
import zipfile
from pathlib import Path

import pandas as pd
import requests
import torch
from _common import base_parser, start
from aslrec.config import save_results
from aslrec.data.splits import (
    check_split_invariants,
    scan_class_folders,
    stratified_split,
    write_manifest,
)

# The dataset metadata endpoint includes per-file download URLs.
MENDELEY_API = "https://data.mendeley.com/public-api/datasets/j4y5w2c8w9"


def inventory_original(cfg) -> dict:
    root = cfg["paths"]["asl_original"]
    inv = {}
    for split in ("Train", "Val", "Test"):
        split_dir = root / split
        if not split_dir.exists():
            inv[split] = None
            continue
        counts = {}
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            counts[class_dir.name] = sum(
                1 for f in class_dir.iterdir()
                if f.suffix.lower() == ".jpg" and not f.name.startswith("._")
            )
        inv[split] = {"classes": len(counts), "total": sum(counts.values()),
                      "per_class": counts}
    return inv


def audit_historical_digits_folders(cfg) -> dict:
    """Measure the overlap in the coursework's Digits_Train/Digits_Test copy
    split. The split script was evidently run more than once (each run copied
    a different random 80/20 partition into the same folders without clearing
    them), leaving images present on BOTH sides. Recorded as evidence; those
    folders are not used for anything else in this repo.
    """
    train = scan_class_folders(cfg["paths"]["digits_train"])
    test = scan_class_folders(cfg["paths"]["digits_test"])
    key = lambda frame: set(zip(frame["label"], frame["path"].map(lambda p: Path(p).name)))
    overlap = len(key(train) & key(test))
    report = {
        "n_train_folder": int(len(train)),
        "n_test_folder": int(len(test)),
        "images_in_both": overlap,
        "share_of_test_also_in_train": overlap / len(test),
    }
    print(
        f"historical digits folders: {report['n_train_folder']} train, "
        f"{report['n_test_folder']} test, {overlap} in BOTH "
        f"({report['share_of_test_also_in_train']:.0%} of test)"
    )
    return report


def build_digits_manifest(cfg, seed: int) -> dict:
    """Disjoint train/val/test split for the self-collected digits data,
    built directly from the 990-image source (ASL_Digits_Dataset).

    The historical Digits_Train/Digits_Test folders are NOT used: they
    overlap (see audit_historical_digits_folders), and the notebook also
    early-stopped on the test set. This manifest replaces both flaws with a
    stratified 70/15/15 split whose invariants are asserted.
    """
    frame = scan_class_folders(cfg["paths"]["digits_source"])
    frame = stratified_split(
        frame, {"train": 0.70, "val": 0.15, "test": 0.15}, seed
    )
    report = check_split_invariants(frame)
    out = write_manifest(frame, cfg["paths"]["manifests"] / "digits.csv")
    print(f"digits manifest -> {out} {report['by_split']}")
    return report


def download_aslhg(cfg) -> bool:
    """Fetch the ASL-HG zips from Mendeley's public API. Returns success."""
    dest = cfg["paths"]["aslhg"]
    if any(dest.glob("**/*.jpg")) or any(dest.glob("**/*.png")):
        print(f"ASL-HG already present under {dest}")
        return True
    dest.mkdir(parents=True, exist_ok=True)
    try:
        listing = requests.get(MENDELEY_API, timeout=30)
        listing.raise_for_status()
        files = listing.json()["files"]
    except Exception as exc:
        print(f"Could not query Mendeley API ({exc}).")
        print("Manual fallback: download from "
              "https://data.mendeley.com/datasets/j4y5w2c8w9/1 and extract "
              f"the zips into {dest}")
        return False
    for entry in files:
        name = entry.get("filename") or entry.get("name", "file.zip")
        url = (entry.get("content_details") or {}).get("download_url") or entry.get("download_url")
        if not url:
            continue
        zip_path = dest / name
        if not zip_path.exists():
            print(f"downloading {name} ...")
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
        if zip_path.suffix.lower() == ".zip":
            print(f"extracting {name} ...")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(dest)
    return True


def locate_class_roots(dest: Path) -> dict[str, list[Path]]:
    """Find directories whose children look like the 36 class folders.

    Returns variant -> list of scan roots. The processed variant ships with
    its own train/test subfolders; both are scanned and re-split by signer,
    so our signer-grouped protocol applies uniformly to both variants.
    """
    variants: dict[str, list[Path]] = {}
    class_names = {*"ABCDEFGHIJKLMNOPQRSTUVWXYZ", *"0123456789"}
    for candidate in [dest, *dest.rglob("*")]:
        if not candidate.is_dir():
            continue
        children = {c.name.upper() for c in candidate.iterdir() if c.is_dir()}
        if len(children & class_names) >= 20:
            key = "processed" if "process" in str(candidate).lower() else "raw"
            variants.setdefault(key, []).append(candidate)
    return variants


SIGNER_PATTERNS = [
    re.compile(r"(?:signer|subject|participant|person|user|vol(?:unteer)?)[_\- ]?(\d+)", re.I),
    re.compile(r"^[Pp](\d+)[_\-]"),
]


def parse_signer(path: Path) -> str | None:
    for part in (*path.parts[-3:],):
        for pat in SIGNER_PATTERNS:
            m = pat.search(part)
            if m:
                return f"signer_{int(m.group(1)):02d}"
    return None


def build_aslhg_manifest(cfg, seed: int) -> dict | None:
    variants = locate_class_roots(cfg["paths"]["aslhg"])
    if not variants:
        print("ASL-HG class folders not found — run again after downloading.")
        return None
    reports = {}
    for variant, roots in variants.items():
        frame = pd.concat(
            [scan_class_folders(r, signer_parser=parse_signer) for r in roots],
            ignore_index=True,
        )
        signers = frame["signer"].dropna().unique()
        if len(signers) >= 5:
            # signer-grouped split: all of a signer's images stay together
            ordered = sorted(signers)
            assignment = {
                "train": ordered[: len(ordered) - 3],
                "val": [ordered[-3]],
                "test": ordered[-2:],
            }
            from aslrec.data.splits import signer_split

            frame = signer_split(frame, assignment)
            strategy = f"signer-grouped ({len(ordered)} signers)"
        else:
            # No recoverable signer ids: stratified random. CAVEAT — images of
            # the same signer can appear in train and test; recorded so the
            # README reports the honest limitation.
            frame = stratified_split(
                frame, {"train": 0.7, "val": 0.15, "test": 0.15}, seed
            )
            strategy = "stratified-random (no signer ids recoverable)"
        report = check_split_invariants(frame)
        report["strategy"] = strategy
        report["classes"] = sorted(frame["label"].unique())
        out = write_manifest(frame, cfg["paths"]["manifests"] / f"aslhg_{variant}.csv")
        print(f"ASL-HG [{variant}] manifest -> {out}")
        print(f"  strategy: {strategy}; splits: {report['by_split']}")
        reports[variant] = report
    return reports


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--skip-download", action="store_true")
    args, cfg = start(parser, "Stage 00: data preparation and inventory")

    if torch.cuda.is_available():
        print(f"CUDA OK: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not available — training stages will be slow. "
              "RTX 50-series needs torch>=2.7 cu128 wheels.")

    results = {"seed": args.seed}
    results["original_dataset"] = inventory_original(cfg)
    for split, info in results["original_dataset"].items():
        if info:
            print(f"original {split}: {info['total']} images, {info['classes']} classes")

    results["historical_digits_folder_overlap"] = audit_historical_digits_folders(cfg)
    results["digits"] = build_digits_manifest(cfg, args.seed)

    if not args.skip_download:
        download_aslhg(cfg)
    aslhg = build_aslhg_manifest(cfg, args.seed)
    if aslhg:
        results["aslhg"] = aslhg

    out = save_results("00_data_inventory", results, cfg["paths"]["results"])
    print(f"\nresults -> {out}")


if __name__ == "__main__":
    main()
