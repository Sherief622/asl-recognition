"""Shared bootstrap for the numbered pipeline scripts."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aslrec.config import (  # noqa: E402
    LETTER_CLASSES,
    ORIGINAL_CLASSES,
    get_device,
    load_config,
    save_results,
    set_seed,
)


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seed", type=int, default=None, help="override config seed")
    parser.add_argument("--device", default="auto", help="cuda | cpu | mps | auto")
    parser.add_argument(
        "--smoke", action="store_true",
        help="tiny-subset, 1-epoch run to validate the code path",
    )
    return parser


def start(parser: argparse.ArgumentParser, banner: str):
    """Parse args, load config, seed, and print the stage banner."""
    args = parser.parse_args()
    cfg = load_config()
    seed = args.seed if args.seed is not None else cfg["seed"]
    set_seed(seed)
    args.seed = seed
    print("=" * 78)
    print(banner)
    print("=" * 78)
    return args, cfg


def smoke_cap(dataset, args, n: int = 200):
    """Cap a torch dataset to n samples when --smoke is set."""
    if not args.smoke:
        return dataset
    import torch.utils.data as tud

    return tud.Subset(dataset, range(min(n, len(dataset))))
