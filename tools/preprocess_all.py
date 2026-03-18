#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from preprocess_camus import preprocess_dataset as preprocess_camus_dataset
from preprocess_echonet import preprocess_dataset as preprocess_echonet_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CAMUS and EchoNet-Dynamic preprocessing for GDKVM."
    )
    parser.add_argument("--input_root", type=Path, default=Path("~/datasets").expanduser())
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("~/datasets/processed").expanduser(),
    )
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_visualizations", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocess_camus_dataset(args)
    preprocess_echonet_dataset(args)


if __name__ == "__main__":
    main()
