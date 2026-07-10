from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a GDKVM/DPFR run directory.")
    parser.add_argument("run_dir", nargs="?", default=".", help="Run directory to inspect.")
    parser.add_argument("--split", default="val", help="Dataset split label for future visual exports.")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"Run directory does not exist: {run_dir}")
    print(f"Visualization entrypoint ready for {run_dir} [{args.split}]")
