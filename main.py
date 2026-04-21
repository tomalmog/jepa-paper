"""Top-level entry point: ``python main.py --config configs/...yaml``"""
from __future__ import annotations

import argparse

from src.train import main as train_main


def cli():
    ap = argparse.ArgumentParser(
        description="Hyperbolic-JEPA pretraining"
    )
    ap.add_argument("--config", "-c", required=True,
                    help="Path to a YAML config (see configs/).")
    args = ap.parse_args()
    train_main(args.config)


if __name__ == "__main__":
    cli()
