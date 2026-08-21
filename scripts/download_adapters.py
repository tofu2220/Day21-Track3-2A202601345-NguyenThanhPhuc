#!/usr/bin/env python3
"""Download the four Lab 21 adapters from Hugging Face."""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


ADAPTERS = {
    "correct": "phucnt9186/lab21-correct",
    "attn_only": "phucnt9186/lab21-attn-only",
    "wrong_lr": "phucnt9186/lab21-wrong-lr",
    "qlora": "phucnt9186/lab21-qlora",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=[*ADAPTERS, "all"], default="all")
    parser.add_argument("--root", type=Path, default=Path("adapters"))
    args = parser.parse_args()

    names = ADAPTERS if args.only == "all" else {args.only: ADAPTERS[args.only]}
    for name, repo_id in names.items():
        destination = args.root / name
        print(f"Downloading {repo_id} -> {destination}")
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(destination),
        )
    print("All requested adapters downloaded.")


if __name__ == "__main__":
    main()
