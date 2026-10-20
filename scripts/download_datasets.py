#!/usr/bin/env python
"""Materialise the 5 datasets into datasets/raw/. Downloads where a stable
public URL exists, synthesises a controlled fallback otherwise (or on failure)."""
import argparse

from agentrel.data.download import materialise_all


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="re-materialise even if present")
    args = ap.parse_args()
    materialise_all(seed=args.seed, force=args.force)


if __name__ == "__main__":
    main()
