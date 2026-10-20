#!/usr/bin/env python
"""Train the silent-failure classifier and its baselines (block D).

Needs the labelled corpus: scripts/generate_runs.py then scripts/label_runs.py.
Reports F2 / PR-AUC / Brier on the grouped-by-template test split.
"""
import argparse

from agentrel.model.train import format_report, train_and_evaluate
from agentrel.utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="results/runs.jsonl")
    ap.add_argument("--labelled", default=None, help="defaults to runs' sibling labelled.parquet")
    ap.add_argument("--out-dir", default="results", help="where to write results JSON")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    payload = train_and_evaluate(args.runs, cfg.raw.get("split", {}),
                                 labelled_path=args.labelled, out_dir=args.out_dir)
    print(format_report(payload))


if __name__ == "__main__":
    main()
