#!/usr/bin/env python
"""Compute topology robustness Delta_{T,f} with bootstrap CIs (block F).

Needs both corpora labelled:
    python scripts/label_runs.py --runs results/runs.jsonl
    python scripts/label_runs.py --runs results/runs_faults.jsonl
"""
import argparse
from pathlib import Path

import pandas as pd

from agentrel.faults.robustness import format_robustness, prepare, robustness_table
from agentrel.utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", default="results/labelled.parquet")
    ap.add_argument("--faulted", default="results/runs_faults_labelled.parquet")
    ap.add_argument("--out", default="results/robustness.csv")
    ap.add_argument("--resamples", type=int, default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    n_resamples = args.resamples or cfg.bootstrap_resamples

    baseline = pd.read_parquet(args.baseline)
    faulted = pd.read_parquet(args.faulted)
    _, _, info = prepare(baseline, faulted)
    table = robustness_table(baseline, faulted, n_resamples=n_resamples,
                             seed=int(cfg.raw.get("split", {}).get("seed", 42)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(format_robustness(table, info))
    print(f"\nWritten to {out}  ({n_resamples} bootstrap resamples)")


if __name__ == "__main__":
    main()
