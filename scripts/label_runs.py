#!/usr/bin/env python
"""Label + clean the run corpus (block B). Produces results/labelled.parquet
(+ .csv) with one row per run, a binary failure label, and the paraphrase-
consistency flag."""
import argparse

from agentrel.labelling.pipeline import run_pipeline, summarise
from agentrel.utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="results/runs.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    df = run_pipeline(args.runs, eps=cfg.numeric_tolerance, out_path=args.out)
    print(summarise(df))


if __name__ == "__main__":
    main()
