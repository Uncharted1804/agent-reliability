#!/usr/bin/env python
"""Generate the fault-injected run corpus (block E).

Re-runs questions under each fault in configs/experiment.yaml, once per
topology, writing results/runs_faults.jsonl. Label it with:
    python scripts/label_runs.py --runs results/runs_faults.jsonl
"""
import argparse

from agentrel.faults.runner import generate_fault_runs
from agentrel.utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None, help="output JSONL path")
    ap.add_argument("--faults", nargs="*", default=None, help="subset of faults to run")
    ap.add_argument("--paraphrases", type=int, default=None,
                    help="phrasings per template (default from config)")
    ap.add_argument("--limit", type=int, default=None,
                    help="max questions per dataset (smoke test)")
    ap.add_argument("--no-mock", action="store_true",
                    help="fail instead of falling back to MockLLM")
    args = ap.parse_args()
    cfg = load_config(args.config)
    generate_fault_runs(cfg, out_path=args.out, faults=args.faults,
                        paraphrases=args.paraphrases, limit=args.limit,
                        allow_mock=not args.no_mock)


if __name__ == "__main__":
    main()
