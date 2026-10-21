#!/usr/bin/env python
"""Generate the agent-run corpus (block A). Needs question banks built first.

With a local Ollama model served, runs real agents; otherwise falls back to a
deterministic MockLLM so the pipeline is exercisable end-to-end offline.
"""
import argparse

from agentrel.harness.generator import generate
from agentrel.utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None, help="output JSONL path")
    ap.add_argument("--limit", type=int, default=None,
                    help="max questions per dataset (smoke test)")
    ap.add_argument("--canonical-only", action="store_true",
                    help="one phrasing per template (the model benchmark design)")
    ap.add_argument("--no-mock", action="store_true",
                    help="fail instead of falling back to MockLLM")
    args = ap.parse_args()
    cfg = load_config(args.config)
    generate(cfg, out_path=args.out, limit=args.limit, allow_mock=not args.no_mock,
             canonical_only=args.canonical_only)


if __name__ == "__main__":
    main()
