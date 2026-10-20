#!/usr/bin/env python
"""Expand templates into question banks and compute deterministic ground truth.

Writes, per dataset, datasets/question_banks/<dataset>.json containing:
    questions   : list of expanded (template x paraphrase) question instances
    ground_truth: {template_id: {value, answer_type, kind, ...}}

These files are the fixed inputs to the agent-run generator. They are checked
into git (small, deterministic) so a run corpus is always reproducible.
"""
import argparse
import json
from pathlib import Path

from agentrel.data.ground_truth import compute_ground_truth
from agentrel.data.questions import expand_questions
from agentrel.utils.config import REPO_ROOT, load_config

OUT_DIR = REPO_ROOT / "datasets" / "question_banks"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    grand_total_q = 0
    summary = []
    for ds in cfg.datasets:
        questions = expand_questions(ds)
        truth = compute_ground_truth(ds)
        bad = [tid for tid, g in truth.items() if g["error"]]
        payload = {"dataset": ds, "questions": questions, "ground_truth": truth}
        (OUT_DIR / f"{ds}.json").write_text(json.dumps(payload, indent=2, default=str))
        grand_total_q += len(questions)
        summary.append((ds, len(truth), len(questions), len(bad)))
        print(f"  {ds:14s} templates={len(truth):2d}  questions={len(questions):3d}"
              f"  truth_errors={len(bad)}")
        if bad:
            print(f"       !! ground-truth errors in: {bad}")

    print(f"\nTotal question instances: {grand_total_q} "
          f"(expected {len(cfg.datasets)*cfg.templates_per_dataset*(cfg.paraphrases_per_template+1)}"
          f" incl. canonical)")
    print(f"Written to {OUT_DIR}")


if __name__ == "__main__":
    main()
