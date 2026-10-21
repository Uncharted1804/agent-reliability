"""Fit the silent-failure gate, or use it to hold or release new runs.

    python scripts/gate.py fit   --runs results/runs.jsonl --out results/gate.joblib
    python scripts/gate.py score --gate results/gate.joblib --runs new_runs.jsonl [--budget 0.2]

`fit` needs the labelled corpus; `score` needs only traces, no labels.
"""
import argparse

from agentrel.features.build import build_feature_frame
from agentrel.gate import Gate, fit_gate, unlabelled_frame
from agentrel.utils.io import read_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--runs", default="results/runs.jsonl")
    f.add_argument("--out", default="results/gate.joblib")
    s = sub.add_parser("score")
    s.add_argument("--gate", default="results/gate.joblib")
    s.add_argument("--runs", required=True)
    s.add_argument("--budget", type=float, default=None, help="hold this share of runs instead of thresholding")
    s.add_argument("--out", default=None, help="CSV of run_id, p_failure, decision")
    args = ap.parse_args()

    if args.cmd == "fit":
        gate = fit_gate(build_feature_frame(args.runs))
        gate.save(args.out)
        print(f"gate saved to {args.out}; hold threshold theta = {gate.threshold:.3f}")
        return
    gate = Gate.load(args.gate)
    out = gate.decide(unlabelled_frame(list(read_jsonl(args.runs))), budget=args.budget)
    if args.out:
        out.to_csv(args.out, index=False)
    print(out.to_string(index=False))
    print(f"\nheld {int((out['decision'] == 'hold').sum())} of {len(out)} runs for review")


if __name__ == "__main__":
    main()
