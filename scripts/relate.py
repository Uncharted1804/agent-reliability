#!/usr/bin/env python
"""Relate the failure predictor to topology robustness (block G).

Needs both corpora labelled and the robustness table:
    python scripts/label_runs.py --runs results/runs.jsonl
    python scripts/label_runs.py --runs results/runs_faults.jsonl
    python scripts/robustness.py
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from agentrel.faults.robustness import mean_degradation
from agentrel.features.build import build_feature_frame
from agentrel.model.metrics import best_f2_threshold
from agentrel.relate import (
    fault_detection,
    oof_probabilities,
    per_topology_drivers,
    per_topology_predictability,
    summary_table,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="results/runs.jsonl")
    ap.add_argument("--fault-runs", default="results/runs_faults.jsonl")
    ap.add_argument("--robustness", default="results/robustness.csv")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    out = Path(args.out_dir)

    base = build_feature_frame(args.runs)
    flt = build_feature_frame(args.fault_runs)
    # fault provenance lives in the labelled fault table, not the feature frame
    prov = pd.read_parquet(Path(args.fault_runs).with_name(
        f"{Path(args.fault_runs).stem}_labelled.parquet"))[["run_id", "injected_fault", "fault_realised"]]
    flt = flt.merge(prov, on="run_id", validate="one_to_one")

    p = oof_probabilities(base)
    threshold = best_f2_threshold(base["label_failure"], p)
    pred = per_topology_predictability(base, p)
    drivers = per_topology_drivers(base)
    detect = fault_detection(base, flt, threshold)
    summary = summary_table(pred, drivers, mean_degradation(pd.read_csv(args.robustness)), detect)

    base.assign(p_failure_oof=p)[["run_id", "topology", "template_id", "label_failure",
                                  "p_failure_oof"]].to_csv(out / "oof_predictions.csv", index=False)
    for name, t in (("predictability", pred), ("drivers", drivers),
                    ("fault_detection", detect), ("summary", summary)):
        t.to_csv(out / f"relate_{name}.csv", index=False)
    (out / "relate.json").write_text(json.dumps(
        {"threshold_f2_oof": threshold, "summary": summary.to_dict("records"),
         "fault_detection": detect.to_dict("records"), "drivers": drivers.to_dict("records")},
        indent=2, default=float))

    pd.set_option("display.width", 160)
    print(f"F2 threshold from out-of-fold baseline predictions: {threshold:.3f}\n")
    for title, t in (("predictability per topology (out-of-fold)", pred),
                     ("drivers per topology (LightGBM gain share)", drivers),
                     ("predictor on fault-injected runs", detect),
                     ("summary, most robust first", summary)):
        print(f"{title}:\n{t.round(3).to_string(index=False)}\n")


if __name__ == "__main__":
    main()
