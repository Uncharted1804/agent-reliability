"""Numbers both report builders compute from results/ (build.py and build_tex.py).

Kept in one place so the reportlab PDF and the LaTeX PDF cannot drift apart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from agentrel.model.metrics import (
    evaluate_extended,
    expected_calibration_error,
    f2_score,
    grouped_bootstrap_ci,
    pr_auc,
    recall_at_budget,
)

TOPO = {"single_react": "Single ReAct", "orchestrator_worker": "Orchestrator-worker",
        "peer_tool_graph": "Peer tool-graph"}
FAULT = {"tool_failure": "Tool failure", "corrupted_output": "Corrupted output",
         "dropped_context": "Dropped context"}
MODEL = {"lgbm_weighted": "LightGBM, class-weighted (ours)", "smote_lgbm": "LightGBM + SMOTE",
         "logreg": "Logistic regression, class-weighted", "smote_logreg": "Logistic regression + SMOTE",
         "dummy_prior": "Prior (always the base rate)"}


def failure_kind(r) -> str:
    """Silent = a clean, well-formed answer that is wrong; visible = no usable answer."""
    if r["label_failure"] == 0:
        return "correct"
    if pd.isna(r["parsed_answer"]) or bool(r.get("harness_error", 0)):
        return "visible"
    if r["answer_type"] != "numeric":
        return "silent_category"
    try:
        err = abs(float(r["parsed_answer"]) - float(r["truth_value"])) / max(abs(float(r["truth_value"])), 1e-9)
    except (TypeError, ValueError):
        return "silent_value"
    return "silent_near" if err <= 0.05 else "silent_value"


def extended(oof: pd.DataFrame, theta: float) -> tuple[dict, dict[str, tuple[float, float]]]:
    """Extended out-of-fold metrics and template-clustered 95% CIs (nothing retrained)."""
    y, prob, groups = oof["label_failure"].to_numpy(), oof["p_failure_oof"].to_numpy(), oof["template_id"]

    def ci(metric):
        return grouped_bootstrap_ci(y, prob, groups, metric, n_resamples=1000)

    ext = evaluate_extended(y, prob, theta)
    ext_ci = {
        "pr_auc": ci(pr_auc),
        "roc_auc": ci(lambda a, b: evaluate_extended(a, b, theta)["roc_auc"]),
        "f2": ci(lambda a, b: f2_score(a, (b >= theta).astype(int))),
        "ece": ci(expected_calibration_error),
        "recall_at_20pct": ci(lambda a, b: recall_at_budget(a, b, 0.2)),
    }
    return ext, ext_ci


def flagged_share(oof: pd.DataFrame, theta: float) -> float:
    return float(np.mean(oof["p_failure_oof"].to_numpy() >= theta))
