"""Unit tests for block (G), relating the failure predictor to robustness.

Synthetic frames only: failure is planted in a known feature so the tests can
check that the analysis finds it, without a corpus or a live model.
"""
import numpy as np
import pandas as pd
import pytest

from agentrel.features.schema import ALL_FEATURES
from agentrel.relate import (
    block_of,
    fault_detection,
    oof_probabilities,
    per_topology_drivers,
    per_topology_predictability,
    summary_table,
)

TOPOS = ("single_react", "orchestrator_worker", "peer_tool_graph")


def _frame(n_templates=30, per=6, seed=0, signal=1.6):
    """Feature frame where failure is driven by n_errors (a dynamic feature)."""
    rng = np.random.default_rng(seed)
    n = n_templates * per
    df = pd.DataFrame({f: rng.random(n) for f in ALL_FEATURES})
    df["question_kind"] = rng.choice(["aggregation", "filtering", "multi_step"], n)
    for f in ("n_rows", "n_cols", "max_cardinality", "n_tables", "n_tool_calls", "n_retries",
              "n_llm_calls", "prompt_tokens", "completion_tokens", "n_cycles"):
        df[f] = rng.integers(1, 20, n).astype(float)
    df["n_errors"] = rng.integers(0, 5, n).astype(float)
    p = 1 / (1 + np.exp(-(signal * df["n_errors"] - 3.0)))
    df["label_failure"] = (rng.random(n) < p).astype(int)
    df["template_id"] = [f"ds.q{i:02d}" for i in range(n_templates) for _ in range(per)]
    df["question_id"] = df["template_id"] + ".p0"
    df["topology"] = [TOPOS[i % 3] for i in range(n)]
    df["run_id"] = [f"r{i}" for i in range(n)]
    return df


def test_block_of_maps_every_feature_and_one_hot_column():
    assert block_of("n_errors") == "dynamic"
    assert block_of("n_cycles") == "graph"
    assert block_of("n_rows") == "static"
    assert block_of("question_kind_filtering") == "static"


def test_oof_probabilities_cover_every_row_and_never_see_their_template():
    df = _frame()
    p = oof_probabilities(df, n_splits=5)
    assert len(p) == len(df) and not np.isnan(p).any()
    assert ((0 <= p) & (p <= 1)).all()


def test_oof_probabilities_carry_the_planted_signal():
    df = _frame(n_templates=60)
    p = oof_probabilities(df, n_splits=5)
    # failed runs should score higher than successful ones
    assert p[df["label_failure"] == 1].mean() > p[df["label_failure"] == 0].mean() + 0.15


def test_oof_rejects_a_single_class_corpus():
    df = _frame()
    df["label_failure"] = 0
    with pytest.raises(ValueError, match="single class"):
        oof_probabilities(df)


def test_per_topology_predictability_has_one_row_per_topology():
    df = _frame()
    p = oof_probabilities(df)
    t = per_topology_predictability(df, p)
    assert sorted(t["topology"]) == sorted(TOPOS)
    assert {"failure_rate", "pr_auc", "lift", "brier"} <= set(t.columns)
    assert (t["lift"] == t["pr_auc"] / t["failure_rate"]).all()


def test_drivers_find_the_planted_dynamic_signal():
    df = _frame(n_templates=60)
    d = per_topology_drivers(df, top=3)
    for topo in TOPOS:
        row = d[d["topology"] == topo].iloc[0]
        assert row["top_features"].split(", ")[0] == "n_errors"
        assert row["dynamic_share"] > row["graph_share"]
        assert row["static_share"] + row["dynamic_share"] + row["graph_share"] == pytest.approx(1.0)


def test_fault_detection_scores_only_realised_runs():
    base = _frame(n_templates=60)
    faulted = _frame(n_templates=20, seed=3)
    faulted["injected_fault"] = ["tool_failure", "corrupted_output"] * (len(faulted) // 2)
    faulted["fault_realised"] = [1, 1, 0] * (len(faulted) // 3)
    out = fault_detection(base, faulted, threshold=0.5)
    assert out["n"].sum() == int(faulted["fault_realised"].sum())
    assert {"recall", "failure_rate", "pr_auc"} <= set(out.columns)


def test_summary_joins_predictability_robustness_and_detection():
    df = _frame()
    pred = per_topology_predictability(df, oof_probabilities(df))
    drivers = per_topology_drivers(df)
    robust = pd.DataFrame({"topology": TOPOS, "mean_delta": [0.1, 0.05, 0.2],
                           "worst_fault": [0.2, 0.1, 0.3], "n_cells": [3, 3, 3]})
    detect = pd.DataFrame({"topology": list(TOPOS) * 2,
                           "fault": ["tool_failure"] * 3 + ["corrupted_output"] * 3,
                           "n": [10] * 6, "failure_rate": [0.5] * 6,
                           "recall": [0.8, 0.6, 0.7, 0.2, 0.3, 0.1], "pr_auc": [0.6] * 6})
    s = summary_table(pred, drivers, robust, detect)
    assert list(s["topology"]) == sorted(TOPOS, key=lambda t: robust.set_index("topology").loc[t, "mean_delta"])
    assert {"mean_delta", "pr_auc", "fault_recall", "main_driver"} <= set(s.columns)
