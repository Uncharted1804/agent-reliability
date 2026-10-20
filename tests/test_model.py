"""Unit tests for the split, the metrics and the end-to-end training protocol.

The split and the threshold policy are where a silent methodological error
would quietly inflate the reported F2, so they are tested directly rather than
only through the training run. The corpus here is synthetic and the static
feature block is stubbed, so these tests need neither a downloaded dataset nor
a live model.
"""
import json

import numpy as np
import pandas as pd
import pytest

from agentrel.features.schema import ALL_FEATURES
from agentrel.model.metrics import (best_f2_threshold, brier, evaluate, f2_score,
                                    format_results, pr_auc)
from agentrel.model.pipeline import adapt_smote, build_models, build_preprocessor
from agentrel.model.split import grouped_split, split_from_config

SPLIT_CFG = {"scheme": "grouped_by_template", "train": 0.70, "val": 0.15,
             "test": 0.15, "seed": 42}


def _groups(n_templates=20, rows_per=15):
    return pd.Series([f"ds.q{i:02d}" for i in range(n_templates)
                      for _ in range(rows_per)])


# --- grouped split --------------------------------------------------------- #
def test_split_folds_are_disjoint_and_cover_every_row():
    g = _groups()
    s = grouped_split(g, seed=42)
    assert s.train.sum() + s.val.sum() + s.test.sum() == len(g)
    assert not (s.train & s.val).any()
    assert not (s.train & s.test).any()
    assert not (s.val & s.test).any()


def test_no_template_appears_in_two_folds():
    # the point of the scheme: a paraphrase of a training question must not test
    g = _groups()
    s = grouped_split(g, seed=42)
    assert not set(s.train_groups) & set(s.val_groups)
    assert not set(s.train_groups) & set(s.test_groups)
    assert not set(s.val_groups) & set(s.test_groups)


def test_split_respects_requested_proportions_approximately():
    s = grouped_split(_groups(n_templates=100, rows_per=15), seed=7)
    sizes = s.sizes()
    total = sum(sizes.values())
    assert sizes["train"] / total == pytest.approx(0.70, abs=0.05)
    assert sizes["test"] / total == pytest.approx(0.15, abs=0.05)


def test_split_is_deterministic_for_a_seed_and_varies_across_seeds():
    g = _groups()
    assert grouped_split(g, seed=42).test_groups == grouped_split(g, seed=42).test_groups
    assert grouped_split(g, seed=42).test_groups != grouped_split(g, seed=1).test_groups


def test_split_rejects_too_few_groups():
    with pytest.raises(ValueError, match="at least 3"):
        grouped_split(pd.Series(["a", "a", "b", "b"]))


def test_split_rejects_proportions_that_do_not_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        grouped_split(_groups(), train=0.8, val=0.3, test=0.1)


def test_split_from_config_reads_the_yaml_block():
    s = split_from_config(_groups(), SPLIT_CFG)
    assert sum(s.sizes().values()) == 300


def test_split_from_config_rejects_unknown_scheme():
    with pytest.raises(NotImplementedError):
        split_from_config(_groups(), {"scheme": "random_rows"})


# --- metrics --------------------------------------------------------------- #
def test_f2_weights_recall_above_precision():
    y = [1, 1, 1, 1, 0, 0, 0, 0]
    # catches every failure, two false alarms
    high_recall = [1, 1, 1, 1, 1, 1, 0, 0]
    # no false alarms, misses two failures
    high_precision = [1, 1, 0, 0, 0, 0, 0, 0]
    assert f2_score(y, high_recall) > f2_score(y, high_precision)


def test_pr_auc_rewards_ranking_failures_first():
    y = [0, 0, 1, 1]
    assert pr_auc(y, [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert pr_auc(y, [0.9, 0.8, 0.2, 0.1]) < 0.5


def test_brier_rewards_calibrated_probabilities():
    y = [1, 1, 0, 0]
    assert brier(y, [1.0, 1.0, 0.0, 0.0]) == pytest.approx(0.0)
    assert brier(y, [0.0, 0.0, 1.0, 1.0]) == pytest.approx(1.0)


def test_best_f2_threshold_separates_a_clean_ranking():
    y = [0, 0, 0, 1, 1, 1]
    p = [0.05, 0.1, 0.2, 0.8, 0.85, 0.9]
    t = best_f2_threshold(y, p)
    assert 0.2 < t <= 0.8
    assert f2_score(y, (np.array(p) >= t).astype(int)) == pytest.approx(1.0)


def test_evaluate_reports_every_metric_and_echoes_the_threshold():
    m = evaluate([0, 1, 1, 0], [0.1, 0.9, 0.7, 0.2], 0.5)
    assert set(m) >= {"f2", "pr_auc", "brier", "precision", "recall", "threshold"}
    assert m["threshold"] == 0.5
    assert m["positive_rate"] == 0.5


def test_format_results_renders_one_line_per_model():
    text = format_results({"a": evaluate([0, 1], [0.2, 0.8], 0.5)})
    assert "F2" in text and text.strip().splitlines()[-1].startswith("a")


# --- preprocessing + models ------------------------------------------------ #
def _frame(n=150, seed=0):
    """Synthetic feature frame where failure depends on the trace features."""
    rng = np.random.default_rng(seed)
    n_tool_calls = rng.integers(0, 12, n)
    n_errors = rng.integers(0, 4, n)
    df = pd.DataFrame({
        "n_rows": rng.integers(500, 50_000, n).astype(float),
        "n_cols": rng.integers(3, 30, n).astype(float),
        "frac_missing": rng.random(n) * 0.5,
        "frac_categorical": rng.random(n),
        "max_cardinality": rng.integers(2, 900, n).astype(float),
        "mean_abs_skew": rng.random(n) * 4,
        "correlation_density": rng.random(n),
        "question_kind": rng.choice(["aggregation", "filtering", "trend",
                                     "group_comparison", "multi_step"], n),
        "n_tables": rng.integers(1, 4, n).astype(float),
        "n_tool_calls": n_tool_calls.astype(float),
        "n_retries": rng.integers(0, 3, n).astype(float),
        "n_errors": n_errors.astype(float),
        "n_llm_calls": rng.integers(1, 24, n).astype(float),
        "prompt_tokens": rng.integers(50, 5000, n).astype(float),
        "completion_tokens": rng.integers(5, 400, n).astype(float),
        "wall_time_s": rng.random(n) * 60,
        "n_cycles": rng.integers(0, 3, n).astype(float),
        "transition_entropy": rng.random(n) * 2,
        "max_betweenness": rng.random(n),
    })
    # a run that did little work and hit errors is the failure mode
    logit = 1.4 * n_errors - 0.45 * n_tool_calls + 0.6
    p = 1 / (1 + np.exp(-logit))
    df["label_failure"] = (rng.random(n) < p).astype(int)
    return df


def test_preprocessor_log1p_and_one_hot_expand_the_matrix():
    df = _frame()
    pre = build_preprocessor(ALL_FEATURES)
    out = pre.fit_transform(df[ALL_FEATURES])
    # 18 numeric features plus 5 one-hot levels for question_kind
    assert out.shape == (len(df), 18 + 5)
    assert np.isfinite(out).all()


def test_preprocessor_tolerates_an_unseen_question_kind():
    pre = build_preprocessor(ALL_FEATURES).fit(_frame()[ALL_FEATURES])
    unseen = _frame(n=5, seed=9)
    unseen["question_kind"] = "brand_new_kind"
    assert pre.transform(unseen[ALL_FEATURES]).shape[1] == 18 + 5


def test_build_models_returns_the_predictor_and_its_baselines():
    models = build_models(ALL_FEATURES)
    assert set(models) == {"lgbm_weighted", "logreg", "smote_logreg", "smote_lgbm",
                           "dummy_prior"}


def test_adapt_smote_lowers_k_for_a_tiny_minority():
    models = adapt_smote(build_models(ALL_FEATURES), n_minority=3)
    assert models["smote_logreg"].named_steps["smote"].get_params()["k_neighbors"] == 2


def test_every_model_fits_and_produces_probabilities():
    df = _frame()
    X, y = df[ALL_FEATURES], df["label_failure"]
    for name, model in build_models(ALL_FEATURES).items():
        model.fit(X, y)
        p = model.predict_proba(X)[:, 1]
        assert p.shape == (len(df),)
        assert ((p >= 0) & (p <= 1)).all(), name


# --- end-to-end training --------------------------------------------------- #
def _write_corpus(tmp_path, n_templates=24, seed=0):
    """Synthetic run corpus + labelled table on disk, as the scripts produce."""
    rng = np.random.default_rng(seed)
    runs, labels = [], []
    for t in range(n_templates):
        template_id = f"titanic.q{t:02d}"
        for p in range(5):
            for topo in ("single_react", "orchestrator_worker", "peer_tool_graph"):
                run_id = f"{template_id}.p{p}::{topo}"
                n_err = int(rng.integers(0, 3))
                tools = list(rng.choice(["load_table", "aggregate", "filter_rows"],
                                        size=int(rng.integers(0, 6))))
                runs.append({
                    "run_id": run_id, "dataset": "titanic", "template_id": template_id,
                    "question_id": f"{template_id}.p{p}", "paraphrase_index": p,
                    "topology": topo, "model": "test",
                    "calls": [{"step": i, "agent_role": "react", "tool": tl, "args": {},
                               "ok": True, "is_retry": False} for i, tl in enumerate(tools)],
                    "n_tool_calls": len(tools), "n_retries": 0, "n_errors": n_err,
                    "n_llm_calls": len(tools) + 1, "prompt_tokens": 400,
                    "completion_tokens": 30, "wall_time_s": 5.0 + len(tools),
                    "timed_out": False, "harness_error": None,
                })
                # failure driven by errors and by doing too little work
                fail = int(rng.random() < 1 / (1 + np.exp(-(1.3 * n_err - 0.5 * len(tools) + 0.4))))
                labels.append({"run_id": run_id, "label_failure": fail,
                               "timed_out": False})

    runs_path = tmp_path / "runs.jsonl"
    with open(runs_path, "w", encoding="utf-8") as fh:
        for r in runs:
            fh.write(json.dumps(r) + "\n")
    pd.DataFrame(labels).to_parquet(tmp_path / "labelled.parquet", index=False)
    return runs_path


@pytest.fixture()
def stub_static(monkeypatch):
    """Stub the static block so the test needs no downloaded dataset."""
    import agentrel.features.build as build_mod

    def fake(dataset, question_kind, n_tables):
        return {"n_rows": 887.0, "n_cols": 8.0, "frac_missing": 0.02,
                "frac_categorical": 0.25, "max_cardinality": 887.0,
                "mean_abs_skew": 1.4, "correlation_density": 0.21,
                "question_kind": question_kind, "n_tables": float(n_tables)}

    monkeypatch.setattr(build_mod, "static_features", fake)
    monkeypatch.setattr(build_mod, "load_question_meta",
                        lambda ds: {f"titanic.q{t:02d}": {"kind": "aggregation",
                                                          "n_tables": 1,
                                                          "answer_type": "numeric"}
                                    for t in range(40)})


def test_build_feature_frame_produces_one_row_per_run(tmp_path, stub_static):
    from agentrel.features.build import build_feature_frame, xy_groups

    runs_path = _write_corpus(tmp_path)
    frame = build_feature_frame(runs_path)
    assert len(frame) == 24 * 5 * 3
    X, y, groups = xy_groups(frame)
    assert list(X.columns) == ALL_FEATURES
    assert groups.nunique() == 24


def test_build_feature_frame_raises_on_unlabelled_runs(tmp_path, stub_static):
    from agentrel.features.build import build_feature_frame

    runs_path = _write_corpus(tmp_path)
    labels = pd.read_parquet(tmp_path / "labelled.parquet").iloc[:-5]
    labels.to_parquet(tmp_path / "labelled.parquet", index=False)
    with pytest.raises(ValueError, match="no label"):
        build_feature_frame(runs_path)


def test_train_and_evaluate_reports_every_model(tmp_path, stub_static):
    from agentrel.model.train import format_report, train_and_evaluate

    runs_path = _write_corpus(tmp_path)
    payload = train_and_evaluate(runs_path, SPLIT_CFG, out_dir=tmp_path / "out")
    assert set(payload["test_metrics"]) == {"lgbm_weighted", "logreg", "smote_logreg",
                                            "smote_lgbm", "dummy_prior"}
    for name, m in payload["test_metrics"].items():
        assert 0.0 <= m["f2"] <= 1.0, name
        assert 0.0 <= m["brier"] <= 1.0, name
    assert (tmp_path / "out" / "classifier_results.json").exists()
    assert "PR-AUC" in format_report(payload)


def test_training_threshold_comes_from_validation_not_test(tmp_path, stub_static):
    from agentrel.model.train import train_and_evaluate

    runs_path = _write_corpus(tmp_path)
    payload = train_and_evaluate(runs_path, SPLIT_CFG)
    for name in payload["test_metrics"]:
        # the same threshold must appear in both reports: chosen on val, applied to test
        assert payload["test_metrics"][name]["threshold"] == \
               payload["val_metrics"][name]["threshold"]


def test_training_refuses_a_single_class_corpus(tmp_path, stub_static):
    from agentrel.model.train import train_and_evaluate

    runs_path = _write_corpus(tmp_path)
    labels = pd.read_parquet(tmp_path / "labelled.parquet")
    labels["label_failure"] = 1
    labels.to_parquet(tmp_path / "labelled.parquet", index=False)
    with pytest.raises(ValueError, match="single class"):
        train_and_evaluate(runs_path, SPLIT_CFG)
