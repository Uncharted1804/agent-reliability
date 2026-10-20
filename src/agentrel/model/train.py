"""Training + evaluation of the silent-failure classifier (block D).

Protocol:
  1. build the feature frame from the run corpus + labels
  2. split 70/15/15 grouped by template (no paraphrase crosses folds)
  3. fit each model on train
  4. pick each model's F2 threshold on val
  5. report F2 / PR-AUC / Brier on test at that fixed threshold

Every model is given the identical frame, split and threshold procedure, so the
comparison isolates the estimator. Feature importance is reported as LightGBM
gain, which is what the report's discussion of which signals matter draws on.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..features.build import build_feature_frame, xy_groups
from ..features.schema import ALL_FEATURES
from .metrics import best_f2_threshold, evaluate, format_results
from .pipeline import adapt_smote, build_models
from .split import split_from_config


def _probabilities(model, X) -> np.ndarray:
    """P(failure) for each row, as a 1-d array."""
    proba = model.predict_proba(X)
    # a fold with one observed class yields a single column; map it to P(y=1)
    if proba.shape[1] == 1:
        only = int(getattr(model, "classes_", [0])[0])
        return np.full(len(X), 1.0 if only == 1 else 0.0)
    return proba[:, 1]


def lgbm_importance(model, top: int = 12) -> list[tuple[str, float]]:
    """Gain-based feature importance from a fitted LightGBM pipeline."""
    clf = model.named_steps["clf"]
    booster = getattr(clf, "booster_", None)
    if booster is None:
        return []
    names = list(model.named_steps["pre"].get_feature_names_out())
    gains = booster.feature_importance(importance_type="gain")
    pairs = sorted(zip(names, map(float, gains)), key=lambda p: -p[1])
    return pairs[:top]


def train_and_evaluate(runs_path: str | Path, cfg_split: dict,
                       labelled_path: str | Path | None = None,
                       out_dir: str | Path | None = None) -> dict[str, Any]:
    """Run the full training protocol and return a results dict."""
    frame = build_feature_frame(runs_path, labelled_path)
    X, y, groups = xy_groups(frame)
    split = split_from_config(groups, cfg_split)

    if y.nunique() < 2:
        raise ValueError(
            f"the corpus has a single class (all label_failure={y.iloc[0]}) — "
            "a classifier cannot be trained; inspect the labelling first")

    X_tr, y_tr = X[split.train], y[split.train]
    X_va, y_va = X[split.val], y[split.val]
    X_te, y_te = X[split.test], y[split.test]

    # The grouped split allocates whole templates, so on a small corpus a fold
    # can end up single-class. Fail loudly here: silently skipping the models
    # would print an empty results table that looks like a formatting glitch.
    n_minority = int(min((y_tr == 0).sum(), (y_tr == 1).sum()))
    if n_minority == 0:
        raise ValueError(
            f"the training fold is single-class (failure rate "
            f"{y_tr.mean():.0%} over {len(y_tr)} runs from "
            f"{len(split.train_groups)} templates). The grouped split allocates "
            f"whole templates, so this means the corpus has too few templates "
            f"for a 3-way split — generate more runs before training.")

    models = adapt_smote(build_models(ALL_FEATURES), n_minority)

    results: dict[str, dict[str, float]] = {}
    val_results: dict[str, dict[str, float]] = {}
    importance: list[tuple[str, float]] = []
    fitted: dict[str, Any] = {}

    for name, model in models.items():
        model.fit(X_tr, y_tr)
        fitted[name] = model

        # threshold chosen on validation only, then frozen for test
        p_va = _probabilities(model, X_va)
        threshold = best_f2_threshold(y_va, p_va) if y_va.nunique() > 1 else 0.5
        val_results[name] = evaluate(y_va, p_va, threshold)
        results[name] = evaluate(y_te, _probabilities(model, X_te), threshold)

        if name == "lgbm_weighted":
            importance = lgbm_importance(model)

    payload = {
        "n_runs": int(len(frame)),
        "failure_rate": float(y.mean()),
        "split": {
            "sizes": split.sizes(),
            "n_groups": {"train": len(split.train_groups), "val": len(split.val_groups),
                         "test": len(split.test_groups)},
            "test_groups": split.test_groups,
        },
        "class_balance": {
            "train": float(y_tr.mean()), "val": float(y_va.mean()),
            "test": float(y_te.mean()),
        },
        "test_metrics": results,
        "val_metrics": val_results,
        "lgbm_gain_importance": importance,
    }

    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "classifier_results.json").write_text(
            json.dumps(payload, indent=2, default=str))
        frame.to_parquet(out_dir / "features.parquet", index=False)
        pd.DataFrame(importance, columns=["feature", "gain"]).to_csv(
            out_dir / "feature_importance.csv", index=False)
    payload["_fitted"] = fitted
    return payload


def format_report(payload: dict[str, Any]) -> str:
    """Render the results dict as the text block that goes in the report."""
    lines = [
        f"runs: {payload['n_runs']}   overall failure rate: {payload['failure_rate']:.1%}",
        f"split (grouped by template): {payload['split']['sizes']}  "
        f"groups {payload['split']['n_groups']}",
        f"failure rate per fold: " + ", ".join(
            f"{k}={v:.1%}" for k, v in payload["class_balance"].items()),
        "",
        "TEST (threshold fixed from validation):",
        format_results(payload["test_metrics"]),
    ]
    if payload.get("lgbm_gain_importance"):
        lines += ["", "LightGBM gain importance (top features):"]
        lines += [f"  {name:28s} {gain:10.1f}"
                  for name, gain in payload["lgbm_gain_importance"]]
    return "\n".join(lines)
