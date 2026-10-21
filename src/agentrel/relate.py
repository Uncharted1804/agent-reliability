"""Relating the failure predictor to topology robustness, block (G).

The report's last step links the two halves of the study: for each topology,
which signals predicted failure, and how fragile that topology was. Four tables:

* **Predictability per topology.** Out-of-fold P(failure) on the baseline
  corpus from grouped K-fold (folds by template, so no run is scored by a model
  that saw a paraphrase of it), then PR-AUC per topology. `lift` divides
  PR-AUC by the failure rate, the PR-AUC a coin would get, so topologies with
  different base rates compare fairly.
* **Drivers per topology.** The class-weighted LightGBM refitted on each
  topology's runs, with its gain split across the static, dynamic and graph
  feature blocks. Topology is still never a feature; it only picks the rows.
* **Fault detection.** The predictor trained on normal operation scoring the
  fault-injected runs: does it flag the failures a broken component causes?
  The interesting cell is corrupted_output, the fault that leaves no trace.
* **Summary.** One row per topology joining the above with mean Delta_{T,f}.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .features.build import xy_groups
from .features.schema import ALL_FEATURES, DYNAMIC_FEATURES, GRAPH_FEATURES, STATIC_FEATURES
from .model.metrics import brier, pr_auc
from .model.pipeline import build_models
from .model.train import _probabilities, lgbm_importance

BLOCKS = {"static": STATIC_FEATURES, "dynamic": DYNAMIC_FEATURES, "graph": GRAPH_FEATURES}


def block_of(feature: str) -> str:
    """Feature block of a model input column, one-hot columns included."""
    base = "question_kind" if feature.startswith("question_kind_") else feature
    for block, names in BLOCKS.items():
        if base in names:
            return block
    raise KeyError(feature)


def _fit(X: pd.DataFrame, y: pd.Series):
    return build_models(ALL_FEATURES)["lgbm_weighted"].fit(X, y)


def oof_probabilities(frame: pd.DataFrame, n_splits: int = 5) -> np.ndarray:
    """Out-of-fold P(failure) for every run, folds grouped by template."""
    X, y, groups = xy_groups(frame)
    if y.nunique() < 2:
        raise ValueError("the corpus has a single class; nothing to predict")
    p = np.full(len(frame), np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        if y.iloc[tr].nunique() < 2:
            raise ValueError("a training fold is single class; too few templates")
        p[te] = _probabilities(_fit(X.iloc[tr], y.iloc[tr]), X.iloc[te])
    return p


def per_topology_predictability(frame: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    rows = []
    for topo, idx in frame.groupby("topology").indices.items():
        y = frame["label_failure"].to_numpy()[idx]
        rate = float(y.mean())
        auc = pr_auc(y, p[idx]) if 0 < rate < 1 else float("nan")
        rows.append({"topology": topo, "n": len(idx), "failure_rate": rate,
                     "pr_auc": auc, "lift": auc / rate if rate else float("nan"),
                     "brier": brier(y, p[idx])})
    return pd.DataFrame(rows)


def per_topology_drivers(frame: pd.DataFrame, top: int = 5) -> pd.DataFrame:
    rows = []
    for topo, sub in frame.groupby("topology"):
        X, y, _ = xy_groups(sub)
        if y.nunique() < 2:
            continue
        gains = lgbm_importance(_fit(X, y), top=10_000)
        total = sum(g for _, g in gains) or 1.0
        share = {b: sum(g for f, g in gains if block_of(f) == b) / total for b in BLOCKS}
        rows.append({"topology": topo,
                     "top_features": ", ".join(f for f, g in gains[:top] if g > 0),
                     **{f"{b}_share": share[b] for b in BLOCKS}})
    return pd.DataFrame(rows)


def fault_detection(baseline: pd.DataFrame, faulted: pd.DataFrame,
                    threshold: float) -> pd.DataFrame:
    """Predictor fitted on the whole baseline, scored on realised fault runs."""
    X, y, _ = xy_groups(baseline)
    model = _fit(X, y)
    flt = faulted[faulted["fault_realised"].astype(bool)].reset_index(drop=True)
    p = _probabilities(model, flt[ALL_FEATURES])
    rows = []
    for (topo, fault), idx in flt.groupby(["topology", "injected_fault"]).indices.items():
        yt = flt["label_failure"].to_numpy()[idx]
        flagged = p[idx] >= threshold
        rows.append({"topology": topo, "fault": fault, "n": len(idx),
                     "failure_rate": float(yt.mean()),
                     "recall": float(flagged[yt == 1].mean()) if yt.any() else float("nan"),
                     "false_alarm": float(flagged[yt == 0].mean()) if (yt == 0).any() else float("nan"),
                     "pr_auc": pr_auc(yt, p[idx]) if 0 < yt.mean() < 1 else float("nan")})
    return pd.DataFrame(rows)


def summary_table(predictability: pd.DataFrame, drivers: pd.DataFrame,
                  robustness: pd.DataFrame, detection: pd.DataFrame) -> pd.DataFrame:
    """One row per topology, most robust (lowest mean Delta) first."""
    det = detection.assign(hits=detection["recall"] * detection["n"] * detection["failure_rate"],
                           fails=detection["n"] * detection["failure_rate"])
    agg = det.groupby("topology")[["hits", "fails"]].sum()
    recall = (agg["hits"] / agg["fails"]).rename("fault_recall")
    silent = (detection[detection["fault"] == "corrupted_output"]
              .set_index("topology")["recall"].rename("corrupted_output_recall"))
    main = drivers.set_index("topology")[[f"{b}_share" for b in BLOCKS]].idxmax(axis=1)
    out = (robustness.set_index("topology")[["mean_delta", "worst_fault"]]
           .join(predictability.set_index("topology")[["failure_rate", "pr_auc", "lift"]])
           .join(recall).join(silent)
           .join(main.str.removesuffix("_share").rename("main_driver"))
           .reset_index().sort_values("mean_delta"))
    return out
