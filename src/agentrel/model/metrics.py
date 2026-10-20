"""Evaluation metrics for the silent-failure classifier (report 3.5).

The three reported metrics answer three different questions, which is why none
of them is accuracy:

    F2        recall-weighted (beta=2): missing a silent failure is far worse
              than flagging a correct run for review, so recall counts four
              times as much as precision.
    PR-AUC    threshold-free quality on the minority (failure) class. Unlike
              ROC-AUC it does not flatter a classifier on an imbalanced corpus.
    Brier     calibration of p-hat itself. The predictor is meant to be *used*
              as a confidence score, so being right is not enough -- 0.8 has to
              mean 0.8.

F2 needs a threshold, and the threshold is a free parameter, so it is chosen on
the validation split only (`best_f2_threshold`) and then applied unchanged to
test. Choosing it on test would report an optimistically biased F2.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, fbeta_score,
                             precision_score, recall_score)

BETA = 2.0


def f2_score(y_true, y_pred) -> float:
    """F-beta with beta=2 (recall weighted 4x precision)."""
    return float(fbeta_score(y_true, y_pred, beta=BETA, zero_division=0))


def pr_auc(y_true, y_prob) -> float:
    """Average precision — area under the precision-recall curve."""
    return float(average_precision_score(y_true, y_prob))


def brier(y_true, y_prob) -> float:
    """Mean squared error of the predicted probabilities (lower is better)."""
    return float(brier_score_loss(y_true, y_prob))


def best_f2_threshold(y_true, y_prob, grid: int = 199) -> float:
    """Threshold maximising F2 on the given (validation) data.

    Scans the interior of (0, 1) rather than the observed probabilities so the
    chosen threshold does not depend on a single run's score, and prefers the
    lower threshold on ties — consistent with recall being the priority.
    """
    thresholds = np.linspace(0.005, 0.995, grid)
    best_t, best_s = 0.5, -1.0
    for t in thresholds:
        s = f2_score(y_true, (np.asarray(y_prob) >= t).astype(int))
        if s > best_s:
            best_t, best_s = float(t), s
    return best_t


def evaluate(y_true, y_prob, threshold: float) -> dict[str, float]:
    """All reported metrics at a fixed threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "f2": f2_score(y_true, y_pred),
        "pr_auc": pr_auc(y_true, y_prob),
        "brier": brier(y_true, y_prob),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
    }


def format_results(results: dict[str, dict[str, float]]) -> str:
    """Render {model_name: metrics} as a fixed-width comparison table."""
    header = (f"{'model':22s} {'F2':>6s} {'PR-AUC':>7s} {'Brier':>7s} "
              f"{'prec':>6s} {'rec':>6s} {'thr':>5s}")
    lines = [header, "-" * len(header)]
    for name, m in results.items():
        lines.append(f"{name:22s} {m['f2']:6.3f} {m['pr_auc']:7.3f} {m['brier']:7.3f} "
                     f"{m['precision']:6.3f} {m['recall']:6.3f} {m['threshold']:5.2f}")
    return "\n".join(lines)
