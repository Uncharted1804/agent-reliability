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

The final report adds metrics that answer the questions a deployment asks:

    ROC-AUC            ranking quality, reported because readers expect it
    MCC                one number using all four confusion cells; unlike F1 it
                       is not inflated when the positive class is common
    balanced accuracy  mean of recall on failures and on successes
    ECE                expected calibration error over 10 equal-width bins:
                       the average gap between predicted and observed rates
    recall@k%          share of failures caught if an analyst re-checks only
                       the k% highest-scored runs -- the review-budget view

and template-clustered bootstrap CIs, since runs of one template are not
independent.

F2 needs a threshold, and the threshold is a free parameter, so it is chosen on
the validation split only (`best_f2_threshold`) and then applied unchanged to
test. Choosing it on test would report an optimistically biased F2.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, fbeta_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

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


def expected_calibration_error(y_true, y_prob, bins: int = 10) -> float:
    """Run-weighted mean |observed rate - mean p-hat| over equal-width bins."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    idx = np.minimum((y_prob * bins).astype(int), bins - 1)
    return float(sum(abs(y_true[idx == b].mean() - y_prob[idx == b].mean()) * (idx == b).sum()
                     for b in range(bins) if (idx == b).any()) / len(y_true))


def recall_at_budget(y_true, y_prob, budget: float) -> float:
    """Recall if only the top `budget` share of runs (by p-hat) is reviewed."""
    y_true = np.asarray(y_true).astype(int)
    k = round(budget * len(y_true))
    top = np.argsort(-np.asarray(y_prob, dtype=float), kind="stable")[:k]
    return float(y_true[top].sum() / y_true.sum()) if y_true.sum() else float("nan")


def evaluate_extended(y_true, y_prob, threshold: float) -> dict[str, float]:
    """evaluate() plus the deployment-facing metrics of the final report."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    out = evaluate(y_true, y_prob, threshold)
    out.update({
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "ece": expected_calibration_error(y_true, y_prob),
        **{f"recall_at_{k}pct": recall_at_budget(y_true, y_prob, k / 100) for k in (10, 20, 30)},
    })
    return out


def grouped_bootstrap_ci(y_true, y_prob, groups, metric, *, n_resamples: int = 1000,
                         seed: int = 42, level: float = 0.95) -> tuple[float, float]:
    """Percentile CI for metric(y, p), resampling whole groups (templates)."""
    y_true, y_prob, groups = (np.asarray(a) for a in (y_true, y_prob, groups))
    members = {g: np.flatnonzero(groups == g) for g in np.unique(groups)}
    keys = np.array(list(members))
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_resamples):
        idx = np.concatenate([members[g] for g in rng.choice(keys, len(keys))])
        if 0 < y_true[idx].mean() < 1:   # a metric needs both classes
            stats.append(metric(y_true[idx], y_prob[idx]))
    alpha = (1 - level) / 2
    return float(np.quantile(stats, alpha)), float(np.quantile(stats, 1 - alpha))


def format_results(results: dict[str, dict[str, float]]) -> str:
    """Render {model_name: metrics} as a fixed-width comparison table."""
    header = (f"{'model':22s} {'F2':>6s} {'PR-AUC':>7s} {'Brier':>7s} "
              f"{'prec':>6s} {'rec':>6s} {'thr':>5s}")
    lines = [header, "-" * len(header)]
    for name, m in results.items():
        lines.append(f"{name:22s} {m['f2']:6.3f} {m['pr_auc']:7.3f} {m['brier']:7.3f} "
                     f"{m['precision']:6.3f} {m['recall']:6.3f} {m['threshold']:5.2f}")
    return "\n".join(lines)
