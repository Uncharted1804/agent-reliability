"""The silent-failure gate, Algorithm 3 of the final report.

A trained gate scores a finished run from its trace and either holds the answer
for review or releases it with p-hat attached as a risk score:

    p-hat = f(phi(trace));  hold if p-hat >= theta, else release

f is the class-weighted LightGBM fitted on the whole labelled corpus. theta is
the F2-optimal threshold on *out-of-fold* scores (grouped by template), so it is
chosen on predictions the model made for templates it had not seen, not on its
own training fit. In budget mode the threshold is ignored and the top
ceil(k * N) runs of a batch are held, for a team that can re-check a fixed share.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features.build import features_for_run, load_question_meta, xy_groups
from .features.schema import ALL_FEATURES
from .model.metrics import best_f2_threshold
from .model.train import _probabilities
from .relate import _fit, oof_probabilities


@dataclass
class Gate:
    model: object
    threshold: float

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        return _probabilities(self.model, frame[ALL_FEATURES])

    def decide(self, frame: pd.DataFrame, budget: float | None = None) -> pd.DataFrame:
        p = self.score(frame)
        if budget is None:
            hold = p >= self.threshold
        else:
            hold = np.zeros(len(p), dtype=bool)
            hold[np.argsort(-p, kind="stable")[:math.ceil(budget * len(p))]] = True
        out = frame[[c for c in ("run_id", "topology", "question_id") if c in frame]].copy()
        out["p_failure"] = p
        out["decision"] = np.where(hold, "hold", "release")
        return out

    def save(self, path: str | Path) -> None:
        joblib.dump({"model": self.model, "threshold": self.threshold}, path)

    @classmethod
    def load(cls, path: str | Path) -> Gate:
        return cls(**joblib.load(path))


def fit_gate(frame: pd.DataFrame) -> Gate:
    """Threshold from out-of-fold scores, then the model refitted on every run."""
    X, y, _ = xy_groups(frame)
    theta = best_f2_threshold(y, oof_probabilities(frame))
    return Gate(model=_fit(X, y), threshold=theta)


def unlabelled_frame(runs: list[dict]) -> pd.DataFrame:
    """Feature rows for runs that have no label yet, i.e. answers about to be shown."""
    meta = load_question_meta(sorted({r["dataset"] for r in runs}))
    return pd.DataFrame([features_for_run(r, meta) for r in runs])
