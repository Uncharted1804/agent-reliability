"""Static features — properties of the input data and the question (report 3.4).

These describe the *task*, not the agent: they are identical across the three
topologies for a given question, which is what makes them useful for asking
"which inputs make silent failure likely regardless of how the agent is wired".

Shape statistics are computed on the table the question centres on (the primary
table, i.e. the largest when a dataset has several), while `n_tables` comes from
the question template and records how many tables the answer actually needs.
Everything here is deterministic and dataset-level, so it is cached per dataset.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from ..data.registry import load_tables
from .schema import STATIC_FEATURES

# |r| above which a numeric column pair counts as correlated. 0.3 is the
# conventional "moderate" threshold; correlation_density is the fraction of
# column pairs clearing it, a proxy for how much redundant structure the agent
# could exploit (or be misled by).
CORR_THRESHOLD = 0.3


def _primary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if "main" in tables:
        return tables["main"]
    return max(tables.values(), key=len)


def correlation_density(df: pd.DataFrame, threshold: float = CORR_THRESHOLD) -> float:
    """Fraction of numeric column pairs whose |Pearson r| exceeds `threshold`."""
    num = df.select_dtypes(include="number")
    if num.shape[1] < 2:
        return 0.0
    corr = num.corr(numeric_only=True).to_numpy()
    n = corr.shape[0]
    iu = np.triu_indices(n, k=1)
    pairs = corr[iu]
    pairs = pairs[~np.isnan(pairs)]
    if pairs.size == 0:
        return 0.0
    return float((np.abs(pairs) > threshold).mean())


def mean_abs_skew(df: pd.DataFrame) -> float:
    """Mean |skewness| over numeric columns — how heavy-tailed the table is."""
    num = df.select_dtypes(include="number")
    if num.shape[1] == 0:
        return 0.0
    skews = []
    for col in num.columns:
        s = num[col].dropna()
        if len(s) < 3 or s.nunique() <= 1:
            continue
        val = s.skew()
        if pd.notna(val):
            skews.append(abs(float(val)))
    return float(np.mean(skews)) if skews else 0.0


def table_features(df: pd.DataFrame) -> dict[str, float]:
    """Shape/messiness statistics for one table."""
    n_rows, n_cols = int(len(df)), int(df.shape[1])
    cells = n_rows * n_cols
    num_cols = df.select_dtypes(include="number").shape[1]
    return {
        "n_rows": float(n_rows),
        "n_cols": float(n_cols),
        "frac_missing": float(df.isna().to_numpy().mean()) if cells else 0.0,
        "frac_categorical": float((n_cols - num_cols) / n_cols) if n_cols else 0.0,
        "max_cardinality": float(max((df[c].nunique(dropna=True) for c in df.columns),
                                     default=0)),
        "mean_abs_skew": mean_abs_skew(df),
        "correlation_density": correlation_density(df),
    }


@lru_cache(maxsize=None)
def dataset_features(dataset: str) -> dict[str, float]:
    """Cached static features for a dataset's primary table."""
    return table_features(_primary(load_tables(dataset)))


def static_features(dataset: str, question_kind: str, n_tables: int) -> dict[str, object]:
    """Full static feature block for one run.

    `question_kind` stays a string here and is one-hot encoded downstream, so
    the raw feature frame remains human-inspectable.
    """
    feats: dict[str, object] = dict(dataset_features(dataset))
    feats["question_kind"] = question_kind
    feats["n_tables"] = float(n_tables)
    missing = set(STATIC_FEATURES) - set(feats)
    if missing:  # pragma: no cover — guards the schema contract
        raise AssertionError(f"static features missing from schema contract: {missing}")
    return feats
