"""Grouped 70/15/15 split (config: split.scheme = grouped_by_template).

Every question template appears as 5 phrasings (canonical + 4 paraphrases) and
is answered by 3 topologies, so the corpus holds up to 15 rows that are near
duplicates of each other. A random row-wise split would put paraphrases of the
same question in both train and test, and the reported F2 would mostly measure
memorisation of the question. Splitting on `template_id` keeps every relative
of a question on one side.

The split is deterministic given the seed so a reported number can be
reproduced exactly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Split:
    """Row index masks for the three folds, plus the groups in each."""
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    train_groups: list[str]
    val_groups: list[str]
    test_groups: list[str]

    def sizes(self) -> dict[str, int]:
        return {"train": int(self.train.sum()), "val": int(self.val.sum()),
                "test": int(self.test.sum())}


def grouped_split(groups: pd.Series, *, train: float = 0.70, val: float = 0.15,
                  test: float = 0.15, seed: int = 42) -> Split:
    """Partition rows by group into train/val/test at the given proportions.

    Groups (not rows) are shuffled and allocated, so the realised row
    proportions are approximate — exactly as intended, since group sizes differ.
    """
    total = train + val + test
    if not np.isclose(total, 1.0):
        raise ValueError(f"split proportions must sum to 1, got {total}")

    groups = pd.Series(groups).astype(str)
    unique = np.array(sorted(groups.unique()))
    if len(unique) < 3:
        raise ValueError(
            f"need at least 3 template groups to make a 3-way split, got {len(unique)}")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    shuffled = unique[order]

    n = len(shuffled)
    n_train = max(1, int(round(train * n)))
    n_val = max(1, int(round(val * n)))
    # test takes the remainder, so every group is used exactly once
    if n_train + n_val >= n:
        n_train, n_val = max(1, n - 2), 1
    g_train = set(shuffled[:n_train])
    g_val = set(shuffled[n_train:n_train + n_val])
    g_test = set(shuffled[n_train + n_val:])

    masks = Split(
        train=groups.isin(g_train).to_numpy(),
        val=groups.isin(g_val).to_numpy(),
        test=groups.isin(g_test).to_numpy(),
        train_groups=sorted(g_train), val_groups=sorted(g_val), test_groups=sorted(g_test),
    )
    assert not (set(masks.train_groups) & set(masks.test_groups))
    assert masks.train.sum() + masks.val.sum() + masks.test.sum() == len(groups)
    return masks


def split_from_config(groups: pd.Series, cfg_split: dict) -> Split:
    """Build the split described by the `split:` block of experiment.yaml."""
    scheme = cfg_split.get("scheme", "grouped_by_template")
    if scheme != "grouped_by_template":
        raise NotImplementedError(f"split scheme {scheme!r} not implemented")
    return grouped_split(groups, train=float(cfg_split.get("train", 0.70)),
                         val=float(cfg_split.get("val", 0.15)),
                         test=float(cfg_split.get("test", 0.15)),
                         seed=int(cfg_split.get("seed", 42)))
