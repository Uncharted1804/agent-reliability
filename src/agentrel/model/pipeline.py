"""Model definitions: the class-weighted LightGBM predictor and its baselines.

Transforms (report 3.4) are applied inside each estimator's pipeline, never to
the frame beforehand, so scaler means, one-hot vocabularies and SMOTE synthesis
are all fitted on training rows only. Fitting them on the whole corpus would
leak validation/test information through the preprocessing.

    log1p   on counts and heavy-tailed magnitudes (tool calls, tokens, rows,
            latency) before scaling -- these span orders of magnitude
    one-hot  on question_kind
    scale    on every numeric feature

Models:
    lgbm_weighted   the predictor -- class_weight="balanced" handles the
                    imbalance by reweighting the loss, keeping every real run
    logreg          linear baseline, also class-weighted: shows how much of the
                    signal is linearly accessible
    smote_logreg    the resampling baseline: SMOTE synthesises minority rows
                    instead of reweighting
    smote_lgbm      the same resampling under the stronger model, so SMOTE vs
                    class weighting is compared on equal footing
    dummy_prior     predicts the training failure rate for every run; any model
                    that cannot beat it has learned nothing
"""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from ..features.build import CATEGORICAL_FEATURES, COUNT_FEATURES, NUMERIC_FEATURES

RANDOM_STATE = 42


def _log1p_scaler() -> Pipeline:
    # counts are non-negative by construction, so log1p is always defined
    return Pipeline([
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])


def build_preprocessor(feature_names: list[str]) -> ColumnTransformer:
    """Column-wise transforms for the feature frame, in schema order."""
    counts = [f for f in COUNT_FEATURES if f in feature_names]
    plain = [f for f in NUMERIC_FEATURES if f in feature_names and f not in counts]
    cats = [f for f in CATEGORICAL_FEATURES if f in feature_names]
    return ColumnTransformer(
        [
            ("counts", _log1p_scaler(), counts),
            ("numeric", StandardScaler(), plain),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             cats),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _lgbm(**kw):
    from lightgbm import LGBMClassifier

    # The corpus is small (hundreds of runs), so the trees are kept shallow and
    # the leaves well populated; the defaults would overfit a corpus this size.
    params = dict(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=7,
        max_depth=3,
        min_child_samples=15,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    params.update(kw)
    return LGBMClassifier(**params)


def build_models(feature_names: list[str]) -> dict[str, Pipeline]:
    """The predictor and its baselines, each as a fit-ready pipeline."""
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    def pre():
        return build_preprocessor(feature_names)

    # k_neighbours must be below the minority count of the smallest training
    # fold; 5 is the default and is lowered automatically at fit time if needed.
    def smote():
        return SMOTE(random_state=RANDOM_STATE, k_neighbors=5)

    return {
        "lgbm_weighted": Pipeline([
            ("pre", pre()),
            ("clf", _lgbm(class_weight="balanced")),
        ]),
        "logreg": Pipeline([
            ("pre", pre()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000,
                                       random_state=RANDOM_STATE)),
        ]),
        "smote_logreg": ImbPipeline([
            ("pre", pre()),
            ("smote", smote()),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "smote_lgbm": ImbPipeline([
            ("pre", pre()),
            ("smote", smote()),
            ("clf", _lgbm()),
        ]),
        "dummy_prior": Pipeline([
            ("pre", pre()),
            ("clf", DummyClassifier(strategy="prior", random_state=RANDOM_STATE)),
        ]),
    }


def adapt_smote(models: dict[str, Pipeline], n_minority: int) -> dict[str, Pipeline]:
    """Lower SMOTE's k_neighbors when the training minority class is tiny.

    SMOTE interpolates between a minority row and its k nearest minority
    neighbours, so it raises if k >= n_minority. On a small corpus that is a
    real possibility, and silently skipping the baseline would be worse than
    running it with a smaller neighbourhood.
    """
    k = max(1, min(5, n_minority - 1))
    for name, model in models.items():
        if "smote" in getattr(model, "named_steps", {}):
            model.named_steps["smote"].set_params(k_neighbors=k)
    return models
