"""Feature assembly — block (C) end to end.

Joins the three feature blocks into one frame, one row per run:

    static   (dataset + question properties)   identical across topologies
    dynamic  (execution trace)                 differs per run
    graph    (tool-interaction graph)          differs per run

plus the identifier columns the split and the robustness study need, and the
failure label from block (B).

Topology is deliberately *not* a feature. It is the variable the robustness
study manipulates, so letting the classifier read it would confound "this
wiring fails more" with "this task is hard". It is carried as metadata instead.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..utils.config import REPO_ROOT
from ..utils.io import read_jsonl
from .dynamic import dynamic_features
from .graph import graph_features
from .schema import ALL_FEATURES, DYNAMIC_FEATURES, GRAPH_FEATURES, STATIC_FEATURES
from .static import static_features

QUESTION_BANK_DIR = REPO_ROOT / "datasets" / "question_banks"

# identifier / metadata columns kept alongside the features
META_COLUMNS = ["run_id", "dataset", "template_id", "question_id", "topology",
                "paraphrase_index"]

# transform groups (report 3.4: log1p on counts, one-hot on categoricals,
# standardise numerics). Applied in the model pipeline, not here, so the raw
# frame stays inspectable and the transforms are fitted on train folds only.
COUNT_FEATURES = [f for f in DYNAMIC_FEATURES] + ["n_rows", "n_cols", "max_cardinality",
                                                  "n_cycles", "n_tables"]
CATEGORICAL_FEATURES = ["question_kind"]
NUMERIC_FEATURES = [f for f in ALL_FEATURES if f not in CATEGORICAL_FEATURES]


def load_question_meta(datasets: list[str]) -> dict[str, dict]:
    """template_id -> {kind, n_tables, answer_type} from the question banks."""
    meta: dict[str, dict] = {}
    for ds in datasets:
        path = QUESTION_BANK_DIR / f"{ds}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"question bank for {ds} missing — run scripts/build_question_banks.py")
        bank = json.loads(path.read_text())
        for q in bank["questions"]:
            meta[q["template_id"]] = {
                "kind": q["kind"],
                "n_tables": q.get("n_tables", 1),
                "answer_type": q["answer_type"],
            }
    return meta


def features_for_run(run: dict, meta: dict[str, dict]) -> dict:
    """One raw run dict -> one feature row (no label)."""
    tpl = meta.get(run["template_id"], {})
    row = {k: run.get(k) for k in META_COLUMNS}
    row.update(static_features(run["dataset"], tpl.get("kind", "unknown"),
                               tpl.get("n_tables", 1)))
    row.update(dynamic_features(run))
    row.update(graph_features(run))
    return row


def build_feature_frame(runs_path: str | Path,
                        labelled_path: str | Path | None = None) -> pd.DataFrame:
    """Build the modelling frame from the run corpus + the labelled table.

    `labelled_path` defaults to the sibling labelled.parquet written by the
    labelling pipeline. The label is joined on run_id so a run missing from the
    labelled table is dropped loudly rather than silently mislabelled.
    """
    runs = list(read_jsonl(runs_path))
    if not runs:
        raise ValueError(f"no runs in {runs_path}")
    datasets = sorted({r["dataset"] for r in runs})
    meta = load_question_meta(datasets)

    frame = pd.DataFrame([features_for_run(r, meta) for r in runs])

    labelled_path = (Path(labelled_path) if labelled_path
                     else Path(runs_path).with_name("labelled.parquet"))
    if not Path(labelled_path).exists():
        raise FileNotFoundError(
            f"labelled table {labelled_path} missing — run scripts/label_runs.py")
    labels = pd.read_parquet(labelled_path)[["run_id", "label_failure", "timed_out"]]

    merged = frame.merge(labels, on="run_id", how="inner", validate="one_to_one")
    if len(merged) != len(frame):
        dropped = sorted(set(frame["run_id"]) - set(merged["run_id"]))
        raise ValueError(f"{len(dropped)} runs have no label, e.g. {dropped[:3]}")
    return merged


def xy_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Split the frame into (X over the schema's features, y, grouping key).

    The grouping key is template_id: every paraphrase of a question, and every
    topology that answered it, stays on one side of the split, so a near-copy of
    a training question can never appear in test.
    """
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"feature frame missing columns: {missing}")
    return df[ALL_FEATURES].copy(), df["label_failure"].astype(int), df["template_id"]


def feature_summary(df: pd.DataFrame) -> str:
    """Human-readable sanity report — catches all-zero or constant blocks."""
    lines = [f"rows: {len(df)}   features: {len(ALL_FEATURES)}", ""]
    for block, names in (("STATIC", STATIC_FEATURES), ("DYNAMIC", DYNAMIC_FEATURES),
                         ("GRAPH", GRAPH_FEATURES)):
        lines.append(f"{block}:")
        for name in names:
            col = df[name]
            if not pd.api.types.is_numeric_dtype(col):
                lines.append(f"  {name:22s} {col.nunique()} levels: "
                             f"{sorted(col.dropna().unique())[:6]}")
            else:
                flag = "  <-- constant" if col.nunique() <= 1 else ""
                lines.append(f"  {name:22s} mean={col.mean():10.3f} "
                             f"min={col.min():9.3f} max={col.max():10.3f}{flag}")
        lines.append("")
    return "\n".join(lines)
