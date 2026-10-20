"""Labelling + cleaning pipeline (block B end-to-end).

Takes the raw run corpus (runs.jsonl) + the question banks' ground truth, and
produces one clean labelled table (one row per run) ready for feature
engineering. Also emits the paraphrase-consistency table as a secondary,
ground-truth-free check.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..utils.config import REPO_ROOT
from ..utils.io import read_jsonl
from .clean import cap_heavy_tails, flatten_run
from .label import label_run, paraphrase_consistency

QUESTION_BANK_DIR = REPO_ROOT / "datasets" / "question_banks"


def _load_all_truth(datasets: list[str]) -> dict[str, dict]:
    truth: dict[str, dict] = {}
    for ds in datasets:
        bank = json.loads((QUESTION_BANK_DIR / f"{ds}.json").read_text())
        truth.update(bank["ground_truth"])
    return truth


def default_label_path(runs_path: str | Path) -> Path:
    """Where the labelled table for `runs_path` goes.

    The baseline corpus keeps the documented results/labelled.parquet, while any
    other corpus (notably runs_faults.jsonl) gets its own <stem>_labelled.parquet.
    Deriving one fixed name for every input would make labelling the fault corpus
    silently overwrite the baseline labels, which the robustness study needs.
    """
    runs_path = Path(runs_path)
    if runs_path.stem == "runs":
        return runs_path.with_name("labelled.parquet")
    return runs_path.with_name(f"{runs_path.stem}_labelled.parquet")


def run_pipeline(runs_path: str | Path, eps: float,
                 out_path: str | Path | None = None) -> pd.DataFrame:
    runs = list(read_jsonl(runs_path))
    datasets = sorted({r["dataset"] for r in runs})
    truth = _load_all_truth(datasets)

    labels = [label_run(r, truth.get(r["template_id"], {}), eps) for r in runs]
    flat = {r["run_id"]: flatten_run(r) for r in runs}

    # merge labels onto flattened run rows
    rows = []
    for lab in labels:
        row = dict(flat[lab["run_id"]])
        row.update({
            "answer_type": lab["answer_type"],
            "label_failure": lab["label_failure"],
            "label_reason": lab["label_reason"],
            "parsed_answer": lab["parsed_answer"],
            "truth_value": lab["truth_value"],
        })
        rows.append(row)
    df = pd.DataFrame(rows)

    # paraphrase consistency -> per-run boolean flag (secondary signal)
    consist = paraphrase_consistency(labels)
    df["paraphrase_consistent"] = df.apply(
        lambda r: int(consist.get((r["template_id"], r["topology"]), True)), axis=1)

    # parsed_answer and truth_value mix numeric + categorical across rows; store
    # as strings so the corpus serialises cleanly (they are records, not features)
    df["injected_fault"] = df["injected_fault"].astype("string")
    df["parsed_answer"] = df["parsed_answer"].astype("string")
    df["truth_value"] = df["truth_value"].astype("string")
    df["label_reason"] = df["label_reason"].astype("string")

    df = cap_heavy_tails(df)

    out_path = Path(out_path) if out_path else default_label_path(runs_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    df.to_csv(out_path.with_suffix(".csv"), index=False)  # human-inspectable copy
    return df


def summarise(df: pd.DataFrame) -> str:
    n = len(df)
    fail = int(df["label_failure"].sum())
    lines = [
        f"runs:                {n}",
        f"failures (label=1):  {fail}  ({fail/n:.1%})",
        f"correct  (label=0):  {n - fail}  ({(n-fail)/n:.1%})",
        f"timed out:           {int(df['timed_out'].sum())}",
        f"paraphrase-inconsistent groups present in: "
        f"{int((df['paraphrase_consistent']==0).sum())} runs",
        "",
        "failure rate by topology:",
    ]
    for topo, g in df.groupby("topology"):
        lines.append(f"  {topo:22s} {g['label_failure'].mean():.1%}  (n={len(g)})")
    lines.append("")
    lines.append("label reasons:")
    for reason, c in df["label_reason"].value_counts(dropna=False).items():
        lines.append(f"  {str(reason):22s} {c}")
    return "\n".join(lines)
