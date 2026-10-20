"""Compute the deterministic reference answer for every template, per dataset.

This is the ground truth that agent answers are scored against (report Eq. 1).
Nothing here touches an LLM — it is pure pandas. The output is a dict keyed by
template_id, so every paraphrase of a template inherits the same truth value.
"""
from __future__ import annotations

from .questions import templates_for
from .registry import load_tables


def compute_ground_truth(dataset: str) -> dict[str, dict]:
    tables = load_tables(dataset)
    out: dict[str, dict] = {}
    for tpl in templates_for(dataset):
        try:
            value = tpl.truth(tables)
            error = None
        except Exception as exc:  # noqa: BLE001 — surface, don't crash the whole run
            value = None
            error = f"{exc.__class__.__name__}: {exc}"
        out[tpl.id] = {
            "template_id": tpl.id,
            "dataset": dataset,
            "kind": tpl.kind,
            "answer_type": tpl.answer_type,
            "value": value,
            "error": error,
        }
    return out


def compute_all_ground_truth(datasets: list[str]) -> dict[str, dict]:
    truth: dict[str, dict] = {}
    for ds in datasets:
        truth.update(compute_ground_truth(ds))
    return truth
