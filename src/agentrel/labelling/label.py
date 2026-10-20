"""Correctness labelling — report Eq. 1.

    y(tau) = 1[ |a - a*| > epsilon ]     (1 == FAILURE, the minority positive class)

For numeric answers we use a relative tolerance epsilon (default 1%). For
categorical answers we use exact match after normalisation. A run whose answer
cannot be parsed, or whose ground truth errored, is labelled a failure (an
unusable answer is a failed answer) and flagged so it can be inspected.

Also provides the paraphrase-consistency signal (report 3.3): if the parsed
answers of the paraphrases of one (template, topology) disagree, at least one is
wrong — a label that needs no ground truth.
"""
from __future__ import annotations

from collections import defaultdict

from .parse import parse_categorical, parse_numeric


def _numeric_fail(pred: float | None, truth: float, eps: float) -> bool:
    if pred is None:
        return True
    denom = max(abs(truth), 1e-9)
    return abs(pred - truth) / denom > eps


def label_run(run: dict, truth_entry: dict, eps: float) -> dict:
    """Return a label record for one run. Does not mutate `run`."""
    answer_type = truth_entry.get("answer_type", "numeric")
    truth_val = truth_entry.get("value")
    raw = run.get("final_answer_raw")

    reason = None
    if truth_entry.get("error"):
        failed, parsed = True, None
        reason = "ground_truth_error"
    elif answer_type == "numeric":
        parsed = parse_numeric(raw)
        try:
            failed = _numeric_fail(parsed, float(truth_val), eps)
        except (TypeError, ValueError):
            failed, reason = True, "unparseable_truth"
        if parsed is None:
            reason = "unparseable_answer"
    else:
        parsed = parse_categorical(raw)
        truth_norm = parse_categorical(str(truth_val))
        failed = (parsed is None) or (parsed != truth_norm)
        if parsed is None:
            reason = "unparseable_answer"

    # an unusable run (harness error / timeout with empty answer) is a failure
    if run.get("harness_error"):
        failed, reason = True, reason or "harness_error"

    return {
        "run_id": run["run_id"],
        "dataset": run["dataset"],
        "template_id": run["template_id"],
        "question_id": run["question_id"],
        "topology": run["topology"],
        "paraphrase_index": run["paraphrase_index"],
        "answer_type": answer_type,
        "truth_value": truth_val,
        "parsed_answer": parsed,
        "raw_answer": raw,
        "timed_out": run.get("timed_out", False),
        "label_failure": int(bool(failed)),   # 1 == silent/overt failure
        "label_reason": reason,
    }


def paraphrase_consistency(labels: list[dict]) -> dict[tuple[str, str], bool]:
    """For each (template_id, topology), True if all paraphrase answers agree.

    Disagreement is a ground-truth-free signal that at least one paraphrase
    failed. Returns {(template_id, topology): is_consistent}."""
    groups: dict[tuple[str, str], list] = defaultdict(list)
    for lab in labels:
        groups[(lab["template_id"], lab["topology"])].append(lab["parsed_answer"])
    out = {}
    for key, answers in groups.items():
        present = [a for a in answers if a is not None]
        # round numerics lightly so float noise doesn't count as disagreement
        norm = set()
        for a in present:
            norm.add(round(a, 6) if isinstance(a, (int, float)) else a)
        out[key] = (len(norm) <= 1) and (len(present) == len(answers))
    return out
