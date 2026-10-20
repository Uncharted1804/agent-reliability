"""Log cleaning (block B).

The run logs are the messy part (report 3.3). This module:
  * flattens each RunTrace dict into one row of trace-level numeric fields,
  * keeps (does NOT drop) timed-out / truncated runs, flagging them, since a
    timeout is itself informative,
  * caps heavy-tailed counts (tokens, latency, tool calls) with the 1.5*IQR
    rule so a few runaway runs don't dominate downstream scaling.
Messiness in the *source datasets* is deliberately left alone — that is signal
the project wants to measure, not noise.
"""
from __future__ import annotations

import pandas as pd

# columns that are heavy-tailed and get IQR-capped (winsorised at the upper fence)
_HEAVY_TAIL = ["n_tool_calls", "n_retries", "n_errors", "n_llm_calls",
               "prompt_tokens", "completion_tokens", "wall_time_s"]


def flatten_run(run: dict) -> dict:
    """One raw run dict -> one flat row of run-level fields (pre-feature)."""
    return {
        "run_id": run["run_id"],
        "dataset": run["dataset"],
        "template_id": run["template_id"],
        "question_id": run["question_id"],
        "topology": run["topology"],
        "paraphrase_index": run["paraphrase_index"],
        "model": run.get("model"),
        "n_tool_calls": run.get("n_tool_calls", 0),
        "n_retries": run.get("n_retries", 0),
        "n_errors": run.get("n_errors", 0),
        "n_llm_calls": run.get("n_llm_calls", 0),
        "prompt_tokens": run.get("prompt_tokens", 0),
        "completion_tokens": run.get("completion_tokens", 0),
        "wall_time_s": run.get("wall_time_s", 0.0),
        "timed_out": int(bool(run.get("timed_out", False))),
        "harness_error": int(bool(run.get("harness_error"))),
    }


def cap_heavy_tails(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Winsorise heavy-tailed count columns at the 1.5*IQR upper fence.

    Adds a `<col>_capped` flag column recording which rows were clipped, so the
    capping is auditable rather than silent."""
    cols = cols or _HEAVY_TAIL
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        hi = q3 + 1.5 * iqr
        if iqr == 0:
            df[f"{col}_capped"] = 0
            continue
        df[f"{col}_capped"] = (df[col] > hi).astype(int)
        df[col] = df[col].clip(upper=hi)
    return df
