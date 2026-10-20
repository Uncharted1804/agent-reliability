"""Topology robustness Delta_{T,f} with bootstrap CIs (block F).

For topology T and fault f:

    Delta_{T,f} = P(failure | T, f) - P(failure | T, no fault)

the increase in failure rate caused by breaking one component. Larger means the
topology degrades more; a value near zero means the topology neutralised the
fault.

Three things make the estimate defensible:

* **Matched questions.** Baseline runs cover all 5 phrasings of every template
  while fault runs may cover fewer, so the baseline is restricted to exactly the
  question_ids that were faulted. Comparing against the unmatched baseline would
  confound Delta with which questions happened to be re-run.
* **Realised faults only.** A run that answered before reaching its target step
  was never actually faulted. Those runs are dropped, and the realisation rate
  is reported, because counting them as faulted would shrink Delta toward zero.
* **Cluster bootstrap.** Resampling individual runs would treat the 5
  paraphrases of a template as 5 independent observations when they are near
  duplicates, giving CIs that are too narrow. Templates are resampled instead
  (with replacement), carrying all their runs — so the CI reflects uncertainty
  over questions, which is the population the result generalises to.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CI_LEVEL = 0.95


def _rate(df: pd.DataFrame) -> float:
    return float(df["label_failure"].mean()) if len(df) else float("nan")


def prepare(baseline: pd.DataFrame, faulted: pd.DataFrame) -> tuple[pd.DataFrame,
                                                                    pd.DataFrame, dict]:
    """Restrict both corpora to matched questions and realised faults."""
    faulted = faulted.copy()
    if "fault_realised" in faulted.columns:
        n_before = len(faulted)
        faulted = faulted[faulted["fault_realised"].astype(bool)]
        realisation = float(len(faulted) / n_before) if n_before else float("nan")
    else:  # pragma: no cover — only for corpora from before the fault layer
        realisation = float("nan")

    shared = sorted(set(baseline["question_id"]) & set(faulted["question_id"]))
    base = baseline[baseline["question_id"].isin(shared)]
    flt = faulted[faulted["question_id"].isin(shared)]
    info = {
        "n_matched_questions": len(shared),
        "fault_realisation_rate": realisation,
        "n_baseline_runs": int(len(base)),
        "n_faulted_runs": int(len(flt)),
    }
    return base, flt, info


def delta(baseline: pd.DataFrame, faulted: pd.DataFrame) -> float:
    """Point estimate of Delta for one (topology, fault) cell."""
    return _rate(faulted) - _rate(baseline)


def bootstrap_ci(baseline: pd.DataFrame, faulted: pd.DataFrame, *,
                 n_resamples: int = 1000, seed: int = 42,
                 cluster: str = "template_id") -> tuple[float, float, float]:
    """Cluster-bootstrap CI for Delta. Returns (lo, hi, std_error)."""
    clusters = np.array(sorted(set(baseline[cluster]) | set(faulted[cluster])))
    if len(clusters) == 0:
        return float("nan"), float("nan"), float("nan")

    base_by = {k: v for k, v in baseline.groupby(cluster)}
    flt_by = {k: v for k, v in faulted.groupby(cluster)}

    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_resamples):
        picked = rng.choice(clusters, size=len(clusters), replace=True)
        b = [base_by[c] for c in picked if c in base_by]
        f = [flt_by[c] for c in picked if c in flt_by]
        if not b or not f:
            continue
        estimates.append(_rate(pd.concat(f)) - _rate(pd.concat(b)))

    if not estimates:  # pragma: no cover — needs an empty cell
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(estimates, dtype=float)
    alpha = (1 - CI_LEVEL) / 2
    return (float(np.quantile(arr, alpha)), float(np.quantile(arr, 1 - alpha)),
            float(arr.std(ddof=1)) if len(arr) > 1 else float("nan"))


def robustness_table(baseline: pd.DataFrame, faulted: pd.DataFrame, *,
                     n_resamples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Delta_{T,f} with CIs for every (topology, fault) cell."""
    base, flt, _ = prepare(baseline, faulted)
    rows = []
    for topo in sorted(flt["topology"].unique()):
        b_topo = base[base["topology"] == topo]
        for fault in sorted(flt["injected_fault"].dropna().unique()):
            f_cell = flt[(flt["topology"] == topo) & (flt["injected_fault"] == fault)]
            lo, hi, se = bootstrap_ci(b_topo, f_cell, n_resamples=n_resamples, seed=seed)
            rows.append({
                "topology": topo,
                "fault": fault,
                "n_baseline": int(len(b_topo)),
                "n_faulted": int(len(f_cell)),
                "baseline_failure_rate": _rate(b_topo),
                "faulted_failure_rate": _rate(f_cell),
                "delta": delta(b_topo, f_cell),
                "ci_lo": lo,
                "ci_hi": hi,
                "std_error": se,
                # a CI excluding 0 means the degradation is distinguishable
                # from noise at this corpus size
                "significant": bool(not np.isnan(lo) and (lo > 0 or hi < 0)),
            })
    return pd.DataFrame(rows)


def mean_degradation(table: pd.DataFrame) -> pd.DataFrame:
    """Mean Delta per topology across faults — the headline robustness ranking."""
    out = (table.groupby("topology")
           .agg(mean_delta=("delta", "mean"),
                worst_fault=("delta", "max"),
                n_cells=("delta", "size"))
           .reset_index()
           .sort_values("mean_delta"))
    return out


def format_robustness(table: pd.DataFrame, info: dict | None = None) -> str:
    """Render the robustness table as the text block that goes in the report."""
    lines = []
    if info:
        lines += [
            f"matched questions: {info['n_matched_questions']}   "
            f"baseline runs: {info['n_baseline_runs']}   "
            f"faulted runs (realised): {info['n_faulted_runs']}",
            f"fault realisation rate: {info['fault_realisation_rate']:.1%}",
            "",
        ]
    header = (f"{'topology':22s} {'fault':18s} {'base':>6s} {'faulted':>8s} "
              f"{'delta':>7s} {'95% CI':>18s}  sig")
    lines += [header, "-" * len(header)]
    for _, r in table.iterrows():
        ci = f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"
        lines.append(f"{r['topology']:22s} {r['fault']:18s} "
                     f"{r['baseline_failure_rate']:6.3f} {r['faulted_failure_rate']:8.3f} "
                     f"{r['delta']:+7.3f} {ci:>18s}  "
                     f"{'*' if r['significant'] else ''}")
    lines += ["", "mean degradation per topology (lower = more robust):"]
    for _, r in mean_degradation(table).iterrows():
        lines.append(f"  {r['topology']:22s} mean delta {r['mean_delta']:+.3f}   "
                     f"worst fault {r['worst_fault']:+.3f}")
    return "\n".join(lines)
