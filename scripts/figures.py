"""Report figures from the results/ tables (run after scripts/relate.py).

Writes PNGs to results/figures/. Topology colours are fixed by entity, never by
rank, and every figure has direct labels or a table beside it in the report.
"""
import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

TOPO = {  # entity -> (label, colour); validated categorical slots 1-3
    "single_react": ("Single ReAct", "#2a78d6"),
    "orchestrator_worker": ("Orchestrator-worker", "#eb6834"),
    "peer_tool_graph": ("Peer tool-graph", "#1baf7a"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FAULT = {"tool_failure": "Tool failure", "corrupted_output": "Corrupted output",
         "dropped_context": "Dropped context"}
DATASET = {"titanic": "Titanic", "adult_census": "Adult Census", "nyc_taxi": "NYC taxi",
           "missingness": "Missingness", "olist": "Olist"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200,
    "savefig.bbox": "tight", "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
})


def _dots(ax, cats, value, lo=None, hi=None, frame=None, xlabel=""):
    """Dot plot: one row per category, one dot per topology, offset vertically."""
    for k, (topo, (label, colour)) in enumerate(TOPO.items()):
        d = frame[frame["topology"] == topo].set_index("cat").reindex(cats)
        y = [i + (k - 1) * 0.22 for i in range(len(cats))]
        if lo is not None:
            ax.hlines(y, d[lo], d[hi], color=colour, linewidth=2, alpha=0.55)
        ax.plot(d[value], y, "o", color=colour, markersize=7, label=label,
                markeredgecolor="#fcfcfb", markeredgewidth=1.5, zorder=3)
    ax.set_yticks(range(len(cats)), cats)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              handletextpad=0.3, columnspacing=1.2)


def failure_by_dataset(res: Path, out: Path) -> None:
    lab = pd.read_parquet(res / "labelled.parquet")
    t = lab.groupby(["dataset", "topology"])["label_failure"].mean().reset_index()
    t["cat"] = t["dataset"].map(DATASET)
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    _dots(ax, [DATASET[d] for d in DATASET], "label_failure", frame=t,
          xlabel="Failure rate (baseline corpus)")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    fig.savefig(out / "failure_by_dataset.png")


def robustness(res: Path, out: Path) -> None:
    r = pd.read_csv(res / "robustness.csv")
    r["cat"] = r["fault"].map(FAULT)
    fig, ax = plt.subplots(figsize=(6.2, 2.4))
    ax.axvline(0, color=MUTED, linewidth=1)
    _dots(ax, list(FAULT.values()), "delta", "ci_lo", "ci_hi", frame=r,
          xlabel="Delta (faulted minus baseline failure rate), 95% cluster-bootstrap CI")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:+.0%}")
    fig.savefig(out / "robustness_delta.png")


def fault_recall(res: Path, out: Path) -> None:
    d = pd.read_csv(res / "relate_fault_detection.csv")
    d["cat"] = d["fault"].map(FAULT)
    fig, ax = plt.subplots(figsize=(6.2, 2.4))
    _dots(ax, list(FAULT.values()), "recall", frame=d,
          xlabel="Share of fault-induced failures the predictor flags (recall)")
    ax.set_xlim(0, 1.02)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    fig.savefig(out / "fault_recall.png")


def pr_curves(res: Path, out: Path) -> None:
    o = pd.read_csv(res / "oof_predictions.csv")
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    for topo, (label, colour) in TOPO.items():
        s = o[o["topology"] == topo]
        prec, rec, _ = precision_recall_curve(s["label_failure"], s["p_failure_oof"])
        ax.plot(rec, prec, color=colour, linewidth=2, label=label)
        ax.axhline(s["label_failure"].mean(), color=colour, linewidth=1, linestyle=":")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision")
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.01, 0.03, "dotted: each topology's failure rate (a random ranker)",
            transform=ax.transAxes, color=MUTED, fontsize=7.5)
    fig.savefig(out / "pr_curves.png")


def calibration(res: Path, out: Path) -> None:
    """Reliability diagram: observed silent-failure rate per bin of p-hat."""
    o = pd.read_csv(res / "oof_predictions.csv")
    o["bin"] = (o["p_failure_oof"] * 10).astype(int).clip(upper=9)
    g = o.groupby("bin").agg(p=("p_failure_oof", "mean"), obs=("label_failure", "mean"),
                             n=("label_failure", "size"))
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=":")
    ax.plot(g["p"], g["obs"], "-o", color="#2a78d6", linewidth=2, markersize=6,
            markeredgecolor="#fcfcfb", markeredgewidth=1.5)
    for _, r in g.iterrows():
        ax.annotate(f"{int(r['n'])}", (r["p"], r["obs"]), textcoords="offset points",
                    xytext=(5, -10), fontsize=7, color=MUTED)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted P(silent failure)",
           ylabel="Observed silent-failure rate")
    ax.text(0.02, 0.93, "dotted: perfect calibration; labels: runs per bin",
            transform=ax.transAxes, color=MUTED, fontsize=7.5)
    fig.savefig(out / "calibration.png")


def review_budget(res: Path, out: Path) -> None:
    """Share of silent failures caught when only the top-k% of runs is re-checked."""
    o = pd.read_csv(res / "oof_predictions.csv").sort_values("p_failure_oof", ascending=False)
    caught = o["label_failure"].cumsum().to_numpy() / o["label_failure"].sum()
    share = [(i + 1) / len(o) for i in range(len(o))]
    rate = o["label_failure"].mean()
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.plot(share, share, color=MUTED, linewidth=1, linestyle=":", label="Random order")
    ax.plot(share, [min(1.0, s / rate) for s in share], color=MUTED, linewidth=1,
            linestyle="--", label="Perfect ranking")
    ax.plot(share, caught, color="#2a78d6", linewidth=2, label="Ranked by p-hat")
    for k in (0.1, 0.2, 0.3):
        v = caught[max(0, round(k * len(o)) - 1)]
        ax.plot(k, v, "o", color="#2a78d6", markersize=6, markeredgecolor="#fcfcfb")
        ax.annotate(f"{v:.0%}", (k, v), textcoords="offset points", xytext=(4, -12), fontsize=7.5)
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Share of runs re-checked by an analyst",
           ylabel="Share of silent failures caught")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(out / "review_budget.png")


def importance(res: Path, out: Path) -> None:
    imp = pd.read_csv(res / "feature_importance.csv").head(10)[::-1]
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.barh(imp["feature"], imp["gain"] / imp["gain"].sum(), color="#2a78d6", height=0.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of LightGBM gain (top 10 features)")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    fig.savefig(out / "feature_importance.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    res = Path(args.results)
    out = res / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for fig in (failure_by_dataset, robustness, fault_recall, pr_curves, importance,
                calibration, review_budget):
        fig(res, out)
        plt.close("all")
        print(f"wrote {out / fig.__name__}")


if __name__ == "__main__":
    main()
