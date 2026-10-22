"""Build the LaTeX edition of the final report (the midsem report's format).

    python report/build_tex.py            # -> Shaurya_Pragun_FDS_Project_Report_LaTeX.pdf

Writes report/latex/generated.tex: every number and table body as a macro, read
from results/, plus the interpretive prose from report/findings.json converted
to LaTeX. The document itself, report/latex/Shaurya_Pragun_FDS_Project_Report.tex,
only \\input's it, so the two report editions share one source of numbers and
prose. Compiles with tectonic when it is on PATH; otherwise upload report/latex/
plus results/figures/ to any TeX distribution.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from corpus_stats import FAULT, MODEL, TOPO, extended, failure_kind, flagged_share

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TEX = ROOT / "report" / "latex"
NAME = "Shaurya_Pragun_FDS_Project_Report"

_ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
        "{": r"\{", "}": r"\}"}
_SYM = {"p̂": r"$\hat{p}$", "θ": r"$\theta$", "Δ": r"$\Delta$", "ε": r"$\varepsilon$", "≥": r"$\geq$",
        "×": r"$\times$", "−": "-"}


def tex(text: str) -> str:
    """Escape plain text and turn the findings' <b>/<i> markup into LaTeX."""
    out = "".join(_ESC.get(c, c) for c in str(text))
    for k, v in _SYM.items():
        out = out.replace(k, v)
    out = re.sub(r"<b>(.*?)</b>", r"\\textbf{\1}", out)
    out = re.sub(r"<i>(.*?)</i>", r"\\emph{\1}", out)
    return re.sub(r"(^|[\s(])'", r"\1`", out)  # opening single quotes


def pct(x, nd=0):
    return f"{x * 100:.{nd}f}\\%"


def signed(x):
    return f"{x * 100:+.0f}"


def main() -> None:
    lab = pd.read_parquet(RES / "labelled.parquet")
    labf = pd.read_parquet(RES / "runs_faults_labelled.parquet")
    clf = json.loads((RES / "classifier_results.json").read_text())
    rob = pd.read_csv(RES / "robustness.csv")
    rel = json.loads((RES / "relate.json").read_text())
    summary = pd.DataFrame(rel["summary"])
    oof = pd.read_csv(RES / "oof_predictions.csv")
    fin = json.loads((ROOT / "report" / "findings.json").read_text())

    kinds = lab.apply(failure_kind, axis=1).value_counts()
    n_runs, n_fail = len(lab), int(lab["label_failure"].sum())
    n_silent = int(kinds.filter(like="silent").sum())
    by_topo = lab.groupby("topology")["label_failure"].mean()
    theta = rel["threshold_f2_oof"]
    ext, ext_ci = extended(oof, theta)
    test, split = clf["test_metrics"], clf["split"]
    mean_delta = summary.set_index("topology")["mean_delta"]

    def ci(k):
        lo, hi = ext_ci[k]
        return f"[{lo:.2f}, {hi:.2f}]"

    m = {  # macro name -> value (LaTeX)
        "nRuns": f"{n_runs:,}", "nFaultRuns": f"{len(labf):,}", "nFail": n_fail,
        "failRate": pct(n_fail / n_runs, 1), "nSilent": n_silent, "silentShare": pct(n_silent / n_fail),
        "nVisible": int(kinds.get("visible", 0)), "nTests": fin["n_tests"],
        "failSingle": pct(by_topo["single_react"]), "failOrch": pct(by_topo["orchestrator_worker"]),
        "failPeer": pct(by_topo["peer_tool_graph"]),
        "realised": pct(labf["fault_realised"].mean()),
        "testN": split["sizes"]["test"], "testGroups": split["n_groups"]["test"],
        "testRate": pct(clf["class_balance"]["test"]),
        "thetaVal": f"{test['lgbm_weighted']['threshold']:.2f}", "thetaOOF": f"{theta:.2f}",
        "testPRAUC": f"{test['lgbm_weighted']['pr_auc']:.2f}",
        "oofN": len(oof), "oofRate": pct(oof["label_failure"].mean()),
        "oofPRAUC": f"{ext['pr_auc']:.2f}", "oofROC": f"{ext['roc_auc']:.2f}",
        "oofPRAUCci": ci("pr_auc"), "oofECE": f"{ext['ece']:.2f}",
        "oofFlagged": pct(flagged_share(oof, theta)),
        "recallTen": pct(ext["recall_at_10pct"]), "recallTwenty": pct(ext["recall_at_20pct"]),
        "recallThirty": pct(ext["recall_at_30pct"]),
        "deltaSingle": signed(mean_delta["single_react"]),
        "deltaOrch": signed(mean_delta["orchestrator_worker"]),
        "deltaPeer": signed(mean_delta["peer_tool_graph"]),
    }

    rows = lambda rs: " \\\\\n".join(" & ".join(str(c) for c in r) for r in rs) + " \\\\"  # noqa: E731
    m["tabBenchmark"] = rows([[tex(b["label"]), pct(b["fail"]), pct(b["timeout"]), pct(b["zero_tool"]),
                               f"{b['sec']:.0f}"] for b in fin["benchmark"]])
    m["tabOutcomes"] = rows([[f"\\SFcap: {label}", int(kinds.get(k, 0)), pct(kinds.get(k, 0) / n_runs), what]
                             for k, label, what in (
                                 ("silent_value", "wrong value ($>$5\\% off)", "a confident, plausible, wrong number"),
                                 ("silent_near", "near miss (1--5\\% off)", "a number off by rounding or a slip"),
                                 ("silent_category", "wrong category", "a confident wrong label"))]
                            + [["Visible failure", int(kinds.get("visible", 0)), pct(kinds.get("visible", 0) / n_runs),
                                "no usable answer"],
                               ["Correct", int(kinds.get("correct", 0)), pct(kinds.get("correct", 0) / n_runs),
                                "the right answer"]])
    m["tabClassifier"] = rows([[("\\textbf{%s}" % label) if k == "lgbm_weighted" else label,
                                f"{test[k]['f2']:.2f}", f"{test[k]['pr_auc']:.2f}", f"{test[k]['brier']:.3f}",
                                f"{test[k]['precision']:.2f}", f"{test[k]['recall']:.2f}"]
                               for k, label in MODEL.items()])
    m["tabExtended"] = rows([
        ["PR-AUC", f"{ext['pr_auc']:.2f}", ci("pr_auc")],
        ["ROC-AUC", f"{ext['roc_auc']:.2f}", ci("roc_auc")],
        [f"F2 at $\\theta$ = {theta:.2f}", f"{ext['f2']:.2f}", ci("f2")],
        ["MCC", f"{ext['mcc']:.2f}", ""],
        ["Balanced accuracy", f"{ext['balanced_accuracy']:.2f}", ""],
        ["Brier", f"{ext['brier']:.3f}", ""],
        ["ECE (10 bins)", f"{ext['ece']:.3f}", ci("ece")],
        ["Recall@10/20/30\\%", f"{pct(ext['recall_at_10pct'])} / {pct(ext['recall_at_20pct'])} / "
                               f"{pct(ext['recall_at_30pct'])}", "20\\%: " + ci("recall_at_20pct")],
    ])
    m["tabRobustness"] = rows([[TOPO[r["topology"]], FAULT[r["fault"]], pct(r["baseline_failure_rate"]),
                                pct(r["faulted_failure_rate"]), signed(r["delta"]),
                                f"[{signed(r['ci_lo'])}, {signed(r['ci_hi'])}]", "*" if r["significant"] else ""]
                               for _, r in rob.iterrows()])
    m["tabSummary"] = rows([[TOPO[r["topology"]], signed(r["mean_delta"]), pct(r["failure_rate"]),
                             f"{r['pr_auc']:.2f} ({r['lift']:.1f}$\\times$)", pct(r["fault_recall"]),
                             pct(r["corrupted_output_recall"]), r["main_driver"]]
                            for _, r in summary.iterrows()])
    for key in ("abstract", "model_choice", "failure_rates", "classifier", "metrics", "robustness",
                "relate", "conclusion"):
        m["F" + "".join(w.title() for w in key.split("_"))] = tex(fin[key])
    for key in ("discussion", "limitations"):
        m["F" + key.title()] = "\n".join(f"\\item {tex(t)}" for t in fin[key])

    lines = ["% Generated by report/build_tex.py from results/ and report/findings.json. Do not edit."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()]
    (TEX / "generated.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {TEX / 'generated.tex'} ({len(m)} macros)")

    if not shutil.which("tectonic"):
        sys.exit("tectonic not found: compile report/latex/ with any TeX distribution")
    subprocess.run(["tectonic", "-X", "compile", f"{NAME}.tex"], cwd=TEX, check=True)
    out = ROOT / f"{NAME}_LaTeX.pdf"
    shutil.move(TEX / f"{NAME}.pdf", out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
