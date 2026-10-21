"""Build the final project report PDF from the results/ tables and figures.

    python report/build.py                     # -> report/final_report.pdf
    python report/build.py out.pdf results_dir  # another corpus

Every number in the results sections is read from results/, so rerunning the
pipeline and this script keeps the report and the data in step. The prose in
FINDINGS states what the numbers mean; it is checked against them by hand.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "report" / "final_report.pdf"
RES = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results"
FIG = RES / "figures"

INK, ACC, MUTED, RULE, HEAD = (colors.HexColor(c) for c in
                               ("#1f2933", "#2457a6", "#5b6670", "#c8ced6", "#e6edf7"))
ss = getSampleStyleSheet()
TITLE = ParagraphStyle("t", parent=ss["Title"], fontSize=17, leading=21, textColor=INK)
SUB = ParagraphStyle("s", parent=ss["Normal"], fontSize=9.5, leading=13, textColor=MUTED, alignment=1)
H1 = ParagraphStyle("h1", parent=ss["Heading2"], fontSize=13, textColor=ACC, spaceBefore=10, spaceAfter=4)
H2 = ParagraphStyle("h2", parent=ss["Heading3"], fontSize=10.5, textColor=INK, spaceBefore=6, spaceAfter=2)
B = ParagraphStyle("b", parent=ss["Normal"], fontSize=9.5, leading=13.2, textColor=INK, spaceAfter=4)
SM = ParagraphStyle("sm", parent=B, fontSize=8.3, leading=10.8, textColor=MUTED)
CELL = ParagraphStyle("c", parent=B, fontSize=8.3, leading=10.5, spaceAfter=0)
BUL = ParagraphStyle("bul", parent=B, leftIndent=12, bulletIndent=2, spaceAfter=2)
ABS = ParagraphStyle("abs", parent=B, leftIndent=8 * mm, rightIndent=8 * mm, fontSize=9.2)

story: list = []
pct = lambda x: f"{x * 100:.0f}%"
signed = lambda x: f"{x * 100:+.0f} pp"


def p(text, style=B):
    story.append(Paragraph(text, style))


def bullets(items):
    for t in items:
        story.append(Paragraph(t, BUL, bulletText="•"))


def table(rows, widths, caption=None):
    data = [[Paragraph(str(c), CELL) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, RULE), ("BACKGROUND", (0, 0), (-1, 0), HEAD),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]))
    block = [t]
    if caption:
        block = [Paragraph(caption, SM), Spacer(1, 2), t]
    story.append(KeepTogether(block + [Spacer(1, 6)]))


def figure(name, width_mm, caption):
    path = FIG / name
    img = Image(str(path))
    scale = width_mm * mm / img.drawWidth
    img.drawWidth, img.drawHeight = img.drawWidth * scale, img.drawHeight * scale
    story.append(KeepTogether([img, Spacer(1, 2), Paragraph(caption, SM), Spacer(1, 6)]))


TOPO = {"single_react": "Single ReAct", "orchestrator_worker": "Orchestrator-worker",
        "peer_tool_graph": "Peer tool-graph"}
FAULT = {"tool_failure": "Tool failure", "corrupted_output": "Corrupted output",
         "dropped_context": "Dropped context"}
MODEL = {"lgbm_weighted": "LightGBM, class-weighted (ours)", "smote_lgbm": "LightGBM + SMOTE",
         "logreg": "Logistic regression, class-weighted", "smote_logreg": "Logistic regression + SMOTE",
         "dummy_prior": "Prior (always the base rate)"}

# ------------------------------------------------------------------ data
lab = pd.read_parquet(RES / "labelled.parquet")
labf = pd.read_parquet(RES / "runs_faults_labelled.parquet")
clf = json.loads((RES / "classifier_results.json").read_text())
rob = pd.read_csv(RES / "robustness.csv")
rel = json.loads((RES / "relate.json").read_text())
summary = pd.DataFrame(rel["summary"])
detect = pd.DataFrame(rel["fault_detection"])
drivers = pd.DataFrame(rel["drivers"])
pred = pd.read_csv(RES / "relate_predictability.csv")
FINDINGS = json.loads((ROOT / "report" / "findings.json").read_text())

n_runs, fail_rate = len(lab), lab["label_failure"].mean()
by_topo = lab.groupby("topology")["label_failure"].mean()
realised = labf["fault_realised"].mean()
test = clf["test_metrics"]
ours = test["lgbm_weighted"]

# ------------------------------------------------------------------ title
p("Reliability and Failure Topology of Autonomous LLM Agents in Data Analysis Pipelines", TITLE)
p("Shaurya Mittal Nair (2410110316) &middot; Pragun Kathuria (2410110237)<br/>"
  "Department of Computer Science and Engineering, Shiv Nadar Institution of Eminence<br/>"
  "Final Project Report, Foundations of Data Science (CSD355) &middot; October 2026<br/>"
  "Code: github.com/Uncharted1804/agent-reliability", SUB)
story.append(Spacer(1, 8))
p("<b>Abstract.</b> " + FINDINGS["abstract"], ABS)

# ------------------------------------------------------------------ 1
p("1. Introduction", H1)
p("LLM agents can take a table and a question and run the whole analysis on their own. When they get it "
  "wrong they rarely crash: they return a clean, confident answer that happens to be incorrect, a "
  "<b>silent failure</b>. Ordinary monitoring sees no exception and no warning, so the only way to be sure is "
  "to redo the analysis, which defeats the point of the agent.")
p("We asked two questions in the midsem proposal and answer both here: (1) can features of the input data "
  "and of the agent's own execution trace predict a silent failure before the answer is used, and (2) does "
  "the structure of the agent's tool-interaction graph tell us which architectures hold up when a component "
  "breaks? One labelled corpus of agent runs feeds both, so the two answers can be compared directly "
  "(block G), which no prior work does for tabular-analysis agents [1-8].")

# ------------------------------------------------------------------ 2
p("2. What we built", H1)
p("The workflow follows Figure 1 of the proposal, blocks A to G. All seven are implemented in the "
  f"repository, with {FINDINGS['n_tests']} automated tests.")
table([
    ["Block", "What it does", "Where"],
    ["A. Run generator", ("Every (dataset, question, phrasing) through 3 topologies on one local model; "
     "full trace per run"), "harness/, topology/, agents/"],
    ["B. Labelling + cleaning", ("Answer parsed from free text and scored against a deterministic pandas "
     "answer (Eq. 1); logs flattened, timeouts kept and flagged, heavy tails IQR-capped"), "labelling/"],
    ["C. Features", "9 static, 7 dynamic and 3 graph features per run", "features/"],
    ["D. Classifier", "Class-weighted LightGBM and four baselines, 70/15/15 split grouped by template",
     "model/"],
    ["E. Fault injection", "One fault per run: tool failure, corrupted output, dropped context", "faults/"],
    ["F. Robustness", "Delta<sub>T,f</sub> with 1,000-resample cluster-bootstrap CIs", "faults/robustness.py"],
    ["G. Relate D and F", "Per-topology predictability, drivers and fault detection", "relate.py"],
], [34 * mm, 104 * mm, 36 * mm])

p("2.1 Corpus", H2)
p("Five tabular datasets of rising difficulty: Titanic (887 rows) and UCI Adult Census (32,561 rows), both "
  "downloaded; and three seeded synthetic stand-ins that reproduce the traits the proposal called for where "
  "no stable download exists: an NYC-taxi-style table (8,000 rows, outliers and mixed types), a "
  "heavy-missingness table (3,000 rows) and an Olist-style e-commerce schema of three tables that needs "
  "joins. Each dataset has 10 question templates (aggregation, filtering, group comparison, trend, "
  "multi-step), each asked in a canonical phrasing plus 4 paraphrases. "
  f"5 x 10 x 5 x 3 topologies gives <b>{n_runs} baseline runs</b>.")

p("2.2 Agents", H2)
p("Three topologies share one model, one tool registry and one prompt preamble, so only the wiring differs: "
  "<b>single ReAct</b> (one loop that retries once on a loud error; the pattern of ChatGPT's and Claude's "
  "data analysis), <b>orchestrator-worker</b> (a planner, a worker and an inspector that can demand a redo; "
  "like LAMBDA) and <b>peer tool-graph</b> (loader, transformer and reducer peers handing work around; closest "
  "to Data Interpreter). The agents act through eight sandboxed tools (load, filter, aggregate including "
  "quantiles, group-aggregate, join, missing rate, list tables) and see a schema preamble that names columns "
  "and types but never a quantity, so every number in an answer must come from a logged tool call.")

p("2.3 Base model", H2)
p(FINDINGS["model_choice"])
table([["Model", "Failure rate", "Hit the 12-step cap", "Used no tool", "Seconds / run"]] + [
    [m["label"], pct(m["fail"]), pct(m["timeout"]), pct(m["zero_tool"]), f"{m['sec']:.0f}"]
    for m in FINDINGS["benchmark"]],
    [52 * mm, 28 * mm, 34 * mm, 28 * mm, 30 * mm],
    caption="<b>Table 1.</b> Model benchmark: canonical phrasing of every template x 3 topologies, 150 runs a "
            "model, scored on the 41 templates unaffected by two harness issues found during the study "
            "(section 5).")

p("2.4 Labels, features, models", H2)
p("A numeric answer fails when it is more than 1% off the scripted value; a categorical one when it does not "
  "match exactly. Static features describe the data and the question (rows, columns, missingness, skew, "
  "correlation density, question kind, tables needed); dynamic ones the trace (tool calls, retries, errors, "
  "LLM calls, tokens, time); graph ones the tool-interaction graph (cycles, transition entropy, maximum "
  "betweenness). Topology is never a feature: it is what the robustness study manipulates. The split groups "
  "by template, so no paraphrase of a training question is tested, and the F2 threshold is chosen on "
  "validation and frozen for test. For fault injection each run gets one fault at a seeded step; runs that "
  "answered before the step are dropped as unrealised, and Delta<sub>T,f</sub> compares against the same "
  "questions' baseline runs, with templates (not runs) resampled for the CI.")

# ------------------------------------------------------------------ 3 results
story.append(PageBreak())
p("3. Results", H1)
p("3.1 How often do agents fail silently?", H2)
p(f"Across the {n_runs} baseline runs {pct(fail_rate)} of answers were wrong: "
  + ", ".join(f"{TOPO[t]} {pct(by_topo[t])}" for t in TOPO) + ". "
  + FINDINGS["failure_rates"])
figure("failure_by_dataset.png", 150, "<b>Figure 1.</b> Baseline failure rate by dataset and topology.")

p("3.2 Can the trace predict failure? (block D)", H2)
rows = [["Model", "F2", "PR-AUC", "Brier", "Precision", "Recall"]]
for k, label in MODEL.items():
    m = test[k]
    rows.append([(f"<b>{label}</b>") if k == "lgbm_weighted" else label, f"{m['f2']:.2f}",
                 f"{m['pr_auc']:.2f}", f"{m['brier']:.3f}", f"{m['precision']:.2f}", f"{m['recall']:.2f}"])
table(rows, [64 * mm, 20 * mm, 22 * mm, 20 * mm, 24 * mm, 22 * mm],
      caption=f"<b>Table 2.</b> Held-out test fold ({clf['split']['sizes']['test']} runs from "
              f"{clf['split']['n_groups']['test']} unseen templates, failure rate "
              f"{pct(clf['class_balance']['test'])}). Threshold chosen on validation.")
p(FINDINGS["classifier"])
figure("feature_importance.png", 115, "<b>Figure 2.</b> What the class-weighted LightGBM relies on "
       "(share of total gain, top 10 features).")

p("3.3 Which topology holds up under faults? (blocks E, F)", H2)
p(f"{len(labf)} fault-injected runs (canonical phrasing x 3 topologies x 3 faults); {pct(realised)} had the "
  "fault actually fire before the agent answered, and only those count.")
rows = [["Topology", "Fault", "Baseline", "Faulted", "Delta", "95% CI", ""]]
for _, r in rob.iterrows():
    rows.append([TOPO[r["topology"]], FAULT[r["fault"]], pct(r["baseline_failure_rate"]),
                 pct(r["faulted_failure_rate"]), signed(r["delta"]),
                 f"[{signed(r['ci_lo'])}, {signed(r['ci_hi'])}]", "*" if r["significant"] else ""])
table(rows, [36 * mm, 32 * mm, 20 * mm, 20 * mm, 20 * mm, 38 * mm, 8 * mm],
      caption="<b>Table 3.</b> Failure-rate increase per topology and fault. * the CI excludes zero.")
figure("robustness_delta.png", 150, "<b>Figure 3.</b> Delta<sub>T,f</sub> with 95% cluster-bootstrap "
       "intervals; right of zero means the fault made the topology fail more.")
p(FINDINGS["robustness"])

p("3.4 Relating the two (block G)", H2)
rows = [["Topology", "Mean Delta", "Failure rate", "PR-AUC (lift)", "Fault failures flagged",
         "Corrupted output flagged", "Main driver"]]
for _, r in summary.iterrows():
    rows.append([TOPO[r["topology"]], signed(r["mean_delta"]), pct(r["failure_rate"]),
                 f"{r['pr_auc']:.2f} ({r['lift']:.1f}x)", pct(r["fault_recall"]),
                 pct(r["corrupted_output_recall"]), r["main_driver"]])
table(rows, [32 * mm, 20 * mm, 20 * mm, 24 * mm, 26 * mm, 28 * mm, 22 * mm],
      caption="<b>Table 4.</b> One row per topology, most robust first. PR-AUC is out-of-fold on the baseline; "
              "lift divides it by the failure rate. Flagged = recall of the baseline-trained predictor on "
              "realised fault runs, at the out-of-fold F2 threshold.")
figure("pr_curves.png", 95, "<b>Figure 4.</b> Out-of-fold precision-recall per topology.")
figure("fault_recall.png", 150, "<b>Figure 5.</b> Share of fault-induced failures the predictor flags.")
p(FINDINGS["relate"])

# ------------------------------------------------------------------ 4-6
p("4. Discussion", H1)
bullets(FINDINGS["discussion"])

p("5. Problems we found in our own measurement", H1)
p("Three issues would each have produced wrong numbers with no error, the failure mode this project "
  "studies, so we record them:")
bullets([
    ("<b>The answer parser split long integers.</b> '32561' parsed as 61 and '2000' as 0, so correct runs were "
    "labelled failures; a first live slice showed 50% failure against a true 10%. Fixed with a regression test."),
    ("<b>The schema preamble leaked answers.</b> It printed row counts and missing rates, the literal answers "
    "to six templates, which then failed at 23% against 47% for the rest. It now names columns and types only."),
    ("<b>Three questions were unanswerable.</b> The 1.5 IQR templates need quartiles and no tool computed one, "
    "so every model failed them on every topology. A quantile operation was added."),
])
p("The corpus in this report was generated after all three fixes.", SM)

p("6. Limitations", H1)
bullets(FINDINGS["limitations"])

p("7. Conclusion", H1)
p(FINDINGS["conclusion"])

p("References", H1)
for ref in [
    "[1] M. Cemri et al., \"Why do multi-agent LLM systems fail?\", NeurIPS Datasets and Benchmarks, 2025.",
    "[2] K. Xu et al., \"LongDS-Bench: On the failure of long-horizon agentic data analysis\", arXiv:2605.30434, 2026.",
    "[3] Y. Shi et al., \"EarlyEval: Cheaper agent evaluation via early outcome prediction\", arXiv:2609.02783, 2026.",
    "[4] Microsoft Research, \"AgentRx: Diagnosing AI agent failures from execution trajectories\", arXiv:2602.02475, 2026.",
    "[5] J. Huang et al., \"On the resilience of LLM-based multi-agent collaboration with faulty agents\", ICML, 2025.",
    "[6] J. Jia et al., \"MAS-FIRE: Fault injection and reliability evaluation for LLM-based multi-agent systems\", arXiv:2602.19843, 2026.",
    "[7] S. Farquhar et al., \"Detecting hallucinations in large language models using semantic entropy\", Nature 630, 2024.",
    "[8] O. Somov and E. Tutubalina, \"Confidence estimation for error detection in text-to-SQL systems\", AAAI, 2025.",
    "[9] S. Hong et al., \"Data Interpreter: An LLM agent for data science\", arXiv:2402.18679, 2024.",
    "[10] M. Sun et al., \"LAMBDA: A large model based data agent\", JASA, 2025.",
    "[11] N. V. Chawla et al., \"SMOTE: Synthetic minority over-sampling technique\", JAIR 16, 2002.",
]:
    p(ref, SM)


def footer(c, d):
    c.setFont("Helvetica", 7.5)
    c.setFillColor(MUTED)
    c.drawRightString(A4[0] - 18 * mm, 10 * mm, f"CSD355 final report · page {d.page}")


OUT.parent.mkdir(parents=True, exist_ok=True)
SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                  bottomMargin=16 * mm, title="Reliability and Failure Topology of LLM Agents",
                  author="Shaurya Mittal Nair, Pragun Kathuria").build(
    story, onFirstPage=footer, onLaterPages=footer)
print(f"wrote {OUT}")
