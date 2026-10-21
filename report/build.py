"""Build the final project report PDF from the results/ tables and figures.

    python report/build.py                      # -> report/final_report.pdf
    python report/build.py out.pdf results_dir  # another corpus

Every number in the results sections is read from results/ (the extended
metrics are computed here from the saved out-of-fold predictions, so nothing is
retrained). The prose that interprets the numbers lives in report/findings.json
and is checked against them by hand.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
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

from agentrel.model.metrics import (
    evaluate_extended,
    expected_calibration_error,
    f2_score,
    grouped_bootstrap_ci,
    pr_auc,
    recall_at_budget,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "report" / "final_report.pdf"
RES = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results"
FIG = RES / "figures"

# DejaVu ships with matplotlib and covers Greek and maths symbols (Delta, epsilon,
# tau, theta, p-hat), which the PDF base fonts cannot draw.
_TTF = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for name, file in (("DV", "DejaVuSans"), ("DV-B", "DejaVuSans-Bold"),
                   ("DV-I", "DejaVuSans-Oblique"), ("DV-BI", "DejaVuSans-BoldOblique"),
                   ("DVM", "DejaVuSansMono")):
    pdfmetrics.registerFont(TTFont(name, str(_TTF / f"{file}.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

INK, ACC, MUTED, RULE, HEAD = (colors.HexColor(c) for c in
                               ("#1f2933", "#2457a6", "#5b6670", "#c8ced6", "#e6edf7"))
SF_RED = "#a8322d"                       # the colour of the term "silent failure"
SF_TINT = colors.HexColor("#fbeeed")
CODE_BG = colors.HexColor("#f4f6f9")


def _style(name, **kw):
    base = {"fontName": "DV", "fontSize": 9.4, "leading": 13.2, "textColor": INK, "spaceAfter": 4}
    base.update(kw)
    return ParagraphStyle(name, **base)


TITLE = _style("t", fontName="DV-B", fontSize=16.5, leading=21, alignment=1, spaceAfter=3)
TAG = _style("tag", fontSize=11, leading=14, alignment=1, textColor=colors.HexColor(SF_RED))
SUB = _style("s", fontSize=8.8, leading=12.5, textColor=MUTED, alignment=1)
H1 = _style("h1", fontName="DV-B", fontSize=12.5, leading=16, textColor=ACC, spaceBefore=10,
            spaceAfter=4)
H2 = _style("h2", fontName="DV-B", fontSize=10.2, leading=13.5, spaceBefore=6, spaceAfter=2)
B = _style("b")
SM = _style("sm", fontSize=8.0, leading=10.6, textColor=MUTED)
CELL = _style("c", fontSize=7.9, leading=10.2, spaceAfter=0)
BUL = _style("bul", leftIndent=12, bulletIndent=2, spaceAfter=2)
ABS = _style("abs", fontSize=9.0, leading=12.8, leftIndent=7 * mm, rightIndent=7 * mm)
EQ = _style("eq", fontSize=9.6, alignment=1, spaceBefore=2, spaceAfter=5)
ALG = _style("alg", fontName="DVM", fontSize=7.7, leading=10.4, spaceAfter=0)
BOX = _style("box", fontSize=9.4, leading=13.2, spaceAfter=0)

SF = f'<font color="{SF_RED}"><b>silent failure</b></font>'
SFS = f'<font color="{SF_RED}"><b>silent failures</b></font>'
SF_CAP = f'<font color="{SF_RED}"><b>Silent failure</b></font>'
SFS_CAP = f'<font color="{SF_RED}"><b>Silent failures</b></font>'

story: list = []


def pct(x):
    return f"{x * 100:.0f}%"


def signed(x):
    return f"{x * 100:+.0f} pp"


def p(text, style=B):
    story.append(Paragraph(text, style))


def bullets(items):
    for t in items:
        story.append(Paragraph(t, BUL, bulletText="•"))


def eq(text, number):
    t = Table([[Paragraph(text, EQ), Paragraph(f"({number})", _style("en", alignment=2))]],
              colWidths=[150 * mm, 24 * mm])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story.append(t)


def table(rows, widths, caption=None, highlight_first=False):
    data = [[Paragraph(str(c), CELL) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1)
    style = [("GRID", (0, 0), (-1, -1), 0.4, RULE), ("BACKGROUND", (0, 0), (-1, 0), HEAD),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if highlight_first:
        style.append(("BACKGROUND", (0, 1), (-1, 1), SF_TINT))
    t.setStyle(TableStyle(style))
    block = [Paragraph(caption, SM), Spacer(1, 2), t] if caption else [t]
    story.append(KeepTogether(block + [Spacer(1, 6)]))


def callout(title, body, tint=SF_TINT, edge=SF_RED):
    t = Table([[Paragraph(f"<b>{title}</b><br/>{body}", BOX)]], colWidths=[174 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), tint),
                           ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(edge)),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story.append(KeepTogether([t, Spacer(1, 7)]))


def algorithm(title, lines):
    body = "<br/>".join(line.replace(" ", "&nbsp;") for line in lines)
    t = Table([[Paragraph(f"<font name='DV-B'>{title}</font>", _style("at", fontSize=8.6))],
               [Paragraph(body, ALG)]], colWidths=[174 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), CODE_BG),
                           ("LINEABOVE", (0, 0), (-1, 0), 1, INK), ("LINEBELOW", (0, 0), (-1, 0), 0.5, INK),
                           ("LINEBELOW", (0, -1), (-1, -1), 1, INK),
                           ("LEFTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 3),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    story.append(KeepTogether([t, Spacer(1, 7)]))


def figure(name, width_mm, caption):
    img = Image(str(FIG / name))
    scale = width_mm * mm / img.drawWidth
    img.drawWidth, img.drawHeight = img.drawWidth * scale, img.drawHeight * scale
    story.append(KeepTogether([img, Spacer(1, 2), Paragraph(caption, SM), Spacer(1, 6)]))


def pipeline_diagram() -> Drawing:
    """Blocks A-G: one corpus feeding the predictor (top) and the robustness study (bottom)."""
    w, h = 174 * mm, 58 * mm
    d = Drawing(w, h)
    box_w, box_h = 25.5 * mm, 12 * mm

    def box(x, y, label, sub, fill="#e6edf7", edge="#2457a6"):
        d.add(Rect(x, y, box_w, box_h, rx=3, ry=3, fillColor=colors.HexColor(fill),
                   strokeColor=colors.HexColor(edge), strokeWidth=0.8))
        d.add(String(x + box_w / 2, y + box_h - 4.6 * mm, label, fontName="DV-B", fontSize=6.9,
                     textAnchor="middle", fillColor=INK))
        d.add(String(x + box_w / 2, y + 2.4 * mm, sub, fontName="DV", fontSize=5.7,
                     textAnchor="middle", fillColor=MUTED))

    def arrow(x1, y1, x2, y2):
        d.add(Line(x1, y1, x2, y2, strokeColor=MUTED, strokeWidth=0.9))
        v = np.array([x2 - x1, y2 - y1], dtype=float)
        v /= np.linalg.norm(v)
        n = np.array([-v[1], v[0]])
        tip = np.array([x2, y2])
        a, b = tip - 2.2 * mm * v + 1.1 * mm * n, tip - 2.2 * mm * v - 1.1 * mm * n
        d.add(Polygon([x2, y2, *a, *b], fillColor=MUTED, strokeColor=MUTED, strokeWidth=0))

    mid = h / 2 - box_h / 2
    top, bot = h - box_h - 3 * mm, 3 * mm
    x0, gap = 0, (w - 6 * box_w) / 5
    box(x0, mid, "Inputs", "5 datasets, 50 questions", "#f4f6f9", "#5b6670")
    xa = x0 + box_w + gap
    box(xa, mid, "A. Run agents", "3 topologies, traced")
    xb = xa + box_w + gap
    box(xb, mid, "B. Label + clean", "y = silent failure?", "#fbeeed", SF_RED)
    xc = xb + box_w + gap
    box(xc, top, "C. Features", "static, dynamic, graph")
    box(xc, bot, "E. Fault injection", "450 faulted runs")
    xd = xc + box_w + gap
    box(xd, top, "D. Gate", "p̂ = P(silent failure)")
    box(xd, bot, "F. Robustness", "Δ(T,f) with 95% CI")
    xg = xd + box_w + gap
    box(xg, mid, "G. Relate D, F", "topology vs signal")
    yc = mid + box_h / 2
    arrow(x0 + box_w, yc, xa, yc)
    arrow(xa + box_w, yc, xb, yc)
    arrow(xb + box_w, yc, xc, top + box_h / 2)
    arrow(xb + box_w, yc, xc, bot + box_h / 2)
    arrow(xc + box_w, top + box_h / 2, xd, top + box_h / 2)
    arrow(xc + box_w, bot + box_h / 2, xd, bot + box_h / 2)
    arrow(xd + box_w, top + box_h / 2, xg, yc)
    arrow(xd + box_w, bot + box_h / 2, xg, yc)
    return d


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
oof = pd.read_csv(RES / "oof_predictions.csv")
FINDINGS = json.loads((ROOT / "report" / "findings.json").read_text())

n_runs, fail_rate = len(lab), lab["label_failure"].mean()
by_topo = lab.groupby("topology")["label_failure"].mean()
realised = labf["fault_realised"].mean()
test = clf["test_metrics"]


def _failure_kind(r) -> str:
    """Silent = a clean, well-formed answer that is wrong; visible = no usable answer."""
    if r["label_failure"] == 0:
        return "correct"
    if pd.isna(r["parsed_answer"]) or bool(r.get("harness_error", 0)):
        return "visible"
    if r["answer_type"] != "numeric":
        return "silent_category"
    try:
        err = abs(float(r["parsed_answer"]) - float(r["truth_value"])) / max(abs(float(r["truth_value"])), 1e-9)
    except (TypeError, ValueError):
        return "silent_value"
    return "silent_near" if err <= 0.05 else "silent_value"


kinds = lab.apply(_failure_kind, axis=1).value_counts()
n_fail = int(lab["label_failure"].sum())
n_silent = int(kinds.filter(like="silent").sum())

# extended out-of-fold metrics with template-clustered CIs (nothing retrained)
y, prob, groups = oof["label_failure"].to_numpy(), oof["p_failure_oof"].to_numpy(), oof["template_id"]
theta = rel["threshold_f2_oof"]
ext = evaluate_extended(y, prob, theta)


def ci(metric):
    lo, hi = grouped_bootstrap_ci(y, prob, groups, metric, n_resamples=1000)
    return f"[{lo:.2f}, {hi:.2f}]"


ext_ci = {
    "pr_auc": ci(pr_auc),
    "roc_auc": ci(lambda a, b: evaluate_extended(a, b, theta)["roc_auc"]),
    "f2": ci(lambda a, b: f2_score(a, (b >= theta).astype(int))),
    "ece": ci(expected_calibration_error),
    "recall_at_20pct": ci(lambda a, b: recall_at_budget(a, b, 0.2)),
}

# ------------------------------------------------------------------ title
p("Catching Silent Failures", TITLE)
p("Reliability and Failure Topology of Autonomous LLM Agents in Data Analysis Pipelines", TAG)
story.append(Spacer(1, 3))
p("Shaurya Mittal Nair (2410110316) · Pragun Kathuria (2410110237)<br/>"
  "Department of Computer Science and Engineering, Shiv Nadar Institution of Eminence<br/>"
  "Final Project Report, Foundations of Data Science (CSD355) · October 2026 · "
  "github.com/Uncharted1804/agent-reliability", SUB)
story.append(Spacer(1, 8))
p("<b>Abstract.</b> " + FINDINGS["abstract"], ABS)
story.append(Spacer(1, 4))
callout("Silent failure: the object of this study",
        f"A {SF} is an agent run that ends normally and returns a clean, well-formed, confident answer that is "
        "<b>wrong</b>. Nothing raises an exception, nothing is logged as an error, no retry fires and no warning "
        "is shown, so every monitor that watches for crashes reports success. It is the opposite of a "
        "<i>visible failure</i> (a crash, an error, an empty or unparseable answer), which existing agents already "
        "catch and retry. This project measures how often silent failures happen, predicts them from signals "
        "available before the answer is shown, and asks which agent architectures let them spread.")

# ------------------------------------------------------------------ 1
p("1. Introduction", H1)
p(f"LLM agents now take a table and a question and run the whole analysis themselves: ChatGPT's data analysis "
  f"[27], Claude's code execution [28], Google's Colab Data Science Agent [29], Data Interpreter [9] and "
  f"LAMBDA [10]. Their error handling reacts to code that crashes. A wrong filter, a misread column or a "
  f"skipped step raises nothing, so the retry loop never fires and the user receives a {SF}: a confident "
  f"number that is wrong and looks exactly like a right one. A crash gets noticed and retried; a "
  f"silent failure gets pasted into a report.")
p("We ask two questions. <b>(Q1)</b> Can features of the input data and of the agent's own execution trace predict "
  f"a {SF} before the answer is used? <b>(Q2)</b> Does the structure of an agent's tool-interaction graph tell "
  "us which architectures hold up when a component breaks, and do silent failures spread or stay contained? "
  "One labelled corpus answers both, so the answers can be related directly (block G). Prior work describes "
  "failures after the fact [1, 2, 4], predicts outcomes on coding and web tasks [3], measures robustness only "
  "as aggregate accuracy [5, 6], or scores single answers rather than multi-step traces [7, 8]; none predicts "
  "silent failure per run for tabular-analysis agents and links it to topology robustness.")
p("<b>Contributions.</b>", B)
bullets([
    ("<b>AgentRel</b>, a measurement framework that wraps any tool-using analysis agent, stated as four algorithms "
    f"(section 3) and implemented end to end with {FINDINGS['n_tests']} unit tests."),
    (f"A labelled corpus of {n_runs} traced runs and {len(labf)} fault-injected runs over five datasets and three "
    "topologies, with every label computed by deterministic code."),
    (f"A pre-answer <b>{SF.replace('silent failure', 'silent-failure')} gate</b>: a class-weighted gradient-boosted "
    "classifier whose score ranks runs for review."),
    ("A fault-injection study with template-clustered confidence intervals, and the first analysis relating which "
    "signals predict silent failure in each topology to how fragile that topology is."),
])

# ------------------------------------------------------------------ 2
p("2. Background and related work", H1)
table([
    ["Work", "What it does", "Gap this project fills"],
    ["MAST [1]", "Expert-annotated taxonomy of 14 multi-agent failure modes", "Descriptive only; no prediction"],
    ["LongDS-Bench [2]", "68 Kaggle-based multi-turn analysis tasks; accuracy falls ~47 pp late in a session",
     "Shows that analysis agents lose state; does not flag which run failed"],
    ["EarlyEval [3]", "LightGBM on partial trajectories predicts the final outcome (coding, terminal tasks)",
     "Not tabular analysis; does not separate silent from visible failure"],
    ["AgentRx [4]", "Localises the first unrecoverable step of a failed run", "Works only after a failure is known"],
    ["Huang et al. [5]; MAS-FIRE [6]", "Inject faulty agents or 15 fault types; rank topologies by resilience",
     "Aggregate accuracy, no per-run risk score"],
    ["Semantic entropy [7]; Somov and Tutubalina [8]", "Confidence for a single answer or SQL query",
     "Single output, not a multi-step tool trace"],
    ["Data Interpreter [9]; LAMBDA [10]", "Analysis agents with per-node verification or an inspector",
     "Catch code that crashes; a silent failure passes their checks"],
], [44 * mm, 72 * mm, 58 * mm], caption="<b>Table 1.</b> Closest prior work.")

# ------------------------------------------------------------------ 3 framework
story.append(PageBreak())
p("3. The AgentRel framework", H1)
p("AgentRel sits <i>outside</i> the agent. It never changes how the agent reasons; it runs controlled tasks, "
  f"labels every run against computed ground truth, learns which signals precede a {SF}, and stress-tests the "
  "agent's wiring. Figure 1 shows the seven blocks; one corpus feeds both branches.")
story.append(pipeline_diagram())
p("<b>Figure 1.</b> The AgentRel pipeline. Blocks A and B build one labelled corpus; the top branch learns to "
  "predict silent failure, the bottom branch measures robustness under faults, and block G relates them.", SM)
story.append(Spacer(1, 6))

p("3.1 Definitions", H2)
p("A <b>run</b> is one agent topology T answering one phrasing π of a question q on dataset d with base model "
  "M. Its <b>trace</b> τ is the ordered list of tool calls (tool, arguments, output, success, retry flag, "
  "latency) plus token counts and the final free-text answer. Let a be the answer parsed from that text and "
  "a* the value computed by a deterministic pandas script for q. The run's label is")
eq("y(τ) = 1[ a = ∅  ∨  |a − a*| > ε·|a*| ],   ε = 0.01", 1)
p("For a categorical answer the second condition is a ≠ a* after normalising case and punctuation.", SM)
p(f"and a failing run is a {SF} when a ≠ ∅ (the agent "
  "returned a clean answer) and a <i>visible failure</i> otherwise. The gate estimates")
eq("p̂(τ) = f( φ<sub>static</sub>(d, q),  φ<sub>dynamic</sub>(τ),  "
   "φ<sub>graph</sub>(G(τ)) )  ≈  P( y = 1 | τ )", 2)
p("where G(τ) = (V, E) is the tool-interaction graph: nodes are tools, edge weights count transitions between "
  "consecutive calls. Topology T is never an input to f, because it is the variable the robustness study "
  "manipulates. f minimises the class-weighted log loss")
eq("L = −(1/N) Σ<sub>i</sub> [ w<sub>1</sub> y<sub>i</sub> log p̂<sub>i</sub> + "
   "w<sub>0</sub> (1 − y<sub>i</sub>) log(1 − p̂<sub>i</sub>) ],   "
   "w<sub>c</sub> = N / (2 N<sub>c</sub>)", 3)
p("and robustness of topology T to fault f is the increase in failure rate on the same questions")
eq("Δ<sub>T,f</sub> = P( y = 1 | T, f ) − P( y = 1 | T, no fault )", 4)
p("(the proposal defined a relative accuracy drop; the absolute difference is used because it stays defined when "
  "baseline accuracy is low and its bootstrap is better behaved).", SM)

p("3.2 Algorithms", H2)
algorithm("Algorithm 1  Build the labelled corpus (blocks A, B)", [
    "Input:  datasets D, templates Q_d with truth functions, phrasings Π(q), topologies Topos, model M, ε",
    "Output: corpus C = {(τ, y)}",
    "1  for d ∈ D, q ∈ Q_d, π ∈ Π(q), T ∈ Topos do",
    "2      τ ← RUN(T, M, d, π)            ▷ fresh session; log every call, retry, error, token",
    "3      a ← PARSE(τ.answer)               ▷ number or category out of free text",
    "4      a* ← TRUTH_q(d)                    ▷ deterministic pandas, never the LLM",
    "5      y ← Eq.(1);  kind ← silent if a ≠ ∅ else visible",
    "6      C ← C ∪ {(τ, y, kind)}",
    "7  C ← CLEAN(C)    ▷ keep and flag timeouts; cap heavy tails at Q3 + 1.5·IQR, with audit flags",
])
algorithm("Algorithm 2  Train the silent-failure gate (blocks C, D)", [
    "Input:  corpus C, feature maps φ_static, φ_dynamic, φ_graph",
    "Output: gate (f, θ)",
    "1  x_i ← [φ_static(d_i, q_i), φ_dynamic(τ_i), φ_graph(G(τ_i))]   ▷ topology excluded",
    "2  split C 70/15/15 by template              ▷ no paraphrase of a training question is tested",
    "3  f ← LightGBM minimising Eq.(3) on train   ▷ log1p, one-hot, scaling fitted on train only",
    "4  θ ← argmax_t F2(y_val, 1[f(x_val) ≥ t])    ▷ chosen on validation, then frozen",
    "5  report F2, PR-AUC, Brier, ROC-AUC, MCC, ECE, recall@budget on test at θ",
])
algorithm("Algorithm 3  Gate an answer before it is shown (deployment)", [
    "Input:  a finished run τ, gate (f, θ), optional review budget k",
    "1  p̂ ← f(φ(τ))",
    "2  if p̂ ≥ θ then hold the answer and send the run for review   ▷ likely silent failure",
    "3  else release the answer with p̂ attached as its risk score",
    "4  batch mode: review the ⌈k·N⌉ runs with the highest p̂          ▷ Eq.(6)",
])
algorithm("Algorithm 4  Topology robustness with a cluster bootstrap (blocks E, F)", [
    "Input:  questions Q (canonical phrasing), topologies Topos, faults F, baseline corpus C, B = 1000",
    "1  for T ∈ Topos, f ∈ F, q ∈ Q do",
    "2      s ← hash(run_id, f) mod 3                 ▷ seeded step: same fault on every rerun",
    "3      τ ← RUN(T, M, q) with fault f injected once at step s",
    "4      keep τ only if the fault fired before the agent answered      ▷ realised faults",
    "5  Δ_T,f ← Eq.(4) on realised runs vs baseline runs of the same questions",
    "6  for b = 1..B: resample templates with replacement; recompute Δ_T,f^(b)",
    "7  CI ← [q_2.5, q_97.5] of {Δ_T,f^(b)};  significant if 0 ∉ CI",
])

# ------------------------------------------------------------------ 4 stages
p("4. The pipeline, stage by stage", H1)
p("Each stage writes an artifact the next one reads, so every stage can be re-run and inspected on its own. "
  f"Because a {SF} in our own measurement would corrupt every downstream number, each stage carries a guard "
  "that turns a would-be silent error into a loud one.")
table([
    ["Stage", "Input", "What it does", "Output", "Guard against a silent error"],
    ["Data", "public CSVs or seeded generators", ("materialise 5 datasets; build 50 templates x 5 phrasings "
     "with pandas ground truth"), "datasets/raw, question banks", ("truth errors counted (0); every phrasing "
     "of a template shares one truth value")],
    ["A. Generate", "question banks, config", ("run every phrasing through 3 topologies on one model at "
     "temperature 0; fresh session per run"), "runs.jsonl (one trace per run)", ("--no-mock fails loudly if the "
     "model is unreachable instead of silently using a mock")],
    ["B. Label", "runs.jsonl", "parse the answer, score Eq. (1), paraphrase consistency, flatten and IQR-cap",
     "labelled.parquet", ("regression tests on the parser; a corpus other than the baseline never "
     "overwrites its labels")],
    ["C. Features", "runs + labels", "9 static, 7 dynamic, 3 graph features per run", "feature frame",
     "each block asserts it produced exactly the schema's keys; one-to-one label join"],
    ["D. Classifier", "feature frame", ("class-weighted LightGBM + 4 baselines; grouped split; F2 threshold "
     "on validation"), "classifier_results.json", ("raises if a training fold is single-class rather than "
     "printing an empty table")],
    ["E. Faults", "question banks", "one seeded fault per run at step 0-2", "runs_faults.jsonl",
     "unrealised faults are tracked and excluded, not counted as survived"],
    ["F. Robustness", "both labelled corpora", "Δ per topology x fault; 1,000 template resamples",
     "robustness.csv", "baseline restricted to the faulted questions"],
    ["G. Relate", "both corpora, robustness", ("out-of-fold p̂, per-topology PR-AUC, drivers, gate recall "
     "on fault runs"), "relate_*.csv, oof_predictions.csv", ("folds grouped by template; topology only selects "
     "rows")],
    ["Report", "results/", "figures and this PDF; extended metrics from saved predictions", "final_report.pdf",
     "every number read from results, none typed by hand"],
], [21 * mm, 24 * mm, 49 * mm, 29 * mm, 51 * mm], caption="<b>Table 2.</b> Stages, artifacts and guards.")

p("4.1 Corpus and agents", H2)
p("Five datasets of rising difficulty: Titanic [30] (887 rows) and UCI Adult Census [25] (32,561 rows), downloaded; "
  "and three seeded synthetic stand-ins with the traits the proposal called for where no stable download "
  "exists: an NYC-taxi-style table (8,000 rows, outliers and mixed types), a heavy-missingness table (3,000 rows) "
  "and an Olist-style schema of three tables that needs joins. Ten templates per dataset cover aggregation, "
  "filtering, group comparison, trend and multi-step questions; each is asked canonically and in four "
  f"paraphrases, giving 5 x 10 x 5 x 3 = <b>{n_runs} runs</b>.")
p("The three topologies share one model, one tool registry and one prompt preamble, so only the wiring differs: "
  "<b>single ReAct</b> [13] (one reason-act loop that retries once on a loud error), <b>orchestrator-worker</b> "
  "(planner, worker and an inspector that can demand a redo, after LAMBDA [10]) and <b>peer tool-graph</b> "
  "(loader, transformer and reducer peers, closest to Data Interpreter [9]). Agents act through eight "
  "sandboxed tools and see a preamble that names columns and types but no quantity, so every number in an "
  "answer must come from a logged tool call.")

p("4.2 Base model", H2)
p(FINDINGS["model_choice"])
table([["Model", "Failure rate", "Hit the 12-step cap", "Used no tool", "Seconds / run"]] + [
    [m["label"], pct(m["fail"]), pct(m["timeout"]), pct(m["zero_tool"]), f"{m['sec']:.0f}"]
    for m in FINDINGS["benchmark"]],
    [52 * mm, 28 * mm, 34 * mm, 28 * mm, 30 * mm],
    caption="<b>Table 3.</b> Model benchmark: canonical phrasing of every template x 3 topologies, 150 runs a "
            "model, on the 41 templates unaffected by two harness issues found during the study (section 7).")

p("4.3 Evaluation metrics", H2)
p("Accuracy is never reported: with a large share of correct runs a model that always says 'fine' scores well "
  "and catches nothing. Each metric answers one question a user of the gate would ask.")
table([
    ["Metric", "Definition", "Question it answers"],
    ["F2 [23]", "(1 + 2²)·P·R / (2²·P + R)", ("Recall weighted 4x precision: a missed silent "
     "failure costs more than a needless review")],
    ["PR-AUC [16]", "area under precision-recall (average precision)", ("Ranking quality on the failure class, "
     "not flattered by many easy negatives")],
    ["ROC-AUC", "P( p̂ of a failure > p̂ of a success )", ("Ranking quality, for comparison with "
     "other work")],
    ["MCC [19]", "(TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))", ("One number using all "
     "four confusion cells")],
    ["Balanced accuracy", "(recall on failures + recall on successes) / 2", "Is the gate fair to both classes?"],
    ["Brier [17]", "(1/N) Σ (p̂<sub>i</sub> − y<sub>i</sub>)²", ("Overall quality of p̂ as "
     "a probability")],
    ["ECE [18]", ("Σ<sub>b</sub> (n<sub>b</sub>/N) |ȳ<sub>b</sub> − mean p̂<sub>b</sub>| over 10 "
     "bins  (5)"), "Does p̂ = 0.3 mean a 30% chance?"],
    ["Recall@k", "share of failures among the ⌈kN⌉ highest-p̂ runs  (6)", ("With time to re-check only "
     "k% of runs, how many silent failures are caught?")],
    ["Δ<sub>T,f</sub> CI [21, 22]", "percentile interval over 1,000 template resamples", ("Is a topology's "
     "degradation distinguishable from noise?")],
], [27 * mm, 68 * mm, 79 * mm], caption="<b>Table 4.</b> Metrics. Every interval in this report resamples whole "
                                         "templates, since the runs of one template are near duplicates [20, 22].")

# ------------------------------------------------------------------ 5 results
story.append(PageBreak())
p("5. Results", H1)
p(f"5.1 How common are {SFS}?", H2)
p(f"Of the {n_runs} baseline runs, {n_fail} ({pct(fail_rate)}) failed: "
  + ", ".join(f"{TOPO[t]} {pct(by_topo[t])}" for t in TOPO) + ". "
  f"<b>{n_silent} of the {n_fail} failures ({pct(n_silent / n_fail)}) were {SFS}</b>: the agent returned a clean "
  "answer and nothing in the run signalled a problem. " + FINDINGS["failure_rates"])
table([
    ["Outcome", "Runs", "Share of runs", "What the user sees"],
    [f"{SF_CAP}: wrong value (> 5% off)", int(kinds.get("silent_value", 0)),
     pct(kinds.get("silent_value", 0) / n_runs), "a confident, plausible, wrong number"],
    [f"{SF_CAP}: near miss (1-5% off)", int(kinds.get("silent_near", 0)),
     pct(kinds.get("silent_near", 0) / n_runs), "a number off by rounding or a small slip"],
    [f"{SF_CAP}: wrong category", int(kinds.get("silent_category", 0)),
     pct(kinds.get("silent_category", 0) / n_runs), "a confident wrong label"],
    ["Visible failure", int(kinds.get("visible", 0)), pct(kinds.get("visible", 0) / n_runs),
     "no usable answer (empty, unparseable, harness error)"],
    ["Correct", int(kinds.get("correct", 0)), pct(kinds.get("correct", 0) / n_runs), "the right answer"],
], [62 * mm, 18 * mm, 24 * mm, 70 * mm], caption="<b>Table 5.</b> What a failure looks like.")
figure("failure_by_dataset.png", 150, "<b>Figure 2.</b> Baseline failure rate by dataset and topology.")

p(f"5.2 Can the trace predict a {SF}? (Q1)", H2)
rows = [["Model", "F2", "PR-AUC", "Brier", "Precision", "Recall"]]
for k, label in MODEL.items():
    m = test[k]
    rows.append([f"<b>{label}</b>" if k == "lgbm_weighted" else label, f"{m['f2']:.2f}",
                 f"{m['pr_auc']:.2f}", f"{m['brier']:.3f}", f"{m['precision']:.2f}", f"{m['recall']:.2f}"])
table(rows, [64 * mm, 20 * mm, 22 * mm, 20 * mm, 24 * mm, 24 * mm], highlight_first=True,
      caption=f"<b>Table 6.</b> Held-out test fold ({clf['split']['sizes']['test']} runs from "
              f"{clf['split']['n_groups']['test']} unseen templates, failure rate "
              f"{pct(clf['class_balance']['test'])}). Threshold chosen on validation.")
p(FINDINGS["classifier"])
table([
    ["Metric", "Value", "95% CI (templates resampled)"],
    ["PR-AUC", f"{ext['pr_auc']:.2f}", ext_ci["pr_auc"]],
    ["ROC-AUC", f"{ext['roc_auc']:.2f}", ext_ci["roc_auc"]],
    [f"F2 at θ = {theta:.2f}", f"{ext['f2']:.2f}", ext_ci["f2"]],
    ["MCC", f"{ext['mcc']:.2f}", ""],
    ["Balanced accuracy", f"{ext['balanced_accuracy']:.2f}", ""],
    ["Brier", f"{ext['brier']:.3f}", ""],
    ["ECE (10 bins)", f"{ext['ece']:.3f}", ext_ci["ece"]],
    ["Recall@10% / 20% / 30% reviewed", (f"{pct(ext['recall_at_10pct'])} / {pct(ext['recall_at_20pct'])} / "
                                        f"{pct(ext['recall_at_30pct'])}"), "20%: " + ext_ci["recall_at_20pct"]],
], [70 * mm, 46 * mm, 58 * mm],
    caption=f"<b>Table 7.</b> Extended evaluation over all {len(oof)} runs, out-of-fold (grouped 5-fold CV by "
            f"template, failure rate {pct(y.mean())}): every run is scored by a model that never saw its template.")
figure("review_budget.png", 105, "<b>Figure 3.</b> The gate as a review queue: share of silent failures caught "
       "when only the highest-scored runs are re-checked.")
figure("calibration.png", 90, "<b>Figure 4.</b> Calibration of p̂ (out-of-fold).")
figure("feature_importance.png", 115, "<b>Figure 5.</b> What the gate relies on (share of LightGBM gain).")
p(FINDINGS["metrics"])

p("5.3 Which topology holds up when a component breaks? (Q2)", H2)
p(f"{len(labf)} fault-injected runs (canonical phrasing x 3 topologies x 3 faults); {pct(realised)} had the fault "
  "fire before the agent answered, and only those count. <i>Corrupted output</i> is the fault that manufactures "
  f"{SFS}: the tool succeeds, the value is quietly wrong, and nothing looks amiss.")
rows = [["Topology", "Fault", "Baseline", "Faulted", "Δ", "95% CI", ""]]
for _, r in rob.iterrows():
    rows.append([TOPO[r["topology"]], FAULT[r["fault"]], pct(r["baseline_failure_rate"]),
                 pct(r["faulted_failure_rate"]), signed(r["delta"]),
                 f"[{signed(r['ci_lo'])}, {signed(r['ci_hi'])}]", "*" if r["significant"] else ""])
table(rows, [36 * mm, 32 * mm, 20 * mm, 20 * mm, 20 * mm, 38 * mm, 8 * mm],
      caption="<b>Table 8.</b> Failure-rate increase per topology and fault, Eq. (4). * the CI excludes zero.")
figure("robustness_delta.png", 150, "<b>Figure 6.</b> Δ<sub>T,f</sub> with 95% cluster-bootstrap intervals; "
       "right of zero means the fault made the topology fail more.")
p(FINDINGS["robustness"])

p("5.4 Relating the two (block G)", H2)
rows = [["Topology", "Mean Δ", "Failure rate", "PR-AUC (lift)", "Fault failures flagged",
         "Corrupted output flagged", "Main driver"]]
for _, r in summary.iterrows():
    rows.append([TOPO[r["topology"]], signed(r["mean_delta"]), pct(r["failure_rate"]),
                 f"{r['pr_auc']:.2f} ({r['lift']:.1f}x)", pct(r["fault_recall"]),
                 pct(r["corrupted_output_recall"]), r["main_driver"]])
table(rows, [32 * mm, 18 * mm, 20 * mm, 24 * mm, 28 * mm, 30 * mm, 22 * mm],
      caption="<b>Table 9.</b> One row per topology, most robust first. PR-AUC is out-of-fold; lift divides it by the "
              "failure rate. Flagged = recall of the baseline-trained gate on realised fault runs.")
figure("pr_curves.png", 95, "<b>Figure 7.</b> Out-of-fold precision-recall per topology.")
figure("fault_recall.png", 150, "<b>Figure 8.</b> Share of fault-induced failures the gate flags.")
p(FINDINGS["relate"])

# ------------------------------------------------------------------ 6-8
p("6. Discussion", H1)
bullets(FINDINGS["discussion"])

p(f"7. {SFS_CAP} in our own measurement", H1)
p("Three defects would each have produced wrong numbers with no error, which is to say they were silent failures "
  "of the measurement layer itself. We found them by checking results against expectations, and record them "
  "because they are the project's subject in miniature:")
bullets([
    ("<b>The answer parser split long integers.</b> '32561' parsed as 61 and '2000' as 0, so correct runs were "
    "labelled failures: a first live slice showed 50% failure against a true 10%. Fixed, with a regression test."),
    ("<b>The schema preamble leaked answers.</b> It printed row counts and missing rates, the literal answers to "
    "six templates, which then failed at 23% against 47% for the rest. It now names columns and types only."),
    ("<b>Three questions were unanswerable.</b> The 1.5·IQR templates need quartiles and no tool computed one, "
    "so every model failed them on every topology. A quantile operation was added."),
])
p("The corpus in this report was generated after all three fixes.", SM)

p("8. Limitations", H1)
bullets(FINDINGS["limitations"])

p("9. Conclusion", H1)
p(FINDINGS["conclusion"])

# ------------------------------------------------------------------ references
p("References", H1)
REFS = [
    ("M. Cemri, M. Z. Pan, S. Yang, et al. Why do multi-agent LLM systems fail? In <i>Advances in Neural "
    "Information Processing Systems (NeurIPS), Datasets and Benchmarks Track</i>, 2025."),
    ("K. Xu, X. Lu, S. Qiao, Z. Ding, H. Xu, L. Liang, N. Zhang. LongDS-Bench: On the failure of long-horizon "
    "agentic data analysis. Preprint, arXiv:2605.30434, 2026."),
    ("Y. Shi, Z. Sun, J. Dong, C. Wan, D. Lo, X. Gu. EarlyEval: Cheaper agent evaluation via early outcome "
    "prediction. Preprint, arXiv:2609.02783, 2026."),
    ("Microsoft Research. AgentRx: Diagnosing AI agent failures from execution trajectories. In <i>Proceedings of "
    "the 43rd International Conference on Machine Learning (ICML)</i>, 2026."),
    ("J.-T. Huang, J. Zhou, T. Jin, X. Zhou, Z. Chen, W. Wang, Y. Yuan, M. R. Lyu, M. Sap. On the resilience of "
    "LLM-based multi-agent collaboration with faulty agents. In <i>Proceedings of the 42nd International "
    "Conference on Machine Learning (ICML)</i>, PMLR 267:26202-26226, 2025."),
    ("J. Jia, Z. Deng, Z. Chen, Y. Wang, Z. Zheng. MAS-FIRE: Fault injection and reliability evaluation for "
    "LLM-based multi-agent systems. Preprint, arXiv:2602.19843, 2026."),
    ("S. Farquhar, J. Kossen, L. Kuhn, Y. Gal. Detecting hallucinations in large language models using semantic "
    "entropy. <i>Nature</i> 630(8017):625-630, 2024. doi:10.1038/s41586-024-07421-0."),
    ("O. Somov, E. Tutubalina. Confidence estimation for error detection in text-to-SQL systems. In "
    "<i>Proceedings of the AAAI Conference on Artificial Intelligence</i> 39(23):25137-25145, 2025. "
    "doi:10.1609/aaai.v39i23.34699."),
    ("S. Hong, Y. Lin, B. Liu, et al. Data Interpreter: An LLM agent for data science. In <i>Findings of the "
    "Association for Computational Linguistics: ACL 2025</i>, pp. 19796-19821, 2025. "
    "doi:10.18653/v1/2025.findings-acl.1016."),
    ("M. Sun, R. Han, B. Jiang, H. Qi, D. Sun, Y. Yuan, J. Huang. LAMBDA: A large model based data agent. "
    "<i>Journal of the American Statistical Association</i>, 2025. doi:10.1080/01621459.2025.2510000."),
    ("X. Hu, Z. Zhao, S. Wei, et al. InfiAgent-DABench: Evaluating agents on data analysis tasks. In "
    "<i>Proceedings of the 41st International Conference on Machine Learning (ICML)</i>, PMLR 235:19544-19572, 2024."),
    ("A. Egg, M. Iglesias Goyanes, F. Kingma, et al. DABstep: Data agent benchmark for multi-step reasoning. "
    "Preprint, arXiv:2506.23719, 2025."),
    ("S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, Y. Cao. ReAct: Synergizing reasoning and acting "
    "in language models. In <i>International Conference on Learning Representations (ICLR)</i>, 2023."),
    ("G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, T.-Y. Liu. LightGBM: A highly efficient gradient "
    "boosting decision tree. In <i>Advances in Neural Information Processing Systems 30 (NeurIPS)</i>, "
    "pp. 3146-3154, 2017."),
    ("N. V. Chawla, K. W. Bowyer, L. O. Hall, W. P. Kegelmeyer. SMOTE: Synthetic minority over-sampling technique. "
    "<i>Journal of Artificial Intelligence Research</i> 16:321-357, 2002."),
    ("T. Saito, M. Rehmsmeier. The precision-recall plot is more informative than the ROC plot when evaluating "
    "binary classifiers on imbalanced datasets. <i>PLoS ONE</i> 10(3):e0118432, 2015."),
    ("G. W. Brier. Verification of forecasts expressed in terms of probability. <i>Monthly Weather Review</i> "
    "78(1):1-3, 1950."),
    ("C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger. On calibration of modern neural networks. In <i>Proceedings "
    "of the 34th International Conference on Machine Learning (ICML)</i>, PMLR 70:1321-1330, 2017."),
    ("D. Chicco, G. Jurman. The advantages of the Matthews correlation coefficient (MCC) over F1 score and "
    "accuracy in binary classification evaluation. <i>BMC Genomics</i> 21:6, 2020."),
    ("S. Kaufman, S. Rosset, C. Perlich, O. Stitelman. Leakage in data mining: Formulation, detection, and "
    "avoidance. <i>ACM Transactions on Knowledge Discovery from Data</i> 6(4):15, 2012."),
    "B. Efron, R. J. Tibshirani. <i>An Introduction to the Bootstrap</i>. Chapman &amp; Hall/CRC, 1993.",
    ("A. C. Cameron, J. B. Gelbach, D. L. Miller. Bootstrap-based improvements for inference with clustered "
    "errors. <i>The Review of Economics and Statistics</i> 90(3):414-427, 2008."),
    "C. J. van Rijsbergen. <i>Information Retrieval</i>, 2nd ed. Butterworths, 1979.",
    "Qwen Team. Qwen3 technical report. Preprint, arXiv:2505.09388, 2025.",
    "B. Becker, R. Kohavi. Adult [dataset]. UCI Machine Learning Repository, 1996. doi:10.24432/C5XW20.",
    ("F. Pedregosa, G. Varoquaux, A. Gramfort, et al. Scikit-learn: Machine learning in Python. <i>Journal of "
    "Machine Learning Research</i> 12:2825-2830, 2011."),
    ("OpenAI. Data analysis with ChatGPT. OpenAI Help Center, help.openai.com/en/articles/8437071 "
    "(accessed October 2026)."),
    ("Anthropic. Create and edit files with Claude. Claude Help Center, "
    "support.claude.com/en/articles/12111783 (accessed October 2026)."),
    "Google. Data Science Agent in Colab. Google Developers Blog, March 2025.",
    "Kaggle. Titanic: Machine learning from disaster [dataset]. kaggle.com/competitions/titanic.",
]
REF = _style("ref", fontSize=7.9, leading=10.4, leftIndent=18, firstLineIndent=-18, spaceAfter=2.5)
for i, ref in enumerate(REFS, 1):
    p(f"[{i}]&nbsp;&nbsp;{ref}", REF)


def footer(c, d):
    c.setFont("DV", 7.3)
    c.setFillColor(MUTED)
    c.drawString(18 * mm, 10 * mm, "Catching Silent Failures · CSD355 final report")
    c.drawRightString(A4[0] - 18 * mm, 10 * mm, f"page {d.page}")


OUT.parent.mkdir(parents=True, exist_ok=True)
SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                  bottomMargin=16 * mm, title="Catching Silent Failures",
                  author="Shaurya Mittal Nair, Pragun Kathuria").build(
    story, onFirstPage=footer, onLaterPages=footer)
print(f"wrote {OUT}")
