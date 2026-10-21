# Reliability and Failure Topology of Autonomous LLM Agents in Data Analysis Pipelines

Measurement layer around LLM data-analysis agents. Two linked analyses are fed by **one labelled corpus** of agent runs:

1. **Silent-failure predictor** — a class-weighted LightGBM classifier that flags wrong-but-confident answers *before* they are used, from features of the input data and the agent's execution trace.
2. **Topology robustness** — a fault-injection study measuring how much three agent topologies degrade when a component breaks.

> FDS (CSD355) Midsem Project — Shaurya Mittal Nair (2410110316), Pragun Kathuria (2410110237). Shiv Nadar Institution of Eminence.

---

## What a "silent failure" is

A data-analysis agent loads a table, decides which tool calls to make (filter, group, aggregate, compute a statistic, plot) and returns an answer. When it is wrong it usually does **not** crash — it returns a clean, confident, incorrect answer. Ordinary monitoring (exceptions, error logs) never fires. This project builds the measurement layer that catches exactly those runs.

## Pipeline (matches the report)

```
Public datasets ─┐
Question templates ├─► (A) Agent-run generator ─► (B) Labelling + ─► (C) Feature ─► (D) Failure
+ paraphrases    │        logs full trace          log cleaning       engineering     classifier  p̂
3 agent topologies┘                │ (shared corpus, 1 row/run)                     │
                                    └──────────────► (E) Fault injection ─► (F) Robustness ΔT,f
                                                                          (G) Relate predictors to robustness
```

## Corpus size

| | count |
|---|---|
| datasets | 5 |
| question templates / dataset | 10 |
| phrasings / template | 5 (canonical + 4 paraphrases) |
| topologies | 3 |
| **base runs** | **5 × 10 × 5 × 3 = 750** |
| fault-injected runs (full design) | 750 × 3 = 2,250 |
| fault-injected runs (default scale) | 450 — canonical phrasing only, see below |

## Repository layout

```
agent-reliability/
├── src/agentrel/
│   ├── data/            # dataset acquisition + deterministic ground-truth
│   ├── agents/          # tool registry, base LLM client (local/Ollama)
│   ├── topology/        # single ReAct, orchestrator-worker, peer tool-graph
│   ├── harness/         # run generator, trace logging schema
│   ├── labelling/       # answer parsing, Eq.1 labelling, log cleaning
│   ├── features/        # static / dynamic / graph feature engineering
│   ├── model/           # split, pipelines, metrics, classifier training
│   ├── faults/          # fault injection + robustness ΔT,f
│   ├── relate.py        # block G: predictor vs robustness
│   └── utils/           # config, io, logging helpers
├── datasets/            # downloaded raw data (gitignored) + question banks
├── configs/             # experiment config (topologies, faults, splits)
├── scripts/             # entry points: download, generate, label, train, faults, relate, figures
├── report/              # build.py + findings.json -> ../Shaurya_Pragun_FDS_Project_Report.pdf
├── tests/               # unit tests for every block
└── results/             # corpora, labels, tables, figures (gitignored)
```

## Status — complete

All seven blocks of the report's workflow are implemented and have been run end to end; results are below. Built on top of Milestone 1:

* **feature engineering** (`features/`) — static (data + question), dynamic (trace), and graph features from the tool-interaction graph G=(V,E). The names in `features/schema.py` are the contract; each block asserts it produced exactly those keys.
* **silent-failure classifier** (`model/`) — class-weighted LightGBM with logistic-regression, SMOTE+LR, SMOTE+LightGBM and prior-dummy baselines, on the grouped-by-template 70/15/15 split. Reports F2 / PR-AUC / Brier, with the F2 threshold chosen on validation and frozen for test.
* **fault injection + robustness** (`faults/`) — `tool_failure`, `corrupted_output`, `dropped_context`, and Δ_T,f with cluster-bootstrap CIs (1000 resamples, resampled over templates).

* **relating the two** (`relate.py`, block G) — out-of-fold failure probabilities from grouped 5-fold CV, PR-AUC per topology, per-topology feature drivers, and the baseline-trained predictor scored on the fault-injected runs.
* **the silent-failure gate** (`gate.py`, `scripts/gate.py`) — the classifier as a deployable step: fit once on the labelled corpus, then score new runs from their traces alone and hold the likely silent failures for review, by threshold or by a fixed review budget.
* **a quantile operation** for the tools, without which the three 1.5×IQR templates could not be answered by any agent.
* **report pipeline** — `scripts/figures.py` and `report/build.py` turn `results/` into `Shaurya_Pragun_FDS_Project_Report.pdf`.

### Methodology notes

Three choices are worth knowing before reading any number this produces:

* **Topology is not a feature.** It is the variable the robustness study manipulates, so letting the classifier read it would confound "this wiring fails more" with "this task is hard".
* **The split groups on `template_id`.** A template appears as 5 phrasings × 3 topologies, i.e. up to 15 near-duplicate rows. A row-wise split would put paraphrases of a training question in test and the reported F2 would mostly measure memorisation.
* **The bootstrap resamples templates, not runs.** Paraphrases of one template are near duplicates; resampling runs treats them as independent observations and gives CIs that are too narrow.

### Corpus actually generated

`expand_questions` emits the canonical phrasing *plus* 4 paraphrases, which is why the table above counts 5 phrasings per template and 750 base runs. An earlier draft of that table said 600, counting the paraphrases only and omitting the canonical phrasing.

Fault runs are scaled by `fault_runs.paraphrases_per_template` in `configs/experiment.yaml` (default 1 = canonical phrasing only → 450 runs). The default keeps every template and topology represented at a third of the cost of the full 750 × 3 design; set it to 5 to reproduce that.

## Quick start

```bash
# Windows (use the py launcher and pin the version -- a venv created from a
# WSL shell inside a Windows checkout yields a .venv whose python cannot run)
py -3.11 -m venv .venv && .venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv && source .venv/bin/activate

pip install -e ".[model,dev,report]" # model: lightgbm, imbalanced-learn; report: reportlab

# 1. fetch datasets + build question banks
python scripts/download_datasets.py
python scripts/build_question_banks.py

# 2. corpus (needs Ollama: `ollama pull qwen3:8b`)
python scripts/generate_runs.py --no-mock --limit 4 --out results/smoke.jsonl   # smoke test
python scripts/generate_runs.py --no-mock                                        # 750 runs, ~1.7 h on an M3
python scripts/label_runs.py --runs results/runs.jsonl

# 3. classifier (block D)
python scripts/train_classifier.py --runs results/runs.jsonl

# 4. fault injection + robustness (blocks E/F)
python scripts/generate_fault_runs.py --no-mock
python scripts/label_runs.py --runs results/runs_faults.jsonl
python scripts/robustness.py

# 5. relate the two (block G), the deployable gate, figures, final report
python scripts/relate.py
python scripts/gate.py fit                                   # -> results/gate.joblib
python scripts/gate.py score --runs new_runs.jsonl --budget 0.2   # hold or release new answers
python scripts/figures.py
python report/build.py                  # -> Shaurya_Pragun_FDS_Project_Report.pdf
```

`make all` runs the whole chain (~3 h on an M3), `make report` builds the figures and the PDF, `make test` runs the suite. Pass `--no-mock` to
the generators to fail loudly instead of falling back to `MockLLM` — worth doing
whenever the numbers are meant to be real.

## Base LLM

The harness targets a **local open model via Ollama**, temperature 0, matching the report's "one fixed low-cost model". No cloud API key required; set `AGENTREL_MODEL` to override.

The model is **`qwen3:8b` with thinking off**, picked by benchmarking four local models on the canonical phrasing of every template × 3 topologies (150 runs each; `generate_runs.py --canonical-only` reproduces the design):

| model | failure rate | hit the 12-step cap | used no tool | s / run |
|---|---|---|---|---|
| llama3 (8B) | 59% | 7% | 39% | 14 |
| llama3.1:8b | 44% | 24% | 1% | 28 |
| qwen2.5:7b | 54% | 5% | 4% | 30 |
| **qwen3:8b, thinking off** | 48% | 11% | 2% | 17 |

Scored on the 41 templates unaffected by two harness issues found during the benchmark (a leaky schema preamble and the missing quantile op, both since fixed). llama3.1 and qwen3 are tied within noise; qwen3 is ~40% faster and loops half as often. llama3 as a single ReAct agent never calls a tool and fails 90% of the time. Run one model at a time: two 8B models on a 16 GB machine evict each other from memory on every request.
