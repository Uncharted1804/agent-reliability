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
| paraphrases / template | 4 |
| topologies | 3 |
| **base runs** | **5 × 10 × 4 × 3 = 600** |
| fault-injected runs | 600 × 3 = 1,800 |

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
│   └── utils/           # config, io, logging helpers
├── datasets/            # downloaded raw data (gitignored) + question banks
├── configs/             # experiment config (topologies, faults, splits)
├── scripts/             # entry points: download, generate, label
├── tests/               # sanity tests for the data + labelling layer
└── results/             # run corpus, labels (gitignored)
```

## Status — Milestone 2

Implemented: everything in Milestone 1, plus

* **feature engineering** (`features/`) — static (data + question), dynamic (trace), and graph features from the tool-interaction graph G=(V,E). The names in `features/schema.py` are the contract; each block asserts it produced exactly those keys.
* **silent-failure classifier** (`model/`) — class-weighted LightGBM with logistic-regression, SMOTE+LR, SMOTE+LightGBM and prior-dummy baselines, on the grouped-by-template 70/15/15 split. Reports F2 / PR-AUC / Brier, with the F2 threshold chosen on validation and frozen for test.
* **fault injection + robustness** (`faults/`) — `tool_failure`, `corrupted_output`, `dropped_context`, and Δ_T,f with cluster-bootstrap CIs (1000 resamples, resampled over templates).

Not yet: the step-(G) analysis linking predictor outputs to robustness.

### Methodology notes

Three choices are worth knowing before reading any number this produces:

* **Topology is not a feature.** It is the variable the robustness study manipulates, so letting the classifier read it would confound "this wiring fails more" with "this task is hard".
* **The split groups on `template_id`.** A template appears as 5 phrasings × 3 topologies, i.e. up to 15 near-duplicate rows. A row-wise split would put paraphrases of a training question in test and the reported F2 would mostly measure memorisation.
* **The bootstrap resamples templates, not runs.** Paraphrases of one template are near duplicates; resampling runs treats them as independent observations and gives CIs that are too narrow.

### Corpus actually generated

`expand_questions` emits the canonical phrasing *plus* 4 paraphrases, so the base corpus is 5 datasets × 10 templates × 5 phrasings × 3 topologies = **750 runs**, not the 600 in the table above (which counts paraphrases only).

Fault runs are scaled by `fault_runs.paraphrases_per_template` in `configs/experiment.yaml` (default 1 = canonical phrasing only → 450 runs). A run costs ~15 s against a CPU-served local model, so the full 750 × 3 design is ~10 h; the default keeps every template and topology represented at a third of the cost. Set it to 5 to reproduce the full design.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .

# 1. fetch datasets + build question banks
python scripts/download_datasets.py
python scripts/build_question_banks.py

# 2. (needs a local model served by Ollama, e.g. `ollama pull llama3.1`)
python scripts/generate_runs.py --config configs/experiment.yaml --limit 4   # smoke test
python scripts/label_runs.py --runs results/runs.jsonl

# 3. classifier (block D)
python scripts/train_classifier.py --runs results/runs.jsonl

# 4. fault injection + robustness (blocks E/F)
python scripts/generate_fault_runs.py --no-mock
python scripts/label_runs.py --runs results/runs_faults.jsonl
python scripts/robustness.py
```

`make all` runs the whole chain; `make test` runs the suite. Pass `--no-mock` to
the generators to fail loudly instead of falling back to `MockLLM` — worth doing
whenever the numbers are meant to be real.

## Base LLM

The harness targets a **local open model via Ollama** (default `llama3.1`, temperature 0), matching the report's "one fixed low-cost model". Set `AGENTREL_MODEL` to override. No cloud API key required.
