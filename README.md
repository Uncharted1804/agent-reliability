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
│   ├── features/        # (stub for later 30%) static/dynamic/graph features
│   └── utils/           # config, io, logging helpers
├── datasets/            # downloaded raw data (gitignored) + question banks
├── configs/             # experiment config (topologies, faults, splits)
├── scripts/             # entry points: download, generate, label
├── tests/               # sanity tests for the data + labelling layer
└── results/             # run corpus, labels (gitignored)
```

## Status — current milestone (~30%)

Implemented: project scaffold; dataset acquisition + preprocessing; deterministic pandas ground-truth; question templates (10/dataset) with 4 paraphrases each; agent-run harness + trace schema; 3 topology implementations against a local model; labelling + log-cleaning module.

Not yet (later milestones): feature engineering (static/dynamic/graph), LightGBM training + baselines, fault-injection experiments, robustness ΔT,f with bootstrap CIs, and the step-(G) analysis linking predictors to robustness.

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
```

## Base LLM

The harness targets a **local open model via Ollama** (default `llama3.1`, temperature 0), matching the report's "one fixed low-cost model". Set `AGENTREL_MODEL` to override. No cloud API key required.
