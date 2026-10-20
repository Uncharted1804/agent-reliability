"""Agent-run generator (block A in the workflow).

Crosses every (dataset, question, paraphrase) with every topology, runs the
agent, and emits one RunTrace per run to a JSONL corpus. This is the single
corpus that feeds both the failure predictor and the robustness study.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from ..agents.llm import build_llm
from ..agents.tools import Session
from ..data.registry import load_tables
from ..utils.config import REPO_ROOT, Config
from ..utils.io import append_jsonl
from ..harness.trace import RunTrace
from ..topology import build_topology

QUESTION_BANK_DIR = REPO_ROOT / "datasets" / "question_banks"


def _load_bank(dataset: str) -> dict:
    path = QUESTION_BANK_DIR / f"{dataset}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"question bank for {dataset} missing — run scripts/build_question_banks.py")
    return json.loads(path.read_text())


def generate(cfg: Config, *, out_path: str | Path | None = None,
             limit: int | None = None, allow_mock: bool = True) -> Path:
    out_path = Path(out_path) if out_path else cfg.log_dir / "runs.jsonl"
    if out_path.exists():
        out_path.unlink()  # fresh corpus
    llm = build_llm(cfg.llm, allow_mock=allow_mock)

    # total is known up front, so a multi-hour generation can report progress
    total = 0
    for dataset in cfg.datasets:
        bank = _load_bank(dataset)
        qs = bank["questions"][:limit] if limit else bank["questions"]
        total += len(qs) * len(cfg.topologies)
    print(f"Generating {total} runs with {llm.__class__.__name__} "
          f"({cfg.llm.model}) -> {out_path}", flush=True)

    t_start = time.time()
    n_runs = 0
    for dataset in cfg.datasets:
        bank = _load_bank(dataset)
        tables = load_tables(dataset)
        questions = bank["questions"]
        if limit:
            questions = questions[:limit]
        for q in questions:
            for topo_name in cfg.topologies:
                run_id = f"{q['question_id']}::{topo_name}"
                trace = RunTrace(
                    run_id=run_id, dataset=dataset, template_id=q["template_id"],
                    question_id=q["question_id"],
                    paraphrase_index=q["paraphrase_index"], topology=topo_name,
                    model=cfg.llm.model, question_text=q["text"],
                )
                # fresh session per run so state never leaks between runs
                session = Session(tables={k: v.copy() for k, v in tables.items()})
                topo = build_topology(topo_name, llm, max_steps=cfg.max_agent_steps)
                try:
                    topo.run(trace, session)
                except Exception as exc:  # noqa: BLE001 — harness-level failure
                    trace.harness_error = f"{exc.__class__.__name__}: {exc}"
                    trace.finish(None)
                append_jsonl(out_path, trace.to_dict())
                n_runs += 1
                if n_runs % 10 == 0 or n_runs == total:
                    elapsed = time.time() - t_start
                    rate = elapsed / n_runs
                    eta = rate * (total - n_runs)
                    print(f"  [{n_runs}/{total}] {dataset} {topo_name} "
                          f"{rate:.1f}s/run  eta {eta/60:.0f}m", flush=True)

    print(f"Generated {n_runs} runs -> {out_path}")
    return out_path
