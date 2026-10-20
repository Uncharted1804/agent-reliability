"""Config loading. Single source of truth = configs/experiment.yaml, with
environment-variable overrides for the few things that change per machine."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiment.yaml"


@dataclass
class LLMConfig:
    provider: str = "ollama"
    model: str = "llama3.1"
    host: str = "http://localhost:11434"
    temperature: float = 0.0
    max_tokens: int = 1024
    request_timeout_s: int = 120


@dataclass
class Config:
    raw: dict[str, Any] = field(default_factory=dict)
    llm: LLMConfig = field(default_factory=LLMConfig)

    # convenience accessors
    @property
    def topologies(self) -> list[str]:
        return list(self.raw.get("topologies", []))

    @property
    def datasets(self) -> list[str]:
        return list(self.raw.get("datasets", []))

    @property
    def faults(self) -> list[str]:
        return list(self.raw.get("faults", []))

    @property
    def numeric_tolerance(self) -> float:
        return float(self.raw.get("labelling", {}).get("numeric_tolerance", 0.01))

    @property
    def max_agent_steps(self) -> int:
        return int(self.raw.get("runtime", {}).get("max_agent_steps", 12))

    @property
    def templates_per_dataset(self) -> int:
        return int(self.raw.get("questions", {}).get("templates_per_dataset", 10))

    @property
    def paraphrases_per_template(self) -> int:
        return int(self.raw.get("questions", {}).get("paraphrases_per_template", 4))

    @property
    def log_dir(self) -> Path:
        return REPO_ROOT / self.raw.get("runtime", {}).get("log_dir", "results")


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    llm_raw = dict(raw.get("base_llm", {}))
    # env overrides — handy on a machine with a specific Ollama model pulled
    llm_raw["model"] = os.environ.get("AGENTREL_MODEL", llm_raw.get("model", "llama3.1"))
    llm_raw["host"] = os.environ.get("AGENTREL_OLLAMA_HOST", llm_raw.get("host"))
    llm = LLMConfig(
        provider=llm_raw.get("provider", "ollama"),
        model=llm_raw.get("model", "llama3.1"),
        host=llm_raw.get("host", "http://localhost:11434"),
        temperature=float(llm_raw.get("temperature", 0.0)),
        max_tokens=int(llm_raw.get("max_tokens", 1024)),
        request_timeout_s=int(llm_raw.get("request_timeout_s", 120)),
    )
    return Config(raw=raw, llm=llm)
