"""Base LLM client.

Targets a local open model served by Ollama (report: "one fixed low-cost
model, temperature 0"). Falls back to an explicit `MockLLM` when no server is
reachable, so the harness, topologies and labelling can all be exercised
offline and in CI. The mock is deterministic and NOT meant to produce correct
analysis — it exists to drive the control flow and produce traces.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

import requests

from ..utils.config import LLMConfig


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLM(Protocol):
    def complete(self, system: str, user: str) -> LLMResponse: ...


class OllamaLLM:
    """Thin client over the Ollama /api/chat endpoint."""

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

    def complete(self, system: str, user: str) -> LLMResponse:
        url = f"{self.cfg.host.rstrip('/')}/api/chat"
        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": self.cfg.temperature,
                "num_predict": self.cfg.max_tokens,
            },
        }
        resp = requests.post(url, json=payload, timeout=self.cfg.request_timeout_s)
        resp.raise_for_status()
        data = resp.json()
        return LLMResponse(
            text=data.get("message", {}).get("content", ""),
            prompt_tokens=int(data.get("prompt_eval_count", 0)),
            completion_tokens=int(data.get("eval_count", 0)),
        )


class MockLLM:
    """Deterministic offline stand-in.

    It emits tool-call JSON that the topologies know how to parse, walking a
    canned plan so traces have realistic structure (a few tool calls then a
    final answer). It does not look at the data, so its answers are usually
    wrong — which is fine: correctness is decided later by labelling, and a mix
    of right/wrong is exactly what the classifier needs.
    """

    def __init__(self, cfg: LLMConfig, seed: int = 0):
        self.cfg = cfg
        self._seed = seed
        self._counter = 0

    def complete(self, system: str, user: str) -> LLMResponse:
        self._counter += 1
        # crude state machine driven by how far into the conversation we are
        if "FINAL ANSWER" in user or self._counter % 3 == 0:
            # pseudo-random but deterministic numeric guess
            val = (hash((self._seed, user)) % 100000) / 100.0
            text = json.dumps({"action": "final", "answer": str(val)})
        else:
            tool = ["load_table", "filter_rows", "aggregate"][self._counter % 3]
            text = json.dumps({"action": "tool", "tool": tool,
                               "args": {"note": f"mock step {self._counter}"}})
        return LLMResponse(text=text, prompt_tokens=len(user)//4,
                           completion_tokens=len(text)//4)


def build_llm(cfg: LLMConfig, *, allow_mock: bool = True, seed: int = 0) -> LLM:
    """Return an OllamaLLM if the server answers, else a MockLLM."""
    if cfg.provider == "ollama":
        try:
            r = requests.get(f"{cfg.host.rstrip('/')}/api/tags", timeout=3)
            r.raise_for_status()
            return OllamaLLM(cfg)
        except Exception:  # noqa: BLE001
            if not allow_mock:
                raise
            print(f"  [llm] Ollama not reachable at {cfg.host} — using MockLLM")
            return MockLLM(cfg, seed=seed)
    if allow_mock:
        return MockLLM(cfg, seed=seed)
    raise RuntimeError(f"unsupported provider {cfg.provider!r}")
