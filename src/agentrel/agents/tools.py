"""Tool registry — the sandboxed data operations an agent may call.

Each tool is a node in the tool-interaction graph G=(V,E) (report 3.4). Tools
operate on a small in-memory session state (the loaded tables and a scratch
area) and return JSON-serialisable results. A tool raising is a *loud* failure
(the agent can see it and retry); a tool returning a plausible-but-wrong value
is the *silent* failure this project studies.

Tools intentionally kept simple and deterministic so that, once the agent
decides WHAT to do, the computation itself is not a source of noise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd


@dataclass
class ToolResult:
    ok: bool
    value: Any = None
    error: str | None = None
    repr: str = ""


@dataclass
class Session:
    """Mutable state threaded through a single agent run."""
    tables: dict[str, pd.DataFrame]
    scratch: dict[str, Any] = field(default_factory=dict)

    def primary(self) -> pd.DataFrame:
        if "main" in self.tables:
            return self.tables["main"]
        # multi-table dataset: default to the largest table
        return max(self.tables.values(), key=len)


ToolFn = Callable[[Session, dict[str, Any]], ToolResult]
_REGISTRY: dict[str, ToolFn] = {}


def tool(name: str) -> Callable[[ToolFn], ToolFn]:
    def deco(fn: ToolFn) -> ToolFn:
        _REGISTRY[name] = fn
        return fn
    return deco


def _ok(value: Any) -> ToolResult:
    r = repr(value)
    return ToolResult(ok=True, value=value, repr=r[:200])


def _err(exc: Exception) -> ToolResult:
    return ToolResult(ok=False, error=f"{exc.__class__.__name__}: {exc}")


@tool("list_tables")
def _list_tables(s: Session, args: dict) -> ToolResult:
    try:
        return _ok({name: {"rows": len(df), "cols": list(df.columns)}
                    for name, df in s.tables.items()})
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


@tool("load_table")
def _load_table(s: Session, args: dict) -> ToolResult:
    try:
        name = args.get("table", "main")
        df = s.tables.get(name, s.primary())
        s.scratch["df"] = df
        return _ok({"loaded": name, "rows": len(df), "columns": list(df.columns)})
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


@tool("filter_rows")
def _filter_rows(s: Session, args: dict) -> ToolResult:
    """args: {query: pandas-query-string}"""
    try:
        df = s.scratch.get("df", s.primary())
        q = args.get("query")
        out = df.query(q) if q else df
        s.scratch["df"] = out
        return _ok({"rows_after": len(out)})
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


@tool("aggregate")
def _aggregate(s: Session, args: dict) -> ToolResult:
    """args: {op: mean|sum|median|count|max|min|nunique, column: str}"""
    try:
        df = s.scratch.get("df", s.primary())
        op = args.get("op", "count")
        col = args.get("column")
        if op == "count":
            val = int(len(df))
        else:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            val = float(getattr(series, op)())
        s.scratch["last_value"] = val
        return _ok(val)
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


@tool("group_aggregate")
def _group_aggregate(s: Session, args: dict) -> ToolResult:
    """args: {by: col, op: mean|sum|..., column: col, pick: idxmax|idxmin|value}"""
    try:
        df = s.scratch.get("df", s.primary())
        by, col, op = args["by"], args.get("column"), args.get("op", "mean")
        pick = args.get("pick", "idxmax")
        if op == "count":
            grouped = df.groupby(by).size()
        else:
            grouped = df.groupby(by)[col].agg(op)
        if pick in ("idxmax", "idxmin"):
            val = getattr(grouped, pick)()
        else:
            val = grouped.to_dict()
        s.scratch["last_value"] = val
        return _ok(val)
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


@tool("join_tables")
def _join_tables(s: Session, args: dict) -> ToolResult:
    """args: {left, right, on, how}"""
    try:
        left = s.scratch.get("df", s.tables[args["left"]])
        right = s.tables[args["right"]]
        out = left.merge(right, on=args["on"], how=args.get("how", "left"))
        s.scratch["df"] = out
        return _ok({"rows_after": len(out), "columns": list(out.columns)})
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


@tool("compute_missing_rate")
def _missing_rate(s: Session, args: dict) -> ToolResult:
    try:
        df = s.scratch.get("df", s.primary())
        col = args["column"]
        return _ok(float(df[col].isna().mean()))
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


def available_tools() -> list[str]:
    return sorted(_REGISTRY)


def call_tool(name: str, session: Session, args: dict[str, Any]) -> ToolResult:
    fn = _REGISTRY.get(name)
    if fn is None:
        return ToolResult(ok=False, error=f"UnknownTool: {name!r}")
    return fn(session, args)


# --------------------------------------------------------------------------- #
# Schema digest — what the agent is allowed to know about the data up front.
# --------------------------------------------------------------------------- #
# Without this the model cannot do real analysis: it does not know the column
# names, so it either calls tools with invalid args or skips tools entirely and
# guesses. A guessed answer produces an empty tool-interaction graph, which
# makes the graph features degenerate, so the digest is load-bearing for the
# whole measurement layer. It is deliberately *descriptive only* (names, dtypes,
# cardinality, missing rate) — never the answer to any question.
_MAX_LEVELS = 8


def schema_digest(session: Session, max_cols: int = 40) -> str:
    """Compact, deterministic description of the session's tables.

    Strictly qualitative about quantities. An earlier version printed the row
    count, each column's missing rate and its distinct-value count, which are
    the literal answers to the count and missing-rate templates: six templates
    could be answered straight off the preamble, and they failed at 23%
    against 47% for the rest. A column is now reported as *having* missing
    values, never how many -- the number has to come from a tool call.
    """
    lines: list[str] = []
    for name, df in session.tables.items():
        lines.append(f"table {name}: columns {list(df.columns)[:max_cols]}")
        for col in list(df.columns)[:max_cols]:
            s = df[col]
            desc = f"  - {col} ({s.dtype}"
            if bool(s.isna().any()):
                desc += ", has missing values"
            if pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
                levels = s.dropna().unique()
                # the level names are the option set a filter needs; the count
                # of them is itself a cardinality answer, so it is withheld
                if len(levels) <= _MAX_LEVELS:
                    desc += f", values: {sorted(map(str, levels))}"
                else:
                    desc += ", many distinct values"
            lines.append(desc + ")")
        if len(df.columns) > max_cols:
            lines.append("  ... and further columns")
    return "\n".join(lines)
