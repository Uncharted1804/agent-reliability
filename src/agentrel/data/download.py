"""Materialise the 5 datasets into datasets/raw/<name>/.

Strategy per dataset:
  * if it has URLs, try to download; on any failure fall back to a synthesiser
    (if one exists) so the pipeline still runs offline.
  * if it has no URLs, synthesise directly.

Every materialised dataset gets a meta.json recording its tables and whether it
is real or synthetic.
"""
from __future__ import annotations

import io

import pandas as pd
import requests

from .registry import ADULT_COLUMNS, SPECS, DatasetSpec, get_spec, write_meta

_UA = {"User-Agent": "agentrel-dataset-fetch/0.1"}


def _try_download(url: str, timeout: int = 30) -> str | None:
    try:
        resp = requests.get(url, headers=_UA, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except Exception as exc:  # noqa: BLE001 — any network issue -> fallback
        print(f"    download failed ({exc.__class__.__name__}): {url}")
        return None


def _materialise_titanic(spec: DatasetSpec) -> dict | None:
    text = _try_download(spec.urls["main"])
    if text is None:
        return None
    df = pd.read_csv(io.StringIO(text))
    df.to_csv(spec.dir / "main.csv", index=False)
    return {"tables": {"main": "main.csv"}, "synthetic": False}


def _materialise_adult(spec: DatasetSpec) -> dict | None:
    text = _try_download(spec.urls["main"])
    if text is None:
        return None
    df = pd.read_csv(io.StringIO(text), header=None, names=ADULT_COLUMNS,
                     skipinitialspace=True, na_values="?")
    df.to_csv(spec.dir / "main.csv", index=False)
    return {"tables": {"main": "main.csv"}, "synthetic": False}


_DOWNLOADERS = {
    "titanic": _materialise_titanic,
    "adult_census": _materialise_adult,
}


def materialise(name: str, seed: int = 42, force: bool = False) -> dict:
    spec = get_spec(name)
    spec.dir.mkdir(parents=True, exist_ok=True)
    if (spec.dir / "meta.json").exists() and not force:
        print(f"  {name}: already present (use force=True to refresh)")
        import json
        return json.loads((spec.dir / "meta.json").read_text())

    meta = None
    if name in _DOWNLOADERS:
        print(f"  {name}: downloading…")
        meta = _DOWNLOADERS[name](spec)

    if meta is None and spec.synthesiser is not None:
        print(f"  {name}: synthesising controlled fallback…")
        meta = spec.synthesiser(spec.dir, seed)

    if meta is None:
        raise RuntimeError(f"could not materialise {name}: no download and no synthesiser")

    write_meta(spec, meta)
    tag = "synthetic" if meta.get("synthetic") else "real"
    print(f"  {name}: done ({tag}, {len(meta['tables'])} table(s))")
    return meta


def materialise_all(seed: int = 42, force: bool = False) -> None:
    print("Materialising datasets:")
    for name in SPECS:
        materialise(name, seed=seed, force=force)
