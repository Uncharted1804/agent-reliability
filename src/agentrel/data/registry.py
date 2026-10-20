"""Dataset registry — the 5 public tabular datasets of increasing difficulty
described in report section 3.3.

    titanic        clean single table                (UCI / stanford mirror)
    adult_census   clean single table, categoricals  (UCI)
    nyc_taxi       larger, outliers, mixed types      (sampled)
    missingness    heavy missing values               (synthesised, controlled)
    olist          multi-table e-commerce (joins)     (synthesised, controlled)

Each entry knows how to materialise itself into datasets/raw/<name>/ as one or
more CSVs plus a small meta.json. Where a stable public URL exists we download
it; where it does not (or the box is offline) we synthesise a dataset with the
*same shape characteristics* the report cares about, so the whole pipeline is
runnable without network access. Synthesised tables are clearly flagged in
meta.json so they can be swapped for the real download later.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..utils.config import REPO_ROOT

RAW_DIR = REPO_ROOT / "datasets" / "raw"


@dataclass
class DatasetSpec:
    name: str
    description: str
    urls: dict[str, str] = field(default_factory=dict)      # logical table name -> url
    primary_table: str = "main"
    multi_table: bool = False
    synthesiser: Callable[[Path, int], dict] | None = None  # fallback builder

    @property
    def dir(self) -> Path:
        return RAW_DIR / self.name


# --------------------------------------------------------------------------- #
# Synthesisers — controlled fallbacks that reproduce the report's shape traits.
# --------------------------------------------------------------------------- #
def _syn_missingness(out: Path, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = 3000
    df = pd.DataFrame(
        {
            "id": np.arange(n),
            "age": rng.normal(45, 15, n).round(1),
            "income": rng.lognormal(10.5, 0.6, n).round(2),
            "score": rng.normal(0, 1, n).round(3),
            "region": rng.choice(["N", "S", "E", "W"], n),
            "tenure_months": rng.integers(0, 240, n),
            "churned": rng.integers(0, 2, n),
        }
    )
    # inject heavy, column-varying missingness (this IS the feature we measure)
    for col, frac in {"age": 0.35, "income": 0.45, "score": 0.25, "region": 0.15}.items():
        mask = rng.random(n) < frac
        df.loc[mask, col] = np.nan
    df.to_csv(out / "main.csv", index=False)
    return {"tables": {"main": "main.csv"}, "synthetic": True}


def _syn_olist(out: Path, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n_cust, n_orders, n_items = 800, 2000, 5000
    customers = pd.DataFrame(
        {
            "customer_id": [f"c{i:05d}" for i in range(n_cust)],
            "state": rng.choice(["SP", "RJ", "MG", "RS", "PR"], n_cust),
        }
    )
    orders = pd.DataFrame(
        {
            "order_id": [f"o{i:06d}" for i in range(n_orders)],
            "customer_id": rng.choice(customers["customer_id"], n_orders),
            "status": rng.choice(["delivered", "shipped", "canceled"], n_orders,
                                 p=[0.85, 0.1, 0.05]),
            "order_month": rng.integers(1, 13, n_orders),
        }
    )
    items = pd.DataFrame(
        {
            "order_id": rng.choice(orders["order_id"], n_items),
            "product_category": rng.choice(
                ["electronics", "home", "toys", "books", "garden"], n_items),
            "price": rng.gamma(2.0, 40.0, n_items).round(2),
            "freight": rng.gamma(1.5, 8.0, n_items).round(2),
        }
    )
    customers.to_csv(out / "customers.csv", index=False)
    orders.to_csv(out / "orders.csv", index=False)
    items.to_csv(out / "order_items.csv", index=False)
    return {
        "tables": {
            "customers": "customers.csv",
            "orders": "orders.csv",
            "order_items": "order_items.csv",
        },
        "synthetic": True,
    }


def _syn_titanic(out: Path, seed: int) -> dict:
    """Clean single table with the columns the titanic templates expect."""
    rng = np.random.default_rng(seed)
    n = 891
    pclass = rng.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55])
    sex = rng.choice(["female", "male"], n, p=[0.35, 0.65])
    # survival depends on class + sex so group comparisons are meaningful
    base = 0.08 + (pclass == 1) * 0.30 + (pclass == 2) * 0.15 + (sex == "female") * 0.45
    survived = (rng.random(n) < np.clip(base, 0, 0.95)).astype(int)
    age = rng.normal(29, 13, n).round(1)
    age[rng.random(n) < 0.20] = np.nan                      # ~20% ages missing
    fare = (10 + (3 - pclass) * 28 + rng.gamma(2, 6, n)).round(4)
    fare[rng.random(n) < 0.005] *= 8                        # a few fare outliers
    df = pd.DataFrame({
        "PassengerId": np.arange(1, n + 1),
        "Survived": survived,
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "Fare": fare,
        "Embarked": rng.choice(["S", "C", "Q"], n, p=[0.72, 0.19, 0.09]),
    })
    df.to_csv(out / "main.csv", index=False)
    return {"tables": {"main": "main.csv"}, "synthetic": True}


def _syn_adult(out: Path, seed: int) -> dict:
    """Clean single table with the UCI Adult columns the adult templates expect."""
    rng = np.random.default_rng(seed)
    n = 6000
    age = rng.integers(17, 90, n)
    edu_num = rng.integers(1, 17, n)
    hours = np.clip(rng.normal(40, 12, n).round().astype(int), 1, 99)
    occupation = rng.choice(
        ["Prof-specialty", "Craft-repair", "Exec-managerial", "Adm-clerical",
         "Sales", "Other-service", "Machine-op-inspct", "Transport-moving"], n)
    education = rng.choice(
        ["HS-grad", "Some-college", "Bachelors", "Masters", "Assoc-voc",
         "Doctorate", "11th", "Prof-school"], n)
    # income probability rises with education-num, hours and age
    logit = -6 + 0.22 * edu_num + 0.03 * hours + 0.02 * age + rng.normal(0, 1, n)
    hi = rng.random(n) < 1 / (1 + np.exp(-logit))
    df = pd.DataFrame({
        "age": age,
        "workclass": rng.choice(["Private", "Self-emp", "Local-gov", "State-gov"], n),
        "fnlwgt": rng.integers(20000, 400000, n),
        "education": education,
        "education_num": edu_num,
        "marital_status": rng.choice(["Married", "Never-married", "Divorced"], n),
        "occupation": occupation,
        "relationship": rng.choice(["Husband", "Not-in-family", "Own-child", "Wife"], n),
        "race": rng.choice(["White", "Black", "Asian-Pac-Islander", "Other"], n),
        "sex": rng.choice(["Male", "Female"], n),
        "capital_gain": (rng.random(n) < 0.1) * rng.integers(0, 20000, n),
        "capital_loss": (rng.random(n) < 0.05) * rng.integers(0, 2000, n),
        "hours_per_week": hours,
        "native_country": rng.choice(["United-States", "Mexico", "India", "Philippines"], n),
        "income": np.where(hi, ">50K", "<=50K"),
    })
    df.to_csv(out / "main.csv", index=False)
    return {"tables": {"main": "main.csv"}, "synthetic": True}


def _syn_nyc_taxi(out: Path, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = 8000
    dist = rng.gamma(2.0, 1.6, n).round(2)
    # heavy-tailed fare with deliberate outliers + mixed-type junk
    fare = (2.5 + dist * 2.8 + rng.normal(0, 2, n)).round(2)
    fare[rng.random(n) < 0.01] *= 20          # outliers
    df = pd.DataFrame(
        {
            "trip_id": np.arange(n),
            "passengers": rng.integers(1, 7, n),
            "trip_distance": dist,
            "fare_amount": fare,
            "payment_type": rng.choice(["card", "cash", "CARD", "Cash", "2"], n),
            "pickup_hour": rng.integers(0, 24, n),
            "tip_amount": (fare * rng.uniform(0, 0.3, n)).round(2),
        }
    )
    df.loc[df.sample(frac=0.03, random_state=seed).index, "fare_amount"] = np.nan
    df.to_csv(out / "main.csv", index=False)
    return {"tables": {"main": "main.csv"}, "synthetic": True}


# --------------------------------------------------------------------------- #
# The registry.
# --------------------------------------------------------------------------- #
SPECS: dict[str, DatasetSpec] = {
    "titanic": DatasetSpec(
        name="titanic",
        description="Titanic passenger survival — clean single table.",
        urls={
            "main": "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/"
                    "stuff/titanic.csv",
        },
        synthesiser=_syn_titanic,
    ),
    "adult_census": DatasetSpec(
        name="adult_census",
        description="UCI Adult Census income — clean single table, many categoricals.",
        urls={
            "main": "https://archive.ics.uci.edu/ml/machine-learning-databases/"
                    "adult/adult.data",
        },
        synthesiser=_syn_adult,
    ),
    "nyc_taxi": DatasetSpec(
        name="nyc_taxi",
        description="NYC taxi trips sample — larger, outliers, mixed types.",
        synthesiser=_syn_nyc_taxi,
    ),
    "missingness": DatasetSpec(
        name="missingness",
        description="Controlled heavy-missingness table.",
        synthesiser=_syn_missingness,
    ),
    "olist": DatasetSpec(
        name="olist",
        description="Olist-style e-commerce — multi-table, needs joins.",
        multi_table=True,
        synthesiser=_syn_olist,
    ),
}

# Column names for the UCI Adult file, which ships headerless.
ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income",
]


def get_spec(name: str) -> DatasetSpec:
    if name not in SPECS:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(SPECS)}")
    return SPECS[name]


def write_meta(spec: DatasetSpec, meta: dict) -> None:
    meta = {"name": spec.name, "description": spec.description,
            "multi_table": spec.multi_table, "primary_table": spec.primary_table,
            **meta}
    with open(spec.dir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


def load_tables(name: str) -> dict[str, pd.DataFrame]:
    """Load a materialised dataset as {table_name: DataFrame}."""
    spec = get_spec(name)
    meta_path = spec.dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{name} not downloaded yet — run scripts/download_datasets.py")
    meta = json.loads(meta_path.read_text())
    out = {}
    for tbl, fname in meta["tables"].items():
        out[tbl] = pd.read_csv(spec.dir / fname)
    return out
