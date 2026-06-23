"""Feature / split helpers shared by training, eval, and serving.

The dataset schema is: feature columns + 'treatment' + 'outcome' (+ optional
'true_uplift' on simulated data). Feature columns are derived dynamically so the same
pipeline works for the simulated RCT (x0..x7) and real data (Hillstrom, Criteo, ...).
"""
from __future__ import annotations

import pandas as pd

TREAT_COL = "treatment"
OUTCOME_COL = "outcome"
TRUTH_COL = "true_uplift"
_RESERVED = {TREAT_COL, OUTCOME_COL, TRUTH_COL}


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Everything that isn't treatment / outcome / ground-truth is a feature."""
    return [c for c in df.columns if c not in _RESERVED]


def split_xyt(df: pd.DataFrame, features: list[str] | None = None):
    """Return (X, treatment, outcome). `features` pins column order for serving parity."""
    cols = features if features is not None else feature_columns(df)
    X = df[cols].copy()
    t = df[TREAT_COL].astype(int).values
    y = df[OUTCOME_COL].astype(int).values
    return X, t, y


def feature_vector(payload: dict, features: list[str]) -> pd.DataFrame:
    """Build a single-row frame matching the trained feature order; missing -> 0.0.

    One-hot columns absent from the payload default to 0 — correct for a category the
    user didn't set. TODO: swap zero-fill for trained imputation stats on numerics.
    """
    row = {f: float(payload.get(f, 0.0)) for f in features}
    return pd.DataFrame([row], columns=features)
