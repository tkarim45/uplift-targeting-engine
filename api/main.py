"""FastAPI decision service: given a user's covariates, return estimated uplift and a
treat / don't-treat decision against a BUDGET-DERIVED threshold.

The threshold is not a magic constant — it's derived from your budget. If you can only
afford to treat the top 30% of the population, the threshold is the 70th percentile of
the held-out uplift distribution (stored with the model as `score_ref` at train time).
Send a different `budget` per request to re-derive it on the fly.

Env:
    MODEL_PATH    path to the model bundle (default artifacts/xlearner.pkl)
    UPLIFT_BUDGET default fraction of the population to treat (default 0.30)
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features import feature_vector

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/xlearner.pkl")
DEFAULT_BUDGET = float(os.getenv("UPLIFT_BUDGET", "0.30"))

app = FastAPI(title="Uplift Targeting Engine", version="0.2.0")
_bundle = None


def _load():
    global _bundle
    if _bundle is None:
        p = Path(MODEL_PATH)
        if not p.exists():
            raise HTTPException(503, f"model not found at {p}; run src.train first")
        with open(p, "rb") as f:
            _bundle = pickle.load(f)
    return _bundle


def _threshold_for_budget(bundle: dict, budget: float) -> float:
    """Uplift cutoff such that ~`budget` fraction of the reference population qualifies.

    threshold = quantile(scores, 1 - budget). budget=1.0 -> treat all (min score);
    budget→0 -> treat only the very top. Falls back to 0.0 if no reference scores exist.
    """
    ref = bundle.get("score_ref")
    if ref is None or len(ref) == 0:
        return 0.0
    return float(np.quantile(np.asarray(ref, dtype=float), 1.0 - budget))


class ScoreRequest(BaseModel):
    features: dict = Field(..., description="covariate map, e.g. {'x0': 0.4, 'x3': 1.1}")
    budget: float | None = Field(
        None, ge=0.0, le=1.0,
        description="fraction of population you can treat; overrides UPLIFT_BUDGET",
    )


class ScoreResponse(BaseModel):
    uplift: float
    decision: str
    budget: float
    threshold: float
    percentile: float
    reason: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _bundle is not None, "model_path": MODEL_PATH}


@app.get("/features")
def features():
    """List the feature columns this model expects (handy when building requests)."""
    return {"features": _load()["features"]}


@app.get("/budget")
def budget_info(budget: float = DEFAULT_BUDGET):
    """Inspect the threshold a given budget maps to (no scoring)."""
    b = _load()
    return {
        "budget": budget,
        "threshold": _threshold_for_budget(b, budget),
        "has_reference": b.get("score_ref") is not None,
    }


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    bundle = _load()
    budget = DEFAULT_BUDGET if req.budget is None else req.budget
    threshold = _threshold_for_budget(bundle, budget)

    X = feature_vector(req.features, bundle["features"])
    uplift = float(bundle["model"].predict_uplift(X.values)[0])
    treat = uplift >= threshold

    ref = bundle.get("score_ref")
    pct = float((np.asarray(ref, float) <= uplift).mean()) if ref is not None and len(ref) else float("nan")

    return ScoreResponse(
        uplift=round(uplift, 5),
        decision="treat" if treat else "skip",
        budget=budget,
        threshold=round(threshold, 5),
        percentile=round(pct, 4),
        reason=(
            f"uplift in top {(1 - pct) * 100:.1f}% — clears the {budget:.0%}-budget threshold"
            if treat else
            f"below the {budget:.0%}-budget threshold — treating wastes spend (sure-thing or sleeping-dog)"
        ),
    )
