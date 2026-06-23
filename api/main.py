"""FastAPI decision service: given a user's covariates, return estimated uplift and a
treat / don't-treat decision against a budget-derived threshold.

The threshold is NOT 0 — in production you treat down the ranked list until budget runs
out. Here it's configurable via UPLIFT_THRESHOLD (default 0.0 = treat anyone with
positive estimated effect). Set it from your Qini/budget analysis.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features import feature_vector

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/xlearner.pkl")
THRESHOLD = float(os.getenv("UPLIFT_THRESHOLD", "0.0"))

app = FastAPI(title="Uplift Targeting Engine", version="0.1.0")
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


class ScoreRequest(BaseModel):
    features: dict = Field(..., description="covariate map, e.g. {'x0': 0.4, 'x3': 1.1}")


class ScoreResponse(BaseModel):
    uplift: float
    decision: str
    threshold: float
    reason: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _bundle is not None, "model_path": MODEL_PATH}


@app.get("/features")
def features():
    """List the feature columns this model expects (handy when building requests)."""
    return {"features": _load()["features"]}


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    bundle = _load()
    X = feature_vector(req.features, bundle["features"])
    uplift = float(bundle["model"].predict_uplift(X.values)[0])
    treat = uplift >= THRESHOLD
    return ScoreResponse(
        uplift=round(uplift, 5),
        decision="treat" if treat else "skip",
        threshold=THRESHOLD,
        reason=("estimated incremental effect clears budget threshold"
                if treat else "below threshold — treating wastes budget (sure-thing or sleeping-dog)"),
    )
