"""API contract: budget maps to a threshold, and a bigger budget lowers the bar."""
import pickle
from pathlib import Path

import numpy as np
import pytest

from src.data import simulate_rct
from src.features import feature_columns, split_xyt
from src.learners import TLearner


@pytest.fixture
def model_bundle(tmp_path, monkeypatch):
    """Train a tiny model, persist a bundle with score_ref, point the API at it."""
    df = simulate_rct(n=6_000, seed=5)
    feats = feature_columns(df)
    X, t, y = split_xyt(df, feats)
    model = TLearner().fit(X.values, t, y)
    score_ref = np.sort(model.predict_uplift(X.values).astype(float))

    path = tmp_path / "m.pkl"
    with open(path, "wb") as f:
        pickle.dump({"model": model, "features": feats, "learner": "tlearner",
                     "score_ref": score_ref}, f)
    monkeypatch.setenv("MODEL_PATH", str(path))
    return path, feats


def _client(model_path):
    pytest.importorskip("httpx", reason="TestClient needs httpx")
    import importlib

    import api.main as m
    importlib.reload(m)  # re-read MODEL_PATH env
    from fastapi.testclient import TestClient
    return TestClient(m.app)


def test_bigger_budget_lowers_threshold(model_bundle):
    path, feats = model_bundle
    client = _client(path)
    t30 = client.get("/budget", params={"budget": 0.3}).json()["threshold"]
    t90 = client.get("/budget", params={"budget": 0.9}).json()["threshold"]
    assert t90 <= t30


def test_score_decision_and_reason(model_bundle):
    path, feats = model_bundle
    client = _client(path)
    payload = {f: (1.5 if f == "x3" else 0.0) for f in feats}  # persuadable profile
    r = client.post("/score", json={"features": payload, "budget": 0.3}).json()
    assert r["decision"] in {"treat", "skip"}
    assert "budget" in r["reason"] and 0.0 <= r["percentile"] <= 1.0
