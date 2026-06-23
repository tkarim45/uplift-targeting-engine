"""Correctness tests for the uplift evaluator, anchored on simulated ground truth.

The point of these: prove the Qini / policy-value implementation is right BEFORE trusting
it on real data where the true effect is unobservable.
"""
import numpy as np

from src.data import simulate_rct
from src.evaluate import policy_value, qini_coefficient, validate_against_truth
from src.features import split_xyt
from src.learners import TLearner


def test_qini_positive_for_informed_ranking():
    """A model trained on the true generating process should beat random (Qini > 0)."""
    df = simulate_rct(n=20_000, seed=1)
    X, t, y = split_xyt(df)
    model = TLearner().fit(X.values, t, y)
    score = model.predict_uplift(X.values)
    assert qini_coefficient(score, t, y) > 0


def test_random_score_has_near_zero_qini():
    """A random ranking carries no information -> Qini ~ 0."""
    rng = np.random.default_rng(0)
    df = simulate_rct(n=20_000, seed=2)
    _, t, y = split_xyt(df)
    score = rng.normal(size=len(df))
    assert abs(qini_coefficient(score, t, y)) < qini_coefficient(
        df["true_uplift"].values, t, y
    )


def test_policy_beats_random_with_true_uplift():
    df = simulate_rct(n=20_000, seed=3)
    _, t, y = split_xyt(df)
    pv = policy_value(df["true_uplift"].values, t, y, treat_rate=0.3)
    assert pv["model_policy"] >= pv["random"]


def test_spearman_recovers_truth_ordering():
    df = simulate_rct(n=20_000, seed=4)
    X, t, y = split_xyt(df)
    model = TLearner().fit(X.values, t, y)
    score = model.predict_uplift(X.values)
    assert validate_against_truth(score, df["true_uplift"])["spearman_vs_true"] > 0.2
