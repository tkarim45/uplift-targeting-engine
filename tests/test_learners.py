"""Learner correctness: R-learner recovers signal; econml cross-check agrees (if installed)."""
import pytest
from scipy.stats import spearmanr

from src.data import simulate_rct
from src.evaluate import qini_coefficient
from src.features import split_xyt
from src.learners import RLearner


def _data(n=15_000, seed=11):
    df = simulate_rct(n=n, seed=seed)
    X, t, y = split_xyt(df)
    return df, X.values, t, y


def test_rlearner_positive_qini():
    df, X, t, y = _data()
    score = RLearner(n_folds=3).fit(X, t, y).predict_uplift(X)
    assert qini_coefficient(score, t, y) > 0


def test_rlearner_recovers_truth_ordering():
    df, X, t, y = _data()
    score = RLearner(n_folds=3).fit(X, t, y).predict_uplift(X)
    assert spearmanr(score, df["true_uplift"].values).statistic > 0.2


def test_econml_agrees_with_scratch():
    """If econml is installed, its R-learner must rank users like ours (Spearman > 0.6)."""
    pytest.importorskip("econml", reason="optional cross-check dependency")
    from src.learners import EconmlRLearner

    df, X, t, y = _data(n=8_000)
    s_ours = RLearner(n_folds=3).fit(X, t, y).predict_uplift(X)
    s_ref = EconmlRLearner(n_folds=3).fit(X, t, y).predict_uplift(X)
    assert spearmanr(s_ours, s_ref).statistic > 0.6
