"""Uplift evaluation: Qini curve, uplift@k, and policy value vs baselines.

Key idea: we never observe a single user's true effect, so we evaluate a *ranking*.
Sort users by predicted uplift; walk down the list; at each prefix compare the realized
outcome rate of treated vs control within that prefix. A good model puts the persuadables
on top -> steep early gain -> high Qini.
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd

from .features import split_xyt

# np.trapz deprecated in NumPy 2.x -> trapezoid; keep a fallback for older installs
_trapz = getattr(np, "trapezoid", np.trapz)


def qini_curve(uplift_score, treatment, outcome):
    """Return (fraction_targeted, cumulative_incremental_gain) for the Qini curve."""
    order = np.argsort(-np.asarray(uplift_score))
    t = np.asarray(treatment)[order]
    y = np.asarray(outcome)[order]

    n = len(t)
    n_t = np.cumsum(t)
    n_c = np.cumsum(1 - t)
    y_t = np.cumsum(y * t)
    y_c = np.cumsum(y * (1 - t))

    # incremental gain = treated responders - control responders scaled to treated pop
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = y_t - y_c * np.where(n_c > 0, n_t / n_c, 0.0)
    frac = np.arange(1, n + 1) / n
    return frac, np.nan_to_num(gain)


def qini_coefficient(uplift_score, treatment, outcome) -> float:
    """Area between the model's Qini curve and the random-targeting diagonal."""
    frac, gain = qini_curve(uplift_score, treatment, outcome)
    rand = gain[-1] * frac  # random targeting = straight line to the total gain
    return float(_trapz(gain - rand, frac))


def uplift_at_k(uplift_score, treatment, outcome, k: float = 0.3) -> float:
    """Incremental responders captured by treating the top-k fraction."""
    frac, gain = qini_curve(uplift_score, treatment, outcome)
    idx = max(0, int(k * len(frac)) - 1)
    return float(gain[idx])


def policy_value(uplift_score, treatment, outcome, treat_rate: float = 0.3) -> dict:
    """Expected outcome of: treat top `treat_rate` by score, vs random, vs treat-all.

    Uses only realized outcomes (no counterfactual peeking) via the standard
    treated/control-rate decomposition within the chosen group.
    """
    s = np.asarray(uplift_score)
    t = np.asarray(treatment)
    y = np.asarray(outcome)
    thresh = np.quantile(s, 1 - treat_rate)
    targeted = s >= thresh

    def grp_rate(mask, tt):
        m = mask & (t == tt)
        return y[m].mean() if m.any() else 0.0

    model = grp_rate(targeted, 1) * treat_rate + grp_rate(~targeted, 0) * (1 - treat_rate)
    treat_all = y[t == 1].mean() if (t == 1).any() else 0.0
    random = treat_all * treat_rate + (y[t == 0].mean() if (t == 0).any() else 0.0) * (1 - treat_rate)
    return {"model_policy": model, "treat_all": treat_all, "random": random}


def validate_against_truth(uplift_score, true_uplift) -> dict:
    """Sanity check only available on simulated data: does predicted ranking match truth?"""
    s = pd.Series(uplift_score)
    u = pd.Series(np.asarray(true_uplift))
    return {"spearman_vs_true": float(s.corr(u, method="spearman"))}


def main() -> None:
    ap = argparse.ArgumentParser(description="evaluate an uplift model")
    ap.add_argument("--data", default="data/processed/experiment.parquet")
    ap.add_argument("--model", default="artifacts/xlearner.pkl")
    ap.add_argument("--treat-rate", type=float, default=0.3)
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    with open(args.model, "rb") as f:
        bundle = pickle.load(f)
    model, features = bundle["model"], bundle["features"]
    X, t, y = split_xyt(df, features)
    score = model.predict_uplift(X.values)

    print(f"Qini coefficient : {qini_coefficient(score, t, y):.4f}")
    print(f"uplift@30%       : {uplift_at_k(score, t, y, 0.3):.1f} incremental responders")
    pv = policy_value(score, t, y, args.treat_rate)
    print(f"policy value     : model={pv['model_policy']:.4f} "
          f"random={pv['random']:.4f} treat_all={pv['treat_all']:.4f}")
    if "true_uplift" in df:
        print(f"spearman vs truth: {validate_against_truth(score, df['true_uplift'])['spearman_vs_true']:.3f}")


if __name__ == "__main__":
    main()
