"""Data layer: simulate an RCT with KNOWN true uplift, or load a real uplift dataset.

The simulated path is deliberate: because we know the true per-user treatment effect,
we can validate that our Qini / policy-value evaluator is correct before trusting it on
real data where the counterfactual is never observable.

Schema produced everywhere downstream:
    - covariate columns: x0..x{d-1}
    - 'treatment': 0/1   (randomly assigned -> unconfounded)
    - 'outcome'  : 0/1   (e.g. converted)
    - 'true_uplift': float  (ONLY present for simulated data; the held-out truth)
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [f"x{i}" for i in range(8)]

# Kevin Hillstrom MineThatData e-mail challenge (2008) — the classic uplift benchmark.
# Canonical source (the author's own host). 64k rows, 12 columns.
HILLSTROM_URL = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
HILLSTROM_NUMERIC = ["recency", "history", "mens", "womens", "newbie"]
HILLSTROM_CATEGORICAL = ["history_segment", "zip_code", "channel"]


def simulate_rct(n: int = 50_000, d: int = 8, seed: int = 7) -> pd.DataFrame:
    """Randomized experiment with heterogeneous, sign-varying treatment effect.

    Effect depends on covariates so that only a subset are 'persuadable' (positive
    uplift), some are unaffected, and some are 'sleeping dogs' (negative uplift) —
    exactly the structure that makes uplift modeling beat response modeling.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))

    # baseline conversion propensity (what a response model would chase)
    base = _sigmoid(0.6 * X[:, 0] - 0.4 * X[:, 1] + 0.3 * X[:, 2])

    # true treatment effect: positive for some, ~0 for most, negative for a few
    tau = 0.25 * np.maximum(X[:, 3], 0) - 0.15 * np.maximum(X[:, 4], 0)

    treatment = rng.binomial(1, 0.5, size=n)  # randomized -> 50/50
    p_treat = np.clip(base + tau, 0.001, 0.999)
    p_ctrl = np.clip(base, 0.001, 0.999)
    p = np.where(treatment == 1, p_treat, p_ctrl)
    outcome = rng.binomial(1, p)

    df = pd.DataFrame(X, columns=FEATURES)
    df["treatment"] = treatment
    df["outcome"] = outcome
    df["true_uplift"] = p_treat - p_ctrl  # the held-out truth (simulated only)
    return df


def load_hillstrom(
    campaign: str = "any",
    outcome: str = "visit",
    cache: str = "data/raw/hillstrom.csv",
    url: str = HILLSTROM_URL,
) -> pd.DataFrame:
    """Load + normalize the Hillstrom email dataset to [features..., treatment, outcome].

    64k customers randomized into 3 arms: 'Womens E-Mail', 'Mens E-Mail', 'No E-Mail'.
    Because assignment is randomized, treatment is unconfounded — a valid uplift setup.

    Args:
        campaign: which arm is "treated" vs the 'No E-Mail' control —
            'any'    -> either email   (largest sample, general lift)
            'womens' -> Womens E-Mail only
            'mens'   -> Mens E-Mail only
        outcome: 'visit' (default, ~15%) or 'conversion' (~0.9%, very sparse).
        cache: local CSV path; downloaded once if absent.
    """
    df = _read_hillstrom_csv(cache, url)

    seg = df["segment"].astype(str)
    treated_label = {"any": None, "womens": "Womens E-Mail", "mens": "Mens E-Mail"}
    if campaign not in treated_label:
        raise ValueError(f"campaign must be one of {list(treated_label)}")

    control = seg == "No E-Mail"
    if campaign == "any":
        treated = ~control
    else:
        treated = seg == treated_label[campaign]
    keep = treated | control  # drop the other email arm for a clean 2-arm contrast
    df = df.loc[keep].copy()

    if outcome not in ("visit", "conversion"):
        raise ValueError("outcome must be 'visit' or 'conversion'")

    # one-hot categoricals, keep numerics; stable, serving-friendly column names
    X_num = df[HILLSTROM_NUMERIC].astype(float)
    X_cat = pd.get_dummies(df[HILLSTROM_CATEGORICAL], prefix=HILLSTROM_CATEGORICAL).astype(float)

    out = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    out["treatment"] = treated.loc[keep].astype(int).values
    out["outcome"] = df[outcome].astype(int).values
    return out


def _read_hillstrom_csv(cache: str, url: str) -> pd.DataFrame:
    """Read the cached CSV, downloading once if missing."""
    path = Path(cache)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading Hillstrom -> {path}")
        urllib.request.urlretrieve(url, path)  # noqa: S310 (trusted, pinned URL)
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    required = set(HILLSTROM_NUMERIC + HILLSTROM_CATEGORICAL + ["segment", "visit", "conversion"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Hillstrom CSV missing columns: {sorted(missing)}; got {list(df.columns)}")
    return df


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def main() -> None:
    ap = argparse.ArgumentParser(description="build experiment dataset")
    ap.add_argument("--dataset", choices=["simulate", "hillstrom"], default="simulate")
    ap.add_argument("--simulate", action="store_true", help="alias for --dataset simulate")
    ap.add_argument("--n", type=int, default=50_000, help="rows (simulate only)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--campaign", choices=["any", "womens", "mens"], default="any",
                    help="Hillstrom treated arm vs No-E-Mail control")
    ap.add_argument("--outcome", choices=["visit", "conversion"], default="visit",
                    help="Hillstrom outcome column")
    ap.add_argument("--out", default="data/processed/experiment.parquet")
    args = ap.parse_args()

    dataset = "simulate" if args.simulate else args.dataset
    if dataset == "hillstrom":
        df = load_hillstrom(campaign=args.campaign, outcome=args.outcome)
    else:
        df = simulate_rct(n=args.n, seed=args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {len(df):,} rows ({dataset}) -> {out}")
    print(f"  treat rate={df.treatment.mean():.3f}  outcome rate={df.outcome.mean():.3f}")
    if "true_uplift" in df:
        print(f"  true ATE={df.true_uplift.mean():.4f}")
    else:  # observed ATE = treated rate - control rate (valid because randomized)
        ate = df.loc[df.treatment == 1, "outcome"].mean() - df.loc[df.treatment == 0, "outcome"].mean()
        print(f"  observed ATE (treated-control)={ate:.4f}")


if __name__ == "__main__":
    main()
