"""Cross-check: validate the from-scratch R-learner against econml's NonParamDML.

Both estimate the same residual-on-residual CATE. If our hand-rolled RLearner is correct,
its per-user uplift scores should rank users almost identically to econml's (high Spearman)
and post a comparable Qini on held-out data. On simulated data we also check both against
the known true uplift.

    python -m src.crosscheck --data data/processed/experiment.parquet
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

from .evaluate import qini_coefficient
from .features import feature_columns, split_xyt
from .learners import EconmlRLearner, RLearner


def main() -> None:
    ap = argparse.ArgumentParser(description="R-learner vs econml cross-check")
    ap.add_argument("--data", default="data/processed/experiment.parquet")
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    feats = feature_columns(df)
    train_df, test_df = train_test_split(df, test_size=args.test_size,
                                         random_state=args.seed, stratify=df["treatment"])
    Xtr, ttr, ytr = split_xyt(train_df, feats)
    Xte, tte, yte = split_xyt(test_df, feats)

    print(f"data={args.data}  features={len(feats)}  "
          f"train={len(train_df):,}  test={len(test_df):,}\n")

    ours = RLearner().fit(Xtr.values, ttr, ytr)
    s_ours = ours.predict_uplift(Xte.values)
    print(f"scratch R-learner : Qini={qini_coefficient(s_ours, tte, yte):.4f}")

    try:
        ref = EconmlRLearner().fit(Xtr.values, ttr, ytr)
    except ImportError as e:
        print(f"\n[skip] {e}")
        return
    s_ref = ref.predict_uplift(Xte.values)
    print(f"econml NonParamDML: Qini={qini_coefficient(s_ref, tte, yte):.4f}")

    rho = spearmanr(s_ours, s_ref).statistic
    print(f"\nSpearman(scratch, econml) = {rho:.3f}   "
          f"(>0.7 => implementations agree on the user ranking)")

    if "true_uplift" in test_df:
        tu = test_df["true_uplift"].values
        print(f"Spearman(scratch, truth)  = {spearmanr(s_ours, tu).statistic:.3f}")
        print(f"Spearman(econml,  truth)  = {spearmanr(s_ref, tu).statistic:.3f}")


if __name__ == "__main__":
    main()
