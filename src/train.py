"""Fit a meta-learner and persist the artifact for serving + evaluation."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .evaluate import policy_value, qini_coefficient
from .features import feature_columns, split_xyt
from .learners import LEARNERS

# held-out uplift scores stored with the model so the API can map a budget (treat top
# k%) to a concrete decision threshold = quantile(scores, 1-k). Capped for artifact size.
_SCORE_REF_MAX = 20_000


def main() -> None:
    ap = argparse.ArgumentParser(description="train an uplift meta-learner")
    ap.add_argument("--data", default="data/processed/experiment.parquet")
    ap.add_argument("--learner", choices=list(LEARNERS), default="xlearner")
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", default="artifacts")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    features = feature_columns(df)  # pinned so serving uses identical columns/order
    train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=args.seed,
                                         stratify=df["treatment"])

    Xtr, ttr, ytr = split_xyt(train_df, features)
    Xte, tte, yte = split_xyt(test_df, features)

    model = LEARNERS[args.learner]().fit(Xtr.values, ttr, ytr)

    score = model.predict_uplift(Xte.values)
    qini = qini_coefficient(score, tte, yte)
    pv = policy_value(score, tte, yte, treat_rate=0.3)
    print(f"[{args.learner}] {len(features)} features, held-out Qini={qini:.4f}  "
          f"policy={pv['model_policy']:.4f} vs random={pv['random']:.4f}")

    # reference score distribution for budget -> threshold mapping at serve time
    score_ref = np.sort(np.asarray(score, dtype=float))
    if len(score_ref) > _SCORE_REF_MAX:
        idx = np.linspace(0, len(score_ref) - 1, _SCORE_REF_MAX).astype(int)
        score_ref = score_ref[idx]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.learner}.pkl"
    with open(path, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "features": features,
                "learner": args.learner,
                "score_ref": score_ref,  # sorted held-out uplift scores
            },
            f,
        )
    print(f"saved -> {path}  ({len(features)} features, {len(score_ref)} ref scores)")


if __name__ == "__main__":
    main()
