"""Benchmark every learner on a dataset: Qini, policy value, and a Qini-curve chart.

Reproducible source for the README results. Trains each meta-learner on the same
train/test split and prints a markdown table; optionally saves a Qini chart of all
learners on one axes.

    python -m src.benchmark --data data/processed/experiment.parquet \
        --learners slearner tlearner xlearner rlearner --chart assets/qini.png
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")  # headless: write PNG, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from .evaluate import policy_value, qini_coefficient, qini_curve, uplift_at_k, validate_against_truth  # noqa: E402
from .features import feature_columns, split_xyt  # noqa: E402
from .learners import LEARNERS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="benchmark learners + emit Qini chart")
    ap.add_argument("--data", default="data/processed/experiment.parquet")
    ap.add_argument("--learners", nargs="+", default=["slearner", "tlearner", "xlearner", "rlearner"])
    ap.add_argument("--treat-rate", type=float, default=0.30)
    ap.add_argument("--test-size", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--chart", default=None, help="path to save the Qini chart PNG")
    ap.add_argument("--title", default=None, help="chart title / dataset label")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    feats = feature_columns(df)
    has_truth = "true_uplift" in df
    train_df, test_df = train_test_split(df, test_size=args.test_size,
                                         random_state=args.seed, stratify=df["treatment"])
    Xtr, ttr, ytr = split_xyt(train_df, feats)
    Xte, tte, yte = split_xyt(test_df, feats)

    # observed ATE on the test split (randomized -> treated rate minus control rate)
    ate = yte[tte == 1].mean() - yte[tte == 0].mean()
    print(f"\ndataset: {args.title or args.data}")
    print(f"  rows={len(df):,}  features={len(feats)}  test={len(test_df):,}  "
          f"treat_rate={df['treatment'].mean():.3f}  observed ATE={ate:+.4f}\n")

    header = "| learner | Qini | policy@30% | random | treat-all | uplift@30% |"
    if has_truth:
        header += " spearman↑ | ATE err↓ |"
    print(header)
    print("|" + "---|" * (header.count("|") - 1))

    if args.chart:
        plt.figure(figsize=(6, 5))

    rows = []
    for name in args.learners:
        model = LEARNERS[name]().fit(Xtr.values, ttr, ytr)
        s = model.predict_uplift(Xte.values)
        q = qini_coefficient(s, tte, yte)
        pv = policy_value(s, tte, yte, args.treat_rate)
        u30 = uplift_at_k(s, tte, yte, args.treat_rate)
        line = (f"| **{name}** | {q:.1f} | {pv['model_policy']:.4f} | {pv['random']:.4f} "
                f"| {pv['treat_all']:.4f} | {u30:.0f} |")
        if has_truth:
            v = validate_against_truth(s, test_df["true_uplift"].values)
            line += f" {v['spearman_vs_true']:.3f} | {v['ate_abs_err']:.4f} |"
        print(line)
        rows.append((name, q))

        if args.chart:
            frac, gain = qini_curve(s, tte, yte)
            plt.plot(frac, gain, label=f"{name} (Q={q:.0f})")

    if args.chart:
        # random baseline shared by all (straight line to total incremental gain)
        frac, gain = qini_curve(rows and model.predict_uplift(Xte.values), tte, yte)
        plt.plot(frac, gain[-1] * frac, "k--", alpha=0.5, label="random")
        plt.xlabel("fraction of population targeted")
        plt.ylabel("cumulative incremental responders")
        plt.title(args.title or "Qini curves")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.chart, dpi=120)
        print(f"\nsaved chart -> {args.chart}")


if __name__ == "__main__":
    main()
