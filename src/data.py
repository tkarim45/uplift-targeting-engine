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

# Criteo Uplift Prediction v2.1 — 25.3M randomized ad impressions, 12 anonymized
# continuous features (f0..f11) + treatment/exposure/visit/conversion. The scale story.
CRITEO_URL = (
    "https://huggingface.co/datasets/criteo/criteo-uplift/"
    "resolve/main/criteo-research-uplift-v2.1.csv.gz"
)
CRITEO_FEATURES = [f"f{i}" for i in range(12)]


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


def load_criteo(
    sample_frac: float = 0.05,
    max_rows: int | None = None,
    outcome: str = "visit",
    seed: int = 7,
    chunksize: int = 500_000,
    cache: str = "data/raw/criteo.csv.gz",
    url: str = CRITEO_URL,
) -> pd.DataFrame:
    """Load + normalize the Criteo Uplift v2.1 dataset to [f0..f11, treatment, outcome].

    25.3M randomized ad impressions (311MB gzip). Read in chunks and sub-sampled per
    chunk so the full file never lands in memory — that's the point of this adapter:
    a real large-scale uplift workload that still runs on a laptop.

    Args:
        sample_frac: fraction kept from each chunk (0.05 -> ~1.25M rows). Use 1.0 for full.
        max_rows: hard cap on rows after sampling (stop early). None = no cap.
        outcome: 'visit' (~4.7%) or 'conversion' (~0.29%, extremely sparse).
        chunksize: rows per read chunk (memory knob).

    Note: treatment is randomized (valid uplift), but 'exposure' marks impressions
    actually served. We model intent-to-treat on 'treatment' (the standard choice);
    swap to 'exposure' for treatment-on-the-treated analysis.
    """
    if outcome not in ("visit", "conversion"):
        raise ValueError("outcome must be 'visit' or 'conversion'")
    if not 0.0 < sample_frac <= 1.0:
        raise ValueError("sample_frac must be in (0, 1]")

    path = _ensure_local(cache, url, label="Criteo (311MB)")
    required = set(CRITEO_FEATURES + ["treatment", outcome])

    parts, total = [], 0
    reader = pd.read_csv(path, compression="gzip", chunksize=chunksize)
    for i, chunk in enumerate(reader):
        chunk.columns = [c.strip().lower() for c in chunk.columns]
        missing = required - set(chunk.columns)
        if missing:
            raise ValueError(f"Criteo CSV missing columns: {sorted(missing)}; got {list(chunk.columns)}")
        if sample_frac < 1.0:
            chunk = chunk.sample(frac=sample_frac, random_state=seed + i)
        parts.append(chunk)
        total += len(chunk)
        if max_rows and total >= max_rows:
            break
    df = pd.concat(parts, ignore_index=True)
    if max_rows:
        df = df.iloc[:max_rows]

    out = df[CRITEO_FEATURES].astype(float)
    out["treatment"] = df["treatment"].astype(int).values
    out["outcome"] = df[outcome].astype(int).values
    return out


def _read_hillstrom_csv(cache: str, url: str) -> pd.DataFrame:
    """Read the cached CSV, downloading once if missing."""
    path = _ensure_local(cache, url, label="Hillstrom")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    required = set(HILLSTROM_NUMERIC + HILLSTROM_CATEGORICAL + ["segment", "visit", "conversion"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Hillstrom CSV missing columns: {sorted(missing)}; got {list(df.columns)}")
    return df


def _ensure_local(cache: str, url: str, label: str = "dataset") -> Path:
    """Return a local path, streaming the download once (with progress) if missing."""
    path = Path(cache)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    print(f"downloading {label} -> {path}")

    def _progress(blocks, block_size, total):
        if total > 0:
            pct = min(100, 100 * blocks * block_size / total)
            print(f"\r  {pct:5.1f}%  ({blocks * block_size // (1 << 20)} MB)", end="", flush=True)

    urllib.request.urlretrieve(url, tmp, reporthook=_progress)  # noqa: S310 (trusted, pinned URL)
    print()
    tmp.replace(path)  # atomic: a partial download never poses as complete
    return path


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def main() -> None:
    ap = argparse.ArgumentParser(description="build experiment dataset")
    ap.add_argument("--dataset", choices=["simulate", "hillstrom", "criteo"], default="simulate")
    ap.add_argument("--simulate", action="store_true", help="alias for --dataset simulate")
    ap.add_argument("--n", type=int, default=50_000, help="rows (simulate only)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--campaign", choices=["any", "womens", "mens"], default="any",
                    help="Hillstrom treated arm vs No-E-Mail control")
    ap.add_argument("--outcome", choices=["visit", "conversion"], default="visit",
                    help="Hillstrom/Criteo outcome column")
    ap.add_argument("--sample-frac", type=float, default=0.05,
                    help="Criteo: fraction kept per chunk (0.05 -> ~1.25M rows)")
    ap.add_argument("--max-rows", type=int, default=None, help="Criteo: hard row cap")
    ap.add_argument("--out", default="data/processed/experiment.parquet")
    args = ap.parse_args()

    dataset = "simulate" if args.simulate else args.dataset
    if dataset == "hillstrom":
        df = load_hillstrom(campaign=args.campaign, outcome=args.outcome)
    elif dataset == "criteo":
        df = load_criteo(sample_frac=args.sample_frac, max_rows=args.max_rows, outcome=args.outcome)
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
