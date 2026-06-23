"""Streamlit UI: score a single user, and sweep the budget to see the Qini frontier.

Dataset-agnostic: feature inputs and the Qini cohort come from the trained model bundle
and the processed parquet, so it works for the simulated RCT or Hillstrom alike.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.evaluate import policy_value, qini_curve
from src.features import split_xyt

st.set_page_config(page_title="Uplift Targeting Engine", layout="wide")
st.title("🎯 Uplift Targeting Engine")
st.caption("Who is *persuadable* — not who converts anyway.")

MODEL_PATH = Path("artifacts/xlearner.pkl")
DATA_PATH = Path("data/processed/experiment.parquet")


@st.cache_resource
def load_bundle():
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


bundle = load_bundle()
if bundle is None:
    st.warning("No model found. Train one first:  `python -m src.train`")
    st.stop()

model, features = bundle["model"], bundle["features"]
st.caption(f"Model: **{bundle.get('learner','?')}** · {len(features)} features")

tab_score, tab_budget = st.tabs(["Score a user", "Budget / Qini frontier"])

with tab_score:
    st.subheader("Single-user uplift")
    st.caption("One-hot features are 0/1; numerics use their natural scale.")
    cols = st.columns(4)
    payload = {f: cols[i % 4].number_input(f, value=0.0, step=1.0, format="%.2f")
               for i, f in enumerate(features)}
    X = pd.DataFrame([payload], columns=features)
    uplift = float(model.predict_uplift(X.values)[0])
    st.metric("estimated uplift", f"{uplift:+.4f}")
    st.write("**Decision:**", "✅ treat" if uplift >= 0 else "⛔ skip (sure-thing or sleeping-dog)")

with tab_budget:
    st.subheader("Qini curve on the held-out cohort")
    if not DATA_PATH.exists():
        st.warning(f"No data at {DATA_PATH}. Build it:  `python -m src.data`")
        st.stop()

    df = pd.read_parquet(DATA_PATH)
    X, t, y = split_xyt(df, features)
    score = model.predict_uplift(X.values)
    rate = st.slider("budget (treat top %)", 0.05, 1.0, 0.30, 0.05)

    frac, gain = qini_curve(score, t, y)
    fig, ax = plt.subplots()
    ax.plot(frac, gain, label="model")
    ax.plot(frac, gain[-1] * frac, "--", label="random")
    ax.axvline(rate, color="grey", ls=":", label=f"budget={rate:.0%}")
    ax.set_xlabel("fraction targeted")
    ax.set_ylabel("cumulative incremental responders")
    ax.legend()
    st.pyplot(fig)

    pv = policy_value(score, t, y, treat_rate=rate)
    c1, c2, c3 = st.columns(3)
    c1.metric("model policy", f"{pv['model_policy']:.4f}")
    c2.metric("random", f"{pv['random']:.4f}")
    c3.metric("treat-all", f"{pv['treat_all']:.4f}")
