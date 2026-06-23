"""Meta-learners for CATE (uplift) estimation, over XGBoost base models.

Implemented from scratch (no causalml dependency required for the core S/T/X learners)
so the mechanics are explicit and reviewable — this is the part interviewers probe.

    S-learner: one model on [X, treatment]; uplift = f(X,1) - f(X,0)
    T-learner: two models (treated / control); uplift = f_t(X) - f_c(X)
    X-learner: T-learner + impute individual effects + propensity-weighted blend

R-learner is left as a stretch (residualize outcome & treatment, then weighted regress);
wire econml's RLearner for a cross-check.
"""
from __future__ import annotations

import numpy as np
from xgboost import XGBClassifier, XGBRegressor

_XGB = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, n_jobs=-1, eval_metric="logloss")


class SLearner:
    """Single model with treatment as a feature."""

    def __init__(self, **kw):
        self.model = XGBClassifier(**{**_XGB, **kw})

    def fit(self, X, t, y):
        Xt = _stack(X, t)
        self.model.fit(Xt, y)
        return self

    def predict_uplift(self, X):
        n = len(X)
        p1 = self.model.predict_proba(_stack(X, np.ones(n)))[:, 1]
        p0 = self.model.predict_proba(_stack(X, np.zeros(n)))[:, 1]
        return p1 - p0


class TLearner:
    """Separate treated / control response models."""

    def __init__(self, **kw):
        self.m_t = XGBClassifier(**{**_XGB, **kw})
        self.m_c = XGBClassifier(**{**_XGB, **kw})

    def fit(self, X, t, y):
        t = np.asarray(t)
        self.m_t.fit(X[t == 1], y[t == 1])
        self.m_c.fit(X[t == 0], y[t == 0])
        return self

    def predict_uplift(self, X):
        return self.m_t.predict_proba(X)[:, 1] - self.m_c.predict_proba(X)[:, 1]


class XLearner:
    """X-learner: impute individual effects, regress them, blend by propensity."""

    def __init__(self, **kw):
        self.m_t = XGBClassifier(**{**_XGB, **kw})
        self.m_c = XGBClassifier(**{**_XGB, **kw})
        self.tau_t = XGBRegressor(**{**_XGB, "eval_metric": "rmse"})
        self.tau_c = XGBRegressor(**{**_XGB, "eval_metric": "rmse"})
        self.prop = XGBClassifier(**{**_XGB, **kw})

    def fit(self, X, t, y):
        t = np.asarray(t)
        Xt, Xc = X[t == 1], X[t == 0]
        yt, yc = y[t == 1], y[t == 0]
        self.m_t.fit(Xt, yt)
        self.m_c.fit(Xc, yc)
        # imputed individual treatment effects
        d_t = yt - self.m_c.predict_proba(Xt)[:, 1]
        d_c = self.m_t.predict_proba(Xc)[:, 1] - yc
        self.tau_t.fit(Xt, d_t)
        self.tau_c.fit(Xc, d_c)
        self.prop.fit(X, t)  # propensity for blending
        return self

    def predict_uplift(self, X):
        g = self.prop.predict_proba(X)[:, 1]
        return g * self.tau_c.predict(X) + (1 - g) * self.tau_t.predict(X)


LEARNERS = {"slearner": SLearner, "tlearner": TLearner, "xlearner": XLearner}


def _stack(X, t):
    t = np.asarray(t).reshape(-1, 1)
    return np.hstack([np.asarray(X), t])
