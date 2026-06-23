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
from sklearn.model_selection import KFold
from xgboost import XGBClassifier, XGBRegressor

_XGB = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, n_jobs=-1, eval_metric="logloss")
_XGB_REG = {**_XGB, "eval_metric": "rmse"}


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


class RLearner:
    """R-learner (Nie & Wager, 2021) — residual-on-residual CATE estimation.

    Cross-fit two nuisances, then regress the Robinson-transformed pseudo-outcome:
        m(x) = E[Y|X]   (outcome model)
        e(x) = E[T|X]   (propensity)
        residuals:   y~ = Y - m(x),  t~ = T - e(x)
        fit tau(x) to minimize  sum  t~^2 * ( y~/t~ - tau(x) )^2
                              == regress (y~/t~) on X with sample_weight t~^2

    Cross-fitting the nuisances (out-of-fold predictions) removes the bias that would
    otherwise leak from overfit nuisance models into the effect estimate.
    """

    def __init__(self, n_folds: int = 5, seed: int = 7, pseudo_clip: float = 50.0, **kw):
        self.n_folds = n_folds
        self.seed = seed
        self.pseudo_clip = pseudo_clip
        self.kw = kw
        self.tau = XGBRegressor(**{**_XGB_REG, **kw})

    def fit(self, X, t, y):
        X = np.asarray(X, dtype=float)
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(X)

        m_hat = np.zeros(n)  # E[Y|X]
        e_hat = np.zeros(n)  # E[T|X]
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        for tr, va in kf.split(X):
            m = XGBClassifier(**{**_XGB, **self.kw}).fit(X[tr], y[tr])
            e = XGBClassifier(**{**_XGB, **self.kw}).fit(X[tr], t[tr])
            m_hat[va] = m.predict_proba(X[va])[:, 1]
            e_hat[va] = e.predict_proba(X[va])[:, 1]

        e_hat = np.clip(e_hat, 1e-3, 1 - 1e-3)  # keep t~ from blowing up via overlap
        y_res = y - m_hat
        t_res = t - e_hat
        weight = t_res ** 2
        # pseudo-outcome; clipped because |t~| can be tiny (weight ~0 there anyway)
        pseudo = np.clip(y_res / t_res, -self.pseudo_clip, self.pseudo_clip)
        self.tau.fit(X, pseudo, sample_weight=weight)
        return self

    def predict_uplift(self, X):
        return self.tau.predict(np.asarray(X, dtype=float))


class EconmlRLearner:
    """Cross-check wrapper around econml's NonParamDML (the library's R-learner).

    Same residual-on-residual estimator, independently implemented + maintained by
    Microsoft Research. We compare its uplift scores against our from-scratch RLearner
    to validate the hand-rolled version. econml is an OPTIONAL dependency — import is
    lazy so the rest of the repo runs without it.
    """

    def __init__(self, n_folds: int = 5, **kw):
        self.n_folds = n_folds
        self.kw = kw
        self._est = None

    def fit(self, X, t, y):
        try:
            from econml.dml import NonParamDML
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError(
                "econml not installed — `pip install econml` to run the cross-check"
            ) from e

        # model_y predicts E[Y|X] as a regression (DML treats Y as continuous, even
        # when binary); only the treatment model is a classifier (discrete_treatment).
        self._est = NonParamDML(
            model_y=XGBRegressor(**{**_XGB_REG, **self.kw}),
            model_t=XGBClassifier(**{**_XGB, **self.kw}),
            model_final=XGBRegressor(**{**_XGB_REG, **self.kw}),
            discrete_treatment=True,
            cv=self.n_folds,
            random_state=7,
        )
        self._est.fit(np.asarray(y, float), np.asarray(t, int), X=np.asarray(X, float))
        return self

    def predict_uplift(self, X):
        return self._est.effect(np.asarray(X, dtype=float))


LEARNERS = {
    "slearner": SLearner,
    "tlearner": TLearner,
    "xlearner": XLearner,
    "rlearner": RLearner,
    "econml": EconmlRLearner,
}


def _stack(X, t):
    t = np.asarray(t).reshape(-1, 1)
    return np.hstack([np.asarray(X), t])
