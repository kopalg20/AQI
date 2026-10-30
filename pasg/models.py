"""Baselines and the proposed PAQ forecaster.

All models predict z_{t+1} = log(PM2.5_{t+1} + 1) and expose
    fit(train_frame, feature_cols) -> self
    predict(test_frame) -> np.ndarray   (log space)

PAQ -- Persistence-Anchored Quantile boosting  (class PASG, gate="none")
-----------------------------------------------------------------------
    1. A multi-quantile gradient-boosted model learns the *change* over persistence,
       dz = z_{t+1} - z_t, at quantiles (0.1, 0.5, 0.9).
    2. Forecast: z_hat = z_t + alpha(x) * q50(x), alpha = 1 for PAQ.
    3. Normalised split-conformal intervals: z_hat +/- c * (q90 - q10), with c the
       finite-sample (1 - miscoverage) quantile of |z - z_hat| / (q90 - q10) on
       cross-fitted out-of-fold predictions.

Gated variants (ablations; reported as a negative result in the paper):
    gate="constant": alpha = sigmoid(a), a global shrinkage toward persistence.
    gate="snr":      alpha(x) = sigmoid(a + b * log s(x)), s = |q50| / (q90 - q10);
                     move away from persistence only as far as the model's
                     confidence in the predicted change justifies.
    Gate parameters are fitted on the out-of-fold predictions to minimise MAE in
    concentration space.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

XGB_PARAMS = dict(n_estimators=500, learning_rate=0.03, max_depth=4, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
                  tree_method="hist", n_jobs=4)
EPS = 1e-6


def to_conc(z):
    return np.expm1(z)


class Persistence:
    name = "Persistence"

    def fit(self, tr, cols):
        return self

    def predict(self, te):
        return te["z_t"].to_numpy()


class Climatology:
    """Mean training z_{t+1} within +/-15 days of the target day-of-year."""
    name = "Climatology"

    def fit(self, tr, cols):
        doy = pd.to_datetime(tr["target_date"]).dt.dayofyear.to_numpy()
        z = tr["z_next"].to_numpy()
        self.table = np.empty(367)
        for d in range(1, 367):
            dist = np.abs(doy - d)
            dist = np.minimum(dist, 366 - dist)
            m = dist <= 15
            self.table[d] = z[m].mean() if m.any() else z.mean()
        return self

    def predict(self, te):
        return self.table[pd.to_datetime(te["target_date"]).dt.dayofyear.to_numpy()]


class _Sklearn:
    def __init__(self, est, name):
        self.est, self.name = est, name

    def fit(self, tr, cols):
        self.cols = cols
        self.est.fit(tr[cols], tr["z_next"])
        return self

    def predict(self, te):
        return self.est.predict(te[self.cols])


def Ridge():
    return _Sklearn(make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                  RidgeCV(alphas=np.logspace(-2, 3, 12))), "Ridge-AR")


def RandomForest(seed=0):
    return _Sklearn(make_pipeline(SimpleImputer(strategy="median"),
                                  RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                                        max_features=0.5, n_jobs=4,
                                                        random_state=seed)), "RF")


class XGBLevel:
    """Standard approach: boosted trees regress the next-day level directly.

    loss="squared" (mean) or "median" (pinball loss at 0.5).
    """
    anchored = False

    def __init__(self, seed=0, loss="squared"):
        self.seed, self.loss = seed, loss
        kind = "delta" if self.anchored else "level"
        self.name = f"XGB-{kind}" + (" (median)" if loss == "median" else "")

    def _model(self):
        extra = dict(objective="reg:quantileerror", quantile_alpha=0.5) if self.loss == "median" else {}
        return XGBRegressor(**XGB_PARAMS, **extra, random_state=self.seed)

    def _anchor(self, fr):
        return fr["z_t"].to_numpy() if self.anchored else 0.0

    def fit(self, tr, cols):
        self.cols = cols
        self.m = self._model().fit(tr[cols], tr["z_next"] - self._anchor(tr))
        return self

    def predict(self, te):
        return self._anchor(te) + self.m.predict(te[self.cols])


class XGBDelta(XGBLevel):
    """Residual-on-persistence boosting (no gate)."""
    anchored = True


def _blocked_folds(n, k):
    edges = np.linspace(0, n, k + 1).astype(int)
    for i in range(k):
        te = np.arange(edges[i], edges[i + 1])
        yield np.setdiff1d(np.arange(n), te), te


class PASG:
    name = "PASG"
    QUANTILES = (0.1, 0.5, 0.9)

    def __init__(self, seed=0, k_folds=4, miscoverage=0.1, gate="snr"):
        self.seed, self.k, self.miscov, self.gate = seed, k_folds, miscoverage, gate

    def _qmodel(self):
        return XGBRegressor(**XGB_PARAMS, objective="reg:quantileerror",
                            quantile_alpha=np.array(self.QUANTILES), random_state=self.seed)

    @staticmethod
    def _snr(q):
        return np.log(np.abs(q[:, 1]) + EPS) - np.log(q[:, 2] - q[:, 0] + EPS)

    def _alpha(self, q, theta=None):
        a, b = self.theta if theta is None else theta
        if self.gate == "none":
            return np.ones(len(q))
        if self.gate == "constant":
            return np.full(len(q), expit(a))
        return expit(a + b * self._snr(q))

    def fit(self, tr, cols):
        tr = tr.sort_values("target_date")
        self.cols = cols
        X, zt, zn = tr[cols], tr["z_t"].to_numpy(), tr["z_next"].to_numpy()
        dz = zn - zt

        # cross-fitted out-of-fold quantile predictions for gate + conformal calibration
        oof = np.zeros((len(tr), 3))
        for fi, (a, b) in enumerate(_blocked_folds(len(tr), self.k)):
            m = self._qmodel().set_params(random_state=self.seed + fi + 1)
            oof[b] = np.sort(m.fit(X.iloc[a], dz[a]).predict(X.iloc[b]), axis=1)

        y = to_conc(zn)

        def loss(theta):
            return np.mean(np.abs(to_conc(zt + self._alpha(oof, theta) * oof[:, 1]) - y))

        if self.gate == "none":
            self.theta = (0.0, 0.0)
        else:
            best = None
            for x0 in ([0.0, 1.0], [2.0, 0.5], [-1.0, 2.0]):
                r = minimize(loss, x0, method="Nelder-Mead",
                             options=dict(xatol=1e-4, fatol=1e-6, maxiter=2000))
                if best is None or r.fun < best.fun:
                    best = r
            self.theta = tuple(best.x)

        zhat = zt + self._alpha(oof) * oof[:, 1]
        width = oof[:, 2] - oof[:, 0] + EPS
        n = len(tr)
        lvl = min(1.0, np.ceil((n + 1) * (1 - self.miscov)) / n)
        self.c_norm = np.quantile(np.abs(zn - zhat) / width, lvl)   # adaptive width
        self.c_abs = np.quantile(np.abs(zn - zhat), lvl)            # constant width

        self.m = self._qmodel().fit(X, dz)
        return self

    def _q(self, te):
        return np.sort(self.m.predict(te[self.cols]), axis=1)

    def predict(self, te):
        q = self._q(te)
        return te["z_t"].to_numpy() + self._alpha(q) * q[:, 1]

    def predict_details(self, te):
        q = self._q(te)
        alpha = self._alpha(q)
        zhat = te["z_t"].to_numpy() + alpha * q[:, 1]
        w = q[:, 2] - q[:, 0] + EPS
        return dict(z=zhat, alpha=alpha, q10=q[:, 0], q50=q[:, 1], q90=q[:, 2],
                    lo=zhat - self.c_norm * w, hi=zhat + self.c_norm * w,
                    lo_abs=zhat - self.c_abs, hi_abs=zhat + self.c_abs)


def model_zoo(seed=0, only=None):
    zoo = [
        Persistence(),
        Climatology(),
        Ridge(),
        RandomForest(seed),
        XGBLevel(seed),
        XGBDelta(seed),
        XGBLevel(seed, loss="median"),
        XGBDelta(seed, loss="median"),
        _named(PASG(seed, gate="none"), "PA-Quantile (no gate)"),
        _named(PASG(seed, gate="constant"), "PA-Quantile + const. shrink"),
        PASG(seed, gate="snr"),
    ]
    return [m for m in zoo if only is None or m.name in only]


def _named(obj, name):
    obj.name = name
    return obj
