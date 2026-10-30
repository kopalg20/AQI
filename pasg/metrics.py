"""Evaluation metrics, including the Diebold-Mariano test."""
from __future__ import annotations

import numpy as np
from scipy import stats

# Indian National AQI breakpoints for 24-h PM2.5 (ug/m3)
NAQI_PM25 = [30, 60, 90, 120, 250]
NAQI_LABELS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
STANDARD_24H = 60.0


def category(y):
    return np.digitize(y, NAQI_PM25, right=True)


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def f1_exceed(y, p, thr=STANDARD_24H):
    t, h = y > thr, p > thr
    tp = np.sum(t & h)
    denom = 2 * tp + np.sum(~t & h) + np.sum(t & ~h)
    return float(2 * tp / denom) if denom else np.nan


def point_metrics(y, p, persist):
    return dict(
        MAE=mae(y, p),
        RMSE=rmse(y, p),
        SS_MAE=1 - mae(y, p) / mae(y, persist),
        CatAcc=float(np.mean(category(y) == category(p))),
        F1_exceed=f1_exceed(y, p),
        n=len(y),
    )


def diebold_mariano(e1, e2, power=1, max_lag=None):
    """DM test (Harvey et al. 1997 small-sample corrected) on |e|^power.

    Negative statistic => model 1 has lower loss. Two-sided p-value.
    """
    return dm_from_differential(np.abs(e1) ** power - np.abs(e2) ** power, max_lag)


def dm_from_differential(d, max_lag=None):
    """DM test on a loss-differential series, Newey-West long-run variance."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    if max_lag is None:
        max_lag = int(np.floor(n ** (1 / 3)))
    dc = d - d.mean()
    lrv = np.dot(dc, dc) / n
    for k in range(1, max_lag + 1):
        w = 1 - k / (max_lag + 1)
        lrv += 2 * w * np.dot(dc[k:], dc[:-k]) / n
    dm = d.mean() / np.sqrt(lrv / n)
    h = 1
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.sf(np.abs(dm), df=n - 1)
    return float(dm), float(p)
