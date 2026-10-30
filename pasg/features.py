"""Feature construction for next-day (t -> t+1) PM2.5 forecasting.

Everything is in log space, z = log(PM2.5 + 1). A sample is issued at the end of
day t and targets day t+1.

Settings
  nowcast      : only information available up to day t
  perfect_prog : additionally uses day-(t+1) meteorology (ERA5 used as a stand-in
                 for a skilful NWP forecast; an upper bound on what forecasts add)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MET_DYNAMIC = ["met_boundary_layer_height", "met_wind_speed_10m", "met_temperature_2m",
               "met_relative_humidity_2m", "met_precipitation", "met_surface_pressure"]


def build(df: pd.DataFrame, setting: str = "nowcast") -> pd.DataFrame:
    df = df.sort_index().copy()
    z_raw = np.log1p(df["pm25"])
    z = z_raw.interpolate(limit=2, limit_area="inside")  # lag features only
    met_cols = [c for c in df.columns if c.startswith("met_")]

    X = pd.DataFrame(index=df.index)
    for k in range(7):
        X[f"z_lag{k}"] = z.shift(k)
    X["dz_1"] = z - z.shift(1)
    X["dz_2"] = z - z.shift(2)
    for w in (3, 7, 14, 30):
        X[f"z_mean{w}"] = z.rolling(w, min_periods=max(2, w // 2)).mean()
    X["z_std7"] = z.rolling(7, min_periods=4).std()
    X["z_anom7"] = z - X["z_mean7"]
    X["z_anom30"] = z - X["z_mean30"]
    X["z_max7"] = z.rolling(7, min_periods=4).max()
    X["z_min7"] = z.rolling(7, min_periods=4).min()

    for c in met_cols:
        X[c] = df[c]
    for c in MET_DYNAMIC:
        X[c + "_d1"] = df[c] - df[c].shift(1)

    tgt_day = df.index + pd.Timedelta(days=1)
    doy = tgt_day.dayofyear.to_numpy()
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    X["dow_target"] = tgt_day.dayofweek

    if setting == "perfect_prog":
        nxt = df[met_cols].shift(-1)
        for c in met_cols:
            X["f1_" + c] = nxt[c]
        for c in MET_DYNAMIC:
            X["f1d_" + c] = nxt[c] - df[c]
    elif setting != "nowcast":
        raise ValueError(setting)

    out = X.copy()
    out["z_t"] = z_raw                     # persistence anchor (raw, not interpolated)
    out["z_next"] = z_raw.shift(-1)        # target
    out["y_next"] = df["pm25"].shift(-1)
    out["target_date"] = tgt_day
    out["city"] = df["city"]
    out = out.dropna(subset=["z_t", "z_next"])
    return out


def feature_columns(frame: pd.DataFrame) -> list[str]:
    skip = {"z_t", "z_next", "y_next", "target_date", "city"}
    return [c for c in frame.columns if c not in skip]
