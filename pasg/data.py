"""Data acquisition: daily PM2.5 + daily meteorology for a set of cities.

Two sources are supported:
  * "openmeteo": CAMS air-quality (Open-Meteo Air Quality API, Aug-2022 onwards)
  * "cpcb":      CPCB station data in the Kaggle `city_day.csv` format (2015-2020)
Meteorology always comes from the ERA5 reanalysis via the Open-Meteo archive API.
All requests are free and need no API key. Responses are cached under data/raw.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

CITIES = {
    "Delhi": (28.61, 77.21),
    "Lucknow": (26.85, 80.95),
    "Patna": (25.59, 85.14),
    "Kolkata": (22.57, 88.36),
    "Ahmedabad": (23.02, 72.57),
    "Mumbai": (19.08, 72.88),
    "Pune": (18.52, 73.86),
    "Hyderabad": (17.39, 78.49),
    "Bengaluru": (12.97, 77.59),
    "Chennai": (13.08, 80.27),
}

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
MET_URL = "https://archive-api.open-meteo.com/v1/archive"
MET_VARS = [
    "temperature_2m", "relative_humidity_2m", "precipitation", "surface_pressure",
    "wind_speed_10m", "wind_direction_10m", "boundary_layer_height",
]
TZ = "Asia/Kolkata"


def _get_json(url: str, params: dict, cache: Path) -> dict:
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    for attempt in range(5):
        r = requests.get(url, params=params, timeout=120)
        if r.status_code == 200:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(r.text, encoding="utf-8")
            return r.json()
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{url} failed: {r.status_code} {r.text[:200]}")


def _hourly_frame(js: dict) -> pd.DataFrame:
    df = pd.DataFrame(js["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")


def fetch_met(city: str, start: str, end: str) -> pd.DataFrame:
    """Daily meteorology (local-day aggregates of hourly ERA5)."""
    lat, lon = CITIES[city]
    js = _get_json(MET_URL, dict(latitude=lat, longitude=lon, start_date=start, end_date=end,
                                 hourly=",".join(MET_VARS), timezone=TZ),
                   RAW / f"met_{city}_{start}_{end}.json")
    h = _hourly_frame(js)
    rad = np.deg2rad(h.pop("wind_direction_10m"))
    h["wind_u"] = -h["wind_speed_10m"] * np.sin(rad)
    h["wind_v"] = -h["wind_speed_10m"] * np.cos(rad)
    g = h.resample("D")
    d = g.mean().add_prefix("met_")
    d["met_precipitation"] = g["precipitation"].sum()
    d["met_blh_min"] = g["boundary_layer_height"].min()
    d["met_blh_max"] = g["boundary_layer_height"].max()
    d["met_temp_range"] = g["temperature_2m"].max() - g["temperature_2m"].min()
    d.index.name = "date"
    return d


def fetch_openmeteo_pm25(city: str, start: str, end: str) -> pd.Series:
    lat, lon = CITIES[city]
    js = _get_json(AQ_URL, dict(latitude=lat, longitude=lon, start_date=start, end_date=end,
                                hourly="pm2_5", timezone=TZ),
                   RAW / f"aq_{city}_{start}_{end}.json")
    h = _hourly_frame(js)["pm2_5"]
    # require >= 18 valid hours for a daily mean
    d = h.resample("D").mean().where(h.resample("D").count() >= 18)
    d.index.name = "date"
    return d.rename("pm25")


def load_cpcb_pm25(csv: str | Path, city: str) -> pd.Series:
    df = pd.read_csv(csv, parse_dates=["Date"])
    s = df.loc[df["City"] == city].set_index("Date")["PM2.5"].sort_index()
    s.index.name = "date"
    return s.asfreq("D").rename("pm25")


def load_city(city: str, source: str = "openmeteo", start: str = "2022-08-01",
              end: str = "2026-10-28", cpcb_csv: str | None = None) -> pd.DataFrame:
    if source == "openmeteo":
        pm = fetch_openmeteo_pm25(city, start, end)
    elif source == "cpcb":
        pm = load_cpcb_pm25(cpcb_csv, city)
        start, end = str(pm.index.min().date()), str(pm.index.max().date())
    else:
        raise ValueError(source)
    met = fetch_met(city, start, end)
    df = met.join(pm, how="left")
    df = df.loc[pm.first_valid_index():pm.last_valid_index()]
    df["city"] = city
    return df
