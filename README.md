# PASG: Persistence-Anchored, Skill-Gated next-day PM2.5 forecasting

Code for the paper *"Know When Not to Forecast: Persistence-Anchored, Skill-Gated
Boosting for Next-Day PM2.5 in Indian Cities"*.

**Question.** Do machine-learning models for next-day air quality actually beat the
naive "tomorrow = today" forecast? And if not, how should they be built so that they do?

**Method (PASG).**
1. A multi-quantile gradient-boosted model predicts the *change* over persistence,
   `dz = log(PM2.5_{t+1}+1) - log(PM2.5_t+1)`, at quantiles 0.1 / 0.5 / 0.9.
2. Signal-to-noise ratio of the predicted change: `s = |q50| / (q90 - q10)`.
3. A two-parameter skill gate `alpha = sigmoid(a + b log s)`, fitted on cross-fitted
   out-of-fold predictions, decides how far to move away from persistence:
   `z_hat = z_t + alpha * q50`.
4. Normalised split-conformal intervals `z_hat +/- c (q90 - q10)` give calibrated
   90% prediction intervals.

## Data (free, no API key)
* PM2.5: CAMS global atmospheric composition (Open-Meteo Air Quality API), Aug 2022 - Oct 2026.
* Meteorology: ERA5 reanalysis (Open-Meteo archive API).
* Optional: CPCB station observations in the Kaggle `city_day.csv` format (2015-2020):
  `python run_experiments.py --source cpcb --cpcb-csv data/city_day.csv`.

10 cities: Delhi, Lucknow, Patna, Kolkata, Ahmedabad, Mumbai, Pune, Hyderabad, Bengaluru, Chennai.

## Reproduce
```
pip install -r requirements.txt
python run_experiments.py        # downloads data, rolling-origin evaluation (~40 min)
python analyze.py                # tables + figures -> results/openmeteo/
python extra_analysis.py         # supplementary tables (~5 min)
```

Evaluation: expanding-window rolling origin, test years 2024 / 2025 / 2026 (to 28 Oct).
Two settings: **nowcast** (information up to day t) and **perfect-prog** (adds
day-(t+1) ERA5 meteorology as an upper bound on what an NWP forecast can contribute).

## Layout
```
pasg/data.py        data download + caching
pasg/features.py    lag / rolling / meteorological / calendar features
pasg/models.py      baselines + PAQ (and gated variants)
pasg/metrics.py     MAE, RMSE, skill score, NAQI category accuracy, Diebold-Mariano
run_experiments.py  rolling-origin experiment
analyze.py          tables (CSV + LaTeX) and figures
extra_analysis.py   data description, season/region/year skill, feature importance
paper/              manuscript draft (LaTeX)
```
