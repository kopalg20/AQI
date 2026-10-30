"""Rolling-origin evaluation of next-day PM2.5 forecasters across Indian cities.

Usage:
    python run_experiments.py                      # CAMS (Open-Meteo), all cities
    python run_experiments.py --source cpcb --cpcb-csv data/city_day.csv
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pasg import data, features
from pasg.models import PASG, model_zoo, to_conc

ROOT = Path(__file__).resolve().parent


def run_city(city, args):
    raw = data.load_city(city, args.source, cpcb_csv=args.cpcb_csv)
    years = sorted(raw.index.year.unique())
    test_years = years[-args.n_folds:]
    rows, details = [], []
    for setting in ("nowcast", "perfect_prog"):
        fr = features.build(raw, setting)
        cols = features.feature_columns(fr)
        td = pd.to_datetime(fr["target_date"])
        for yr in test_years:
            tr = fr[td < f"{yr}-01-01"]
            te = fr[(td >= f"{yr}-01-01") & (td <= f"{yr}-12-31")]
            if len(tr) < 300 or len(te) < 60:
                continue
            y = te["y_next"].to_numpy()
            for m in model_zoo(args.seed, args.models):
                m.fit(tr, cols)
                p = to_conc(m.predict(te))
                rows.append(pd.DataFrame(dict(
                    city=city, setting=setting, fold=yr, target_date=te["target_date"].values,
                    model=m.name, y=y, pred=p, persist=to_conc(te["z_t"].to_numpy()))))
                if isinstance(m, PASG):
                    d = m.predict_details(te)
                    det = pd.DataFrame({k: v for k, v in d.items()})
                    det[["lo", "hi", "lo_abs", "hi_abs"]] = to_conc(
                        det[["lo", "hi", "lo_abs", "hi_abs"]])
                    det = det.assign(model=m.name, city=city, setting=setting, fold=yr,
                                     target_date=te["target_date"].values, y=y,
                                     gate_a=m.theta[0], gate_b=m.theta[1])
                    details.append(det)
    print(f"  done {city}", flush=True)
    return pd.concat(rows), (pd.concat(details) if details else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="openmeteo", choices=["openmeteo", "cpcb"])
    ap.add_argument("--cpcb-csv", default=None)
    ap.add_argument("--cities", nargs="*", default=list(data.CITIES))
    ap.add_argument("--n-folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--models", nargs="*", default=None, help="subset of model names")
    ap.add_argument("--tag", default="", help="suffix for output files")
    args = ap.parse_args()

    out = ROOT / "results" / args.source
    out.mkdir(parents=True, exist_ok=True)

    # download sequentially (polite to the free API), then model in parallel
    for c in args.cities:
        data.load_city(c, args.source, cpcb_csv=args.cpcb_csv)
    t0 = time.time()
    res = Parallel(n_jobs=args.jobs)(delayed(run_city)(c, args) for c in args.cities)
    preds = pd.concat([r[0] for r in res], ignore_index=True)
    preds.to_csv(out / f"predictions_seed{args.seed}{args.tag}.csv.gz", index=False)
    dets = [r[1] for r in res if r[1] is not None]
    if dets:
        pd.concat(dets, ignore_index=True).to_csv(
            out / f"pasg_details_seed{args.seed}{args.tag}.csv.gz", index=False)
    print(f"finished in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
