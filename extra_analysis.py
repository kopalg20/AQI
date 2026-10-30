"""Supplementary analyses: data description, seasonal / yearly skill, feature-group importance.

    python extra_analysis.py [--source openmeteo]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pasg import data, features
from pasg.models import PASG

ROOT = Path(__file__).resolve().parent
SEASONS = {12: "Winter (DJF)", 1: "Winter (DJF)", 2: "Winter (DJF)",
           3: "Pre-monsoon (MAM)", 4: "Pre-monsoon (MAM)", 5: "Pre-monsoon (MAM)",
           6: "Monsoon (JJAS)", 7: "Monsoon (JJAS)", 8: "Monsoon (JJAS)", 9: "Monsoon (JJAS)",
           10: "Post-monsoon (ON)", 11: "Post-monsoon (ON)"}
IGP = {"Delhi", "Lucknow", "Patna", "Kolkata"}
MODELS = ["Ridge-AR", "XGB-level", "PA-Quantile (no gate)"]


def describe(source):
    rows = []
    for c in data.CITIES:
        s = data.load_city(c, source)["pm25"]
        rows.append(dict(city=c, region="IGP" if c in IGP else "Other",
                         days=int(s.notna().sum()), mean=s.mean(), sd=s.std(),
                         p95=s.quantile(0.95), pct_above_60=(s > 60).mean() * 100,
                         lag1_acf_log=np.log1p(s).autocorr(1)))
    return pd.DataFrame(rows)


def skill_by(p, key):
    out = []
    for (s, k), g in p.groupby(["setting", key]):
        base = g[g.model == "Persistence"]
        mae_p = (base.y - base.pred).abs().mean()
        for m in MODELS:
            gm = g[g.model == m]
            out.append(dict(setting=s, **{key: k}, model=m, n=len(gm),
                            SS=100 * (1 - (gm.y - gm.pred).abs().mean() / mae_p),
                            persist_MAE=mae_p))
    return pd.DataFrame(out)


def feature_groups(cols):
    def grp(c):
        if c in ("z_lag0", "dz_1", "dz_2"):
            return "Current level & recent change"
        if c.startswith("z_"):
            return "Lagged / rolling pollution"
        if c.startswith(("f1_", "f1d_")):
            return "Next-day meteorology"
        if c.startswith("met_"):
            return "Day-t meteorology"
        return "Calendar"
    return {c: grp(c) for c in cols}


def importance(source):
    """Total-gain importance of the PAQ model (median output), last fold, grouped."""
    rows = []
    for setting in ("nowcast", "perfect_prog"):
        for c in data.CITIES:
            fr = features.build(data.load_city(c, source), setting)
            cols = features.feature_columns(fr)
            tr = fr[pd.to_datetime(fr["target_date"]) < "2026-01-01"]
            m = PASG(gate="none").fit(tr, cols)
            gain = m.m.get_booster().get_score(importance_type="total_gain")
            g = pd.Series(gain).reindex(cols).fillna(0.0)
            g = g / g.sum()
            grp = feature_groups(cols)
            for name, v in g.groupby(g.index.map(grp)).sum().items():
                rows.append(dict(setting=setting, city=c, group=name, share=v))
            top = g.sort_values(ascending=False).head(5)
            for rank, (f, v) in enumerate(top.items(), 1):
                rows.append(dict(setting=setting, city=c, group=f"TOP{rank}:{f}", share=v))
            print(f"  importance {setting} {c}", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="openmeteo")
    args = ap.parse_args()
    res = ROOT / "results" / args.source
    pd.set_option("display.width", 200)

    d = describe(args.source)
    d.to_csv(res / "table_data.csv", index=False)
    print(d.round(2).to_string(index=False))

    p = pd.concat([pd.read_csv(f) for f in sorted(res.glob("predictions_seed0*.csv.gz"))])
    p = p.drop_duplicates(["city", "setting", "fold", "target_date", "model"], keep="last")
    p["season"] = pd.to_datetime(p.target_date).dt.month.map(SEASONS)
    p["region"] = np.where(p.city.isin(IGP), "IGP", "Other")
    for key in ("season", "fold", "region"):
        t = skill_by(p, key)
        t.to_csv(res / f"table_skill_by_{key}.csv", index=False)
        print(t.pivot_table(index=["setting", key], columns="model", values="SS").round(1))

    imp = importance(args.source)
    imp.to_csv(res / "table_importance.csv", index=False)
    grp = imp[~imp.group.str.startswith("TOP")]
    print((grp.pivot_table(index="group", columns="setting", values="share", aggfunc="mean") * 100).round(1))
    top = imp[imp.group.str.startswith("TOP")].copy()
    top["feature"] = top.group.str.split(":").str[1]
    print(top.groupby(["setting", "feature"]).size().sort_values(ascending=False).groupby("setting").head(6))


if __name__ == "__main__":
    main()
