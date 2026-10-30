"""Tables (CSV + LaTeX) and figures from saved predictions.

    python analyze.py [--source openmeteo]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit

from pasg import metrics as M

ROOT = Path(__file__).resolve().parent
RENAME = {"PA-Quantile (no gate)": "PAQ (ours)", "PA-Quantile + const. shrink": "PAQ + const. gate",
          "PASG": "PAQ + SNR gate"}
REF = "PAQ (ours)"
ORDER = ["Persistence", "Climatology", "Ridge-AR", "RF", "XGB-level", "XGB-level (median)",
         "XGB-delta", "XGB-delta (median)", "PAQ (ours)", "PAQ + const. gate", "PAQ + SNR gate"]
SET_NAME = {"nowcast": "Nowcast", "perfect_prog": "Perfect-prog"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C_PASG, C_XGB, C_RIDGE = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "legend.frameon": False, "figure.dpi": 200,
})


def summarise(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (s, m), g in p.groupby(["setting", "model"]):
        r = M.point_metrics(g.y.to_numpy(), g.pred.to_numpy(), g.persist.to_numpy())
        # mean of per-city skill (each city weighted equally)
        r["SS_city_mean"] = g.groupby("city").apply(
            lambda c: 1 - M.mae(c.y, c.pred) / M.mae(c.y, c.persist), include_groups=False).mean()
        r["cities_beating_persist"] = int((g.groupby("city").apply(
            lambda c: M.mae(c.y, c.pred) < M.mae(c.y, c.persist), include_groups=False)).sum())
        rows.append(dict(setting=s, model=m, **r))
    t = pd.DataFrame(rows)
    t["model"] = pd.Categorical(t["model"], ORDER, ordered=True)
    return t.sort_values(["setting", "model"])


def dm_table(p: pd.DataFrame, ref=REF) -> pd.DataFrame:
    rows = []
    for s, gs in p.groupby("setting"):
        wide = gs.pivot_table(index=["city", "fold", "target_date"], columns="model",
                              values="pred").join(
            gs.groupby(["city", "fold", "target_date"])["y"].first())
        for m in ORDER:
            if m == ref:
                continue
            per_city_sig = 0
            for c, gc in wide.groupby(level="city"):
                st, pv = M.diebold_mariano(gc.y - gc[ref], gc.y - gc[m])
                per_city_sig += int(st < 0 and pv < 0.05)
            # pooled: average the loss differential across cities per date
            d = (np.abs(wide.y - wide[ref]) - np.abs(wide.y - wide[m])).groupby(
                level="target_date").mean()
            st, pv = M.dm_from_differential(d.to_numpy())
            rows.append(dict(setting=s, vs=m, DM_pooled=st, p_pooled=pv,
                             cities_ours_sig_better=per_city_sig))
    return pd.DataFrame(rows)


def city_skill(p):
    t = p.groupby(["setting", "model", "city"]).apply(
        lambda g: 1 - M.mae(g.y, g.pred) / M.mae(g.y, g.persist), include_groups=False)
    return t.rename("SS_MAE").reset_index()


def coverage(d):
    rows = []
    for s, g in d.groupby("setting"):
        for kind, lo, hi in (("normalised (adaptive)", "lo", "hi"),
                             ("absolute (constant)", "lo_abs", "hi_abs")):
            cov = ((g.y >= g[lo]) & (g.y <= g[hi])).mean()
            rows.append(dict(setting=s, interval=kind, coverage=cov,
                             median_width=(g[hi] - g[lo]).median(),
                             mean_width=(g[hi] - g[lo]).mean(),
                             worst_city_cov=g.groupby("city").apply(
                                 lambda c: ((c.y >= c[lo]) & (c.y <= c[hi])).mean(),
                                 include_groups=False).min()))
    return pd.DataFrame(rows)


def fig_city_skill(cs, out):
    cities = (cs[(cs.model == REF) & (cs.setting == "nowcast")]
              .sort_values("SS_MAE").city.tolist())
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2), sharey=True)
    for ax, s in zip(axes, ["nowcast", "perfect_prog"]):
        ax.axvline(0, color=INK2, lw=0.8)
        ax.grid(axis="x", color=GRID, lw=0.6)
        for i, (m, c, mk) in enumerate([("XGB-level", C_XGB, "s"), ("Ridge-AR", C_RIDGE, "^"),
                                        (REF, C_PASG, "o")]):
            v = cs[(cs.model == m) & (cs.setting == s)].set_index("city").loc[cities, "SS_MAE"]
            ax.scatter(v.values * 100, np.arange(len(cities)) + (i - 1) * 0.18, s=22, color=c,
                       marker=mk, label=m, zorder=3, edgecolor=SURF, linewidth=0.8)
        ax.set_yticks(range(len(cities)), cities)
        ax.set_title(SET_NAME[s], color=INK, fontsize=9, loc="left")
        ax.set_xlabel("MAE skill vs. persistence (%)")
    axes[0].legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_city_skill.pdf")
    fig.savefig(out / "fig_city_skill.png")
    plt.close(fig)


def fig_gate(d, out):
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    x = np.linspace(-5, 2, 200)
    params = d[d.setting == "nowcast"].groupby(["city", "fold"])[["gate_a", "gate_b"]].first()
    for (_, _), r in params.iterrows():
        ax.plot(x, expit(r.gate_a + r.gate_b * x), color=C_PASG, alpha=0.18, lw=1)
    med = params.median()
    ax.plot(x, expit(med.gate_a + med.gate_b * x), color=C_PASG, lw=2, label="median fit")
    snr = np.log(np.abs(d.q50) + 1e-6) - np.log(d.q90 - d.q10 + 1e-6)
    ax2 = ax.twinx()
    ax2.hist(snr[d.setting == "nowcast"], bins=60, range=(-5, 2), color=GRID, zorder=0)
    ax2.set_yticks([])
    ax2.spines[["top", "right"]].set_visible(False)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_xlabel(r"log SNR  $\log(|\hat q_{.5}| / (\hat q_{.9}-\hat q_{.1}))$")
    ax.set_ylabel(r"gate $\alpha$")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_gate.pdf")
    fig.savefig(out / "fig_gate.png")
    plt.close(fig)


def fig_timeseries(p, d, out, city="Delhi", start="2025-10-01", end="2026-01-31"):
    g = d[(d.city == city) & (d.setting == "nowcast")].copy()
    g["target_date"] = pd.to_datetime(g.target_date)
    g = g[(g.target_date >= start) & (g.target_date <= end)]
    if g.empty:
        return
    x = p[(p.city == city) & (p.setting == "nowcast") & (p.model == "XGB-level")].copy()
    x["target_date"] = pd.to_datetime(x.target_date)
    x = x[(x.target_date >= start) & (x.target_date <= end)]
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.fill_between(g.target_date, g.lo, g.hi, color="#cde2fb", lw=0, label="PAQ 90% interval")
    ax.plot(g.target_date, g.y, color=INK, lw=1.4, label="observed")
    ax.plot(x.target_date, x.pred, color=C_XGB, lw=1.2, label="XGB-level")
    ax.plot(g.target_date, np.expm1(g.z), color=C_PASG, lw=1.6, label="PAQ (ours)")
    ax.set_ylabel(r"PM$_{2.5}$ ($\mu$g m$^{-3}$)")
    ax.set_title(f"{city}, next-day forecasts (nowcast setting)", loc="left", fontsize=9)
    ax.legend(ncol=4, fontsize=7, loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "fig_timeseries.pdf")
    fig.savefig(out / "fig_timeseries.png")
    plt.close(fig)


def gate_stability(d):
    """Learned SNR-gate parameters per city and fold: is the gate a stable, real effect?"""
    g = d.groupby(["setting", "city", "fold"])[["gate_a", "gate_b"]].first().reset_index()
    return g.groupby("setting").agg(
        b_median=("gate_b", "median"), b_min=("gate_b", "min"), b_max=("gate_b", "max"),
        frac_b_positive=("gate_b", lambda v: (v > 0).mean()),
        alpha_test_mean=("gate_b", "size")).drop(columns="alpha_test_mean")


def to_latex(t, path, cols, fmt):
    t = t[cols].copy()
    for c, f in fmt.items():
        t[c] = t[c].map(lambda v: f.format(v))
    path.write_text(t.to_latex(index=False, escape=True), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="openmeteo")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    res = ROOT / "results" / args.source
    p = pd.concat([pd.read_csv(f) for f in sorted(res.glob(f"predictions_seed{args.seed}*.csv.gz"))])
    p = p.drop_duplicates(["city", "setting", "fold", "target_date", "model"], keep="last")
    p["model"] = p["model"].replace(RENAME)
    d_gate = pd.read_csv(res / f"pasg_details_seed{args.seed}.csv.gz")       # SNR-gate variant
    d = pd.read_csv(res / f"pasg_details_seed{args.seed}_paq.csv.gz")         # PAQ (no gate)

    s = summarise(p)
    s.to_csv(res / "table_overall.csv", index=False)
    cs = city_skill(p)
    cs.pivot_table(index=["setting", "city"], columns="model", values="SS_MAE")[ORDER].to_csv(
        res / "table_city_skill.csv")
    dm = dm_table(p)
    dm.to_csv(res / "table_dm.csv", index=False)
    cov = coverage(d)
    cov.to_csv(res / "table_coverage.csv", index=False)

    for st in ("nowcast", "perfect_prog"):
        to_latex(s[s.setting == st], res / f"table_overall_{st}.tex",
                 ["model", "MAE", "RMSE", "SS_MAE", "SS_city_mean", "CatAcc", "F1_exceed",
                  "cities_beating_persist"],
                 {"MAE": "{:.2f}", "RMSE": "{:.2f}", "SS_MAE": "{:.3f}", "SS_city_mean": "{:.3f}",
                  "CatAcc": "{:.3f}", "F1_exceed": "{:.3f}"})

    fig_city_skill(cs, res)
    fig_gate(d_gate, res)
    gs = gate_stability(d_gate)
    gs.to_csv(res / "table_gate_stability.csv")
    fig_timeseries(p, d, res)

    pd.set_option("display.width", 200)
    print(s.round(3).to_string(index=False))
    print(dm.round(4).to_string(index=False))
    print(cov.round(3).to_string(index=False))
    print(gs.round(3).to_string())


if __name__ == "__main__":
    main()
