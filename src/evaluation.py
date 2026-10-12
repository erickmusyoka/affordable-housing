"""
One rolling-origin backtest for every model (CLAUDE.md Section 14). Never a random split.

For each test origin T the model is fitted ONLY on rows whose target year is T or earlier (what a forecaster standing at
T could have observed) and asked to forecast the rows with origin year T. Every model, econometric or machine learning,
goes through run_backtest() and writes the same long-format results, so comparisons are like for like.

A model is any object with  name,  fit(train: DataFrame),  predict(test: DataFrame) -> array.
It may also provide  predict_interval(test) -> (lower, upper)  for the calibration check.

Metrics (summarize): MAE and RMSE, MAE relative to persistence, and the Diebold-Mariano test against persistence on the
per-origin mean absolute-error differential, with the Harvey-Leybourne-Newbold small-sample correction. With about ten
origins the test has little power, so the table also reports the share of origins won. Results are reported with and
without 2021 (2020 has no ACS data): exclude_years drops evaluation rows whose origin or target year is listed.
"""

import numpy as np
import pandas as pd
from scipy import stats


class Forecaster:
    name = "forecaster"

    def fit(self, train):
        return self

    def predict(self, test):
        raise NotImplementedError


def run_backtest(model, supervised, target, horizon, first_origin=2013, min_train_rows=100, train_exclude_years=()):
    """Return a long frame of forecasts: model, target, horizon, origin_year, target_year, state_fips, y_true, y_origin,
    y_pred (and y_lo, y_hi if the model gives intervals). first_origin is the earliest test origin."""
    sup = supervised
    rows = []
    for T in sorted(sup["origin_year"].unique()):
        if T < first_origin:
            continue
        train = sup[sup["target_year"] <= T]
        if train_exclude_years:
            bad = train["origin_year"].isin(train_exclude_years) | train["target_year"].isin(train_exclude_years)
            train = train[~bad]
        test = sup[sup["origin_year"] == T]
        if len(test) == 0 or len(train) < min_train_rows:
            continue
        # leakage guards: nothing in training may be dated after T, and nothing in test may be from before T
        assert train["target_year"].max() <= T and train["origin_year"].max() <= T - horizon
        assert (test["origin_year"] == T).all()
        m = model.fit(train.copy()) or model
        pred = np.asarray(m.predict(test.copy()), dtype=float)
        frame = pd.DataFrame({"model": model.name, "target": target, "horizon": horizon, "origin_year": T,
                              "target_year": test["target_year"].to_numpy(), "state_fips": test["state_fips"].to_numpy(),
                              "y_true": test["y_target"].to_numpy(), "y_origin": test["y_origin"].to_numpy(), "y_pred": pred})
        if hasattr(m, "predict_interval"):
            lo, hi = m.predict_interval(test.copy())
            frame["y_lo"], frame["y_hi"] = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
        rows.append(frame)
    if not rows:
        raise ValueError("no test origins had enough training data")
    return pd.concat(rows, ignore_index=True)


def diebold_mariano(d, h=1):
    """Diebold-Mariano test of equal accuracy on a loss differential series d (model minus benchmark; negative favors
    the model), with the Harvey-Leybourne-Newbold correction and Student t reference. Returns (statistic, two-sided p)."""
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 4 or np.allclose(d, d[0]):
        return np.nan, np.nan
    dbar = d.mean()
    gamma0 = np.mean((d - dbar) ** 2)
    var = gamma0
    for k in range(1, h):
        var += 2 * np.mean((d[k:] - dbar) * (d[:-k] - dbar))
    var = max(var / n, 1e-12)
    dm = dbar / np.sqrt(var)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = hln * dm
    return stat, 2 * (1 - stats.t.cdf(abs(stat), df=n - 1))


def summarize(results, benchmark="persistence", exclude_years=()):
    r = results.copy()
    if exclude_years:
        r = r[~(r["origin_year"].isin(exclude_years) | r["target_year"].isin(exclude_years))]
    r["abs_err"] = (r["y_true"] - r["y_pred"]).abs()
    r["sq_err"] = (r["y_true"] - r["y_pred"]) ** 2
    out = []
    for (tgt, h), g in r.groupby(["target", "horizon"]):
        bench = g[g["model"] == benchmark].set_index(["origin_year", "state_fips"])["abs_err"]
        bmae = bench.mean() if len(bench) else np.nan
        bench_by_origin = bench.groupby(level="origin_year").mean()
        for model, gm in g.groupby("model"):
            e = gm.set_index(["origin_year", "state_fips"])["abs_err"]
            by_origin = e.groupby(level="origin_year").mean()
            common = by_origin.index.intersection(bench_by_origin.index)
            diff = (by_origin.loc[common] - bench_by_origin.loc[common]).to_numpy()
            stat, p = diebold_mariano(diff, h=int(h)) if model != benchmark else (np.nan, np.nan)
            out.append({"target": tgt, "horizon": h, "model": model, "n": len(gm), "origins": by_origin.size,
                        "mae": e.mean(), "rmse": np.sqrt(gm["sq_err"].mean()),
                        "mae_vs_persistence": e.mean() / bmae if model != benchmark else 1.0,
                        "origins_won_vs_persistence": float((diff < 0).mean()) if model != benchmark else np.nan,
                        "dm_stat": stat, "dm_p": p})
    return pd.DataFrame(out).sort_values(["target", "horizon", "mae"]).reset_index(drop=True)


def interval_coverage(results, nominal=0.90):
    """Share of outcomes inside the model's interval, by model and horizon, against the nominal level."""
    r = results.dropna(subset=["y_lo", "y_hi"])
    r = r.assign(inside=(r["y_true"] >= r["y_lo"]) & (r["y_true"] <= r["y_hi"]), width=r["y_hi"] - r["y_lo"])
    t = r.groupby(["model", "target", "horizon"]).agg(coverage=("inside", "mean"), mean_width=("width", "mean"), n=("inside", "size"))
    t["nominal"] = nominal
    return t.reset_index()


def paired_state_bootstrap(results, model, benchmark="persistence", n=1000, seed=20261006, level=0.95, exclude_years=()):
    """Ratio of a model's mean absolute error to the benchmark's, with a percentile interval from resampling STATES.
    Each state first contributes its mean absolute error over all of its test forecasts. This ignores correlation of errors across
    states within a year (common shocks), so the interval is too narrow; treat it as a lower bound on the uncertainty and read it
    together with the share of origins won. Returns (ratio, low, high)."""
    r = results
    if exclude_years:
        r = r[~(r["origin_year"].isin(exclude_years) | r["target_year"].isin(exclude_years))]
    ae = (r["y_true"] - r["y_pred"]).abs().rename("ae")
    r = r.assign(ae=ae)
    m = r[r["model"] == model].groupby("state_fips")["ae"].mean()
    b = r[r["model"] == benchmark].groupby("state_fips")["ae"].mean()
    common = m.index.intersection(b.index)
    m, b = m.loc[common].to_numpy(), b.loc[common].to_numpy()
    point = m.mean() / b.mean()
    rng = np.random.default_rng(seed)
    draws = np.empty(n)
    for k in range(n):
        pick = rng.integers(0, len(m), len(m))
        draws[k] = m[pick].mean() / b[pick].mean()
    lo, hi = np.percentile(draws, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return point, lo, hi
