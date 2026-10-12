"""
Part 8: what could be done. Planning arithmetic on top of the prediction models; NOT effect estimates.

The prediction models say what is likely to happen and where (units likely to be lost, how many households need housing). They do not say which action
works or what it costs: there is no cost data and no policy variation in the data. This module therefore answers three narrower questions with the model
outputs, and every result is conditional on the stated assumptions:

    targeting       if preservation effort could be aimed at properties, in what order should it go, and how many properties does it take to cover a share of the
                    units that the rescue model expects not to be rescued? Compared rules: largest first (most units per action), lowest rescue chance first
                    (smallest properties), and random. "Covered" means the expected unrescued units in the properties acted on; it assumes an action succeeds.
    capacity        how does the units-at-stake figure compare with current LIHTC output (units placed in service, 2015-2019, the last complete years)? Not all LIHTC
                    output is net new, so this is a scale comparison.
    levers          the 2035 ratio of rent-affordable units to ELI renters under combinations of preservation and new production, and what it takes to hold the
                    2023 ratio. The stock is the CHAS 2018-2022 count held fixed; losses are taken out of it in full (an upper bound on deterioration, because some
                    subsidized units that leave a program stay rent-affordable), demand is the baseline scenario.
"""

import numpy as np
import pandas as pd

import preservation as P
from common import INTERIM

HORIZON_YEARS = 2035 - 2023
LIHTC_YEARS = (2015, 2019)


def scored_properties():
    """HOME-only and HUD-program properties whose last active subsidy ends 2026 to 2035, with the modeled rescue chance and expected unrescued units."""
    ev = P.build_events()
    q, _ = P.rescue_projection(ev)
    s = q[q.p_rescue.notna()].copy()
    s["unrescued_ub"] = s.units_ub * (1 - s.p_rescue)
    s["unrescued_lb"] = s.units_lb * (1 - s.p_rescue)
    return s


def coverage_curve(df, key, value="unrescued_ub"):
    """Cumulative share of the value covered as properties are acted on in descending order of `key`."""
    o = df.assign(_k=key).sort_values("_k", ascending=False, kind="stable")
    cum = o[value].cumsum().to_numpy() / o[value].sum()
    return pd.DataFrame({"properties": np.arange(1, len(o) + 1), "share_properties": np.arange(1, len(o) + 1) / len(o), "share_covered": cum})


def properties_for_share(curve, share):
    i = int(np.searchsorted(curve.share_covered.to_numpy(), share) + 1)
    return i, i / len(curve)


def targeting_table(df, shares=(0.25, 0.5, 0.8)):
    rules = {"largest first": df.unrescued_ub, "lowest rescue chance first": -df.p_rescue}
    rows = []
    for name, key in rules.items():
        c = coverage_curve(df, key)
        for s in shares:
            n, f = properties_for_share(c, s)
            rows.append({"rule": name, "share of unrescued units covered": s, "properties to act on": n, "share of scored properties": f})
    for s in shares:
        rows.append({"rule": "random", "share of unrescued units covered": s, "properties to act on": int(round(s * len(df))), "share of scored properties": s})
    return pd.DataFrame(rows)


def size_bands(df):
    b = pd.cut(df.total_units, [0, 10, 25, 50, 100, 200, 10**6], labels=["1-10", "11-25", "26-50", "51-100", "101-200", "200+"])
    t = df.assign(band=b).groupby("band", observed=True).agg(properties=("property_id", "size"), units=("units_ub", "sum"), rescue_chance=("p_rescue", "mean"), unrescued=("unrescued_ub", "sum"))
    for c, name in (("properties", "share_properties"), ("units", "share_units"), ("unrescued", "share_unrescued")):
        t[name] = t[c] / t[c].sum()
    t["unrescued_per_property"] = t["unrescued"] / t["properties"]
    return t


def capacity_by_state(stake):
    """Stake per year and years of current LIHTC output needed. `stake` is outputs/tables/units_at_stake_state.csv."""
    l = pd.read_csv(INTERIM / "lihtc_state_year.csv", dtype={"state_fips": str})
    a, b = LIHTC_YEARS
    out = l[l.year.between(a, b)].groupby("state_fips").lihtc_units.sum() / (b - a + 1)
    t = stake.set_index("state_fips").copy()
    t["lihtc_per_year"] = out
    t["stake_per_year"] = t.stake_mid / HORIZON_YEARS
    t["years_of_lihtc_needed"] = t.stake_mid / t.lihtc_per_year
    return t.reset_index()


def lever_grid(stock, eli_2035, losses, preserved_shares=(0, 0.25, 0.5, 0.75, 1.0), build_per_year=(0, 25_000, 50_000, 75_000, 100_000)):
    """2035 affordable units per 100 ELI renters: (stock - losses x (1 - share preserved) + years x new units per year) / ELI renters."""
    g = pd.DataFrame(index=[f"{int(s * 100)}% preserved" for s in preserved_shares], columns=[f"{b // 1000}k per year" for b in build_per_year], dtype=float)
    for s, rn in zip(preserved_shares, g.index):
        for b, cn in zip(build_per_year, g.columns):
            g.loc[rn, cn] = 100 * (stock - losses * (1 - s) + HORIZON_YEARS * b) / eli_2035
    return g
