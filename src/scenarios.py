"""
Stock-flow scenarios for 2030 and 2035 (CLAUDE.md Section 14, Part 4). These are SCENARIOS, not forecasts: they say what renter
households and extremely low income (ELI) renter households would be if population by age follows the Cooper Center projection and
renting behaves as described below. Nothing here is validated as a forecast; the hindcast in notebook 4.1 only checks the mechanics.

Method, per state
    renter households in age band b (householder age): 15-24, 25-34, 35-44, 45-54, 55-64, 65-74, 75-84, 85+
        R_b(t) = R_b(base) * [Cooper population in band b at t / at the base year]
    The base is the 2022-24 mean of ACS renter households by age (B25007), centered on 2023. Holding each band's renter rate
    (renter households per person) fixed means renter households grow only through population aging and growth.
    ELI renters = renter households * ELI share, the share held at the CHAS 2018-2022 value (ELI renters / all renters).
    The Cooper projection has years 2020, 2030, 2040, 2050 only. 2023 is a geometric interpolation of 2020 and 2030, and 2035 is the
    geometric interpolation of 2030 and 2040 already made by the cleaner (so 2035 is interpolated, as the plan requires us to say).

Scenarios (the low and high variants are bounds built from history, not confidence intervals)
    baseline   renter rates and the ELI share constant
    low        renter rates and the ELI share drift DOWN each year at the 10th percentile of their own historical annual drift across states
    high       ... drift UP at the 90th percentile
    Historical drift, renter rate: 2014-16 pooled age-specific rates applied to the 2022-24 age structure give the renter households that age
        alone predicts; actual divided by predicted is the rate effect, annualized over the 8 years between window centers (2015 and 2023).
    Historical drift, ELI share: CHAS 2010-2014 to 2018-2022, annualized over 8 years.
    The same national percentile is applied to every state, and the two drifts are combined at the same percentile, so low and high are
    deliberately wide: they are less likely than a 10 percent tail of either drift alone.
"""

import numpy as np
import pandas as pd

from common import INTERIM, PROCESSED

BANDS = ["15_24", "25_34", "35_44", "45_54", "55_64", "65_74", "75_84", "85plus"]
COOPER_PARTS = {"15_24": ["15_19", "20_24"], "25_34": ["25_29", "30_34"], "35_44": ["35_39", "40_44"], "45_54": ["45_49", "50_54"],
                "55_64": ["55_59", "60_64"], "65_74": ["65_69", "70_74"], "75_84": ["75_79", "80_84"], "85plus": ["85plus"]}
BASE_YEAR = 2023
BASE_WINDOW = (2022, 2023, 2024)
EARLY_WINDOW = (2014, 2015, 2016)
TARGET_YEARS = (2030, 2035)


def load_inputs():
    acs = pd.read_csv(INTERIM / "acs_age_tenure_state_year.csv", dtype={"state_fips": str})
    cooper = pd.read_csv(INTERIM / "population_projection_age_sex.csv", dtype={"state_fips": str})
    cooper = cooper[cooper["sex"] == "Total"].copy()
    for b, parts in COOPER_PARTS.items():
        cooper[f"cooper_{b}"] = cooper[[f"pop_age_{p}" for p in parts]].sum(axis=1)
    cooper = cooper.set_index(["state_fips", "year"])[[f"cooper_{b}" for b in BANDS]]
    chas = pd.read_csv(INTERIM / "chas_state_vintage.csv", dtype={"state_fips": str})
    panel = pd.read_csv(PROCESSED / "panel.csv", dtype={"state_fips": str})
    return acs, cooper, chas, panel


def cooper_pop(cooper, year):
    """Cooper population by band at any year from 2020 to 2040: the stored years as they are, others by geometric interpolation."""
    years = sorted(cooper.index.get_level_values("year").unique())
    if year in years:
        return cooper.xs(year, level="year")
    lo = max(y for y in years if y < year)
    hi = min(y for y in years if y > year)
    w = (year - lo) / (hi - lo)
    a, b = cooper.xs(lo, level="year"), cooper.xs(hi, level="year")
    return a ** (1 - w) * b ** w


def window_mean(acs, years, prefix):
    d = acs[acs["year"].isin(years)].groupby("state_fips")[[f"{prefix}_{b}" for b in BANDS]].mean()
    d.columns = BANDS
    return d


def renter_rates(acs, years):
    """Renter households per person by age band, pooled over the years (sum of households over sum of population)."""
    d = acs[acs["year"].isin(years)].groupby("state_fips")[[f"hh_renter_{b}" for b in BANDS] + [f"pop_{b}" for b in BANDS]].sum()
    rates = pd.DataFrame({b: d[f"hh_renter_{b}"] / d[f"pop_{b}"] for b in BANDS})
    return rates


def eli_share(chas, vintage):
    c = chas[chas["chas_vintage"] == vintage].set_index("state_fips")
    tot = c[[f"chas_renters_{b}_all" for b in ("le30", "30_50", "50_80", "80_100", "gt100")]].sum(axis=1)
    return c["chas_renters_le30_all"] / tot


def historical_drift(acs, chas):
    """Annualized drift of the renter rate (age-adjusted) 2015 to 2023 and of the ELI share 2012 to 2020 (CHAS windows), per state."""
    r_early = renter_rates(acs, EARLY_WINDOW)
    pop_late = window_mean(acs, BASE_WINDOW, "pop")
    predicted = (r_early * pop_late).sum(axis=1)
    actual = window_mean(acs, BASE_WINDOW, "hh_renter").sum(axis=1)
    rate_effect = actual / predicted
    g_rate = np.log(rate_effect) / 8.0
    share_ratio = eli_share(chas, "2018-2022") / eli_share(chas, "2010-2014")
    g_eli = np.log(share_ratio) / 8.0
    return pd.DataFrame({"rate_effect": rate_effect, "g_rate": g_rate, "eli_share_ratio": share_ratio, "g_eli": g_eli})


def drift_scenarios(drift):
    q = drift[["g_rate", "g_eli"]].quantile([0.1, 0.5, 0.9])
    return {"baseline": (0.0, 0.0), "low": (q.loc[0.1, "g_rate"], q.loc[0.1, "g_eli"]), "high": (q.loc[0.9, "g_rate"], q.loc[0.9, "g_eli"])}


def project(acs, cooper, chas, years=TARGET_YEARS, scenarios=None, drift=None):
    """Tidy frame: state_fips, year, scenario, renter_hh, eli_renters, and the inputs behind them."""
    drift = drift if drift is not None else historical_drift(acs, chas)
    scenarios = scenarios or drift_scenarios(drift)
    base_r = window_mean(acs, BASE_WINDOW, "hh_renter")
    base_pop = cooper_pop(cooper, BASE_YEAR)
    base_pop.columns = BANDS
    share = eli_share(chas, "2018-2022")
    rows = []
    for year in (BASE_YEAR,) + tuple(years):
        pop = cooper_pop(cooper, year)
        pop.columns = BANDS
        growth = pop / base_pop
        stock = (base_r * growth).sum(axis=1)
        dt = year - BASE_YEAR
        for name, (g_rate, g_eli) in scenarios.items():
            r = stock * np.exp(g_rate * dt)
            eli = r * share * np.exp(g_eli * dt)
            rows.append(pd.DataFrame({"state_fips": r.index, "year": year, "scenario": name, "renter_hh": r.to_numpy(), "eli_renters": eli.to_numpy(),
                                      "eli_share": (share * np.exp(g_eli * dt)).reindex(r.index).to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def hindcast(acs):
    """Constant 2014-16 renter rates with the ACS age structure of 2022-24: how well does age structure alone explain the change in renter
    households between the 2015 and 2023 window centers? Returns a per-state frame."""
    r_early = renter_rates(acs, EARLY_WINDOW)
    pop_early = window_mean(acs, EARLY_WINDOW, "pop")
    pop_late = window_mean(acs, BASE_WINDOW, "pop")
    hh_early = window_mean(acs, EARLY_WINDOW, "hh_renter").sum(axis=1)
    hh_late = window_mean(acs, BASE_WINDOW, "hh_renter").sum(axis=1)
    pred_age = (r_early * pop_late).sum(axis=1)
    pred_pop_only = hh_early * pop_late.sum(axis=1) / pop_early.sum(axis=1)
    return pd.DataFrame({"actual_growth_pct": 100 * (hh_late / hh_early - 1), "age_rates_growth_pct": 100 * (pred_age / hh_early - 1),
                         "population_only_growth_pct": 100 * (pred_pop_only / hh_early - 1)})


def need_table(acs, cooper, chas, panel, years=TARGET_YEARS):
    """Scenario results plus the policy quantities of CLAUDE.md Section 6, one row per state, year, and scenario.
    afford_units: renter-occupied units with complete facilities and rent affordable at 30 percent of HAMFI, CHAS 2018-2022, held fixed (no net
        additions or losses). It counts affordable units, not units that are affordable AND available.
    nhpd_*: active NHPD units whose LAST active subsidy ends by the year (lower and upper bound); a separate risk measure, not subtracted from
        afford_units because the two overlap only partly.
    permits_5plus_pa: mean annual permitted units in buildings of 5 or more units, 2022-24; new_renter_hh_pa: change in baseline renter households per
        year since 2023."""
    proj = project(acs, cooper, chas, years=years)
    base = proj[(proj.year == BASE_YEAR) & (proj.scenario == "baseline")].set_index("state_fips")
    units = chas[chas["chas_vintage"] == "2018-2022"].set_index("state_fips")["chas_units_rent_le_rhud30"]
    nh = pd.read_csv(INTERIM / "nhpd_expiry_state.csv", dtype={"state_fips": str}).set_index("state_fips")
    permits = panel[panel["year"].isin(BASE_WINDOW)].groupby("state_fips")["permits_5plus"].mean()
    out = proj.copy()
    out["eli_renters_2023"] = out["state_fips"].map(base["eli_renters"])
    out["renter_hh_2023"] = out["state_fips"].map(base["renter_hh"])
    out["renter_hh_change_pct"] = 100 * (out["renter_hh"] / out["renter_hh_2023"] - 1)
    out["eli_change"] = out["eli_renters"] - out["eli_renters_2023"]
    out["eli_change_pct"] = 100 * out["eli_change"] / out["eli_renters_2023"]
    out["afford_units"] = out["state_fips"].map(units)
    out["afford_per100_eli"] = 100 * out["afford_units"] / out["eli_renters"]
    out["extra_units_to_hold_2023_ratio"] = out["afford_units"] * (out["eli_renters"] / out["eli_renters_2023"] - 1)
    out["shortfall_vs_one_per_eli"] = (out["eli_renters"] - out["afford_units"]).clip(lower=0)
    for y in years:
        m = out["year"] == y
        for kind in ("lb", "ub"):
            out.loc[m, f"nhpd_last_end_units_{kind}"] = out.loc[m, "state_fips"].map(nh[f"nhpd_last_end_by_{y}_units_{kind}"])
    out["nhpd_active_units_ub"] = out["state_fips"].map(nh["nhpd_active_units_ub"])
    out["permits_5plus_pa"] = out["state_fips"].map(permits)
    dt = (out["year"] - BASE_YEAR).replace(0, np.nan)
    out["new_renter_hh_pa"] = (out["renter_hh"] - out["renter_hh_2023"]) / dt
    return out
