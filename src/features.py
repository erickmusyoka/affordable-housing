"""
Feature sets, timing rules, and the supervised dataset for forecasting (CLAUDE.md Section 14).

A forecast made "at origin T" may use only information that exists at the end of calendar year T. Each feature has a
publication lag in whole years (FEATURE_LAG): a feature with lag 1 is taken from year T - 1 when forecasting from T.
Lags are conservative. ACS 1-year data for year T are released in September of T + 1, so in practice an origin T
forecast is made late in T + 1; every lag below is counted from that same point. Examples:
    lag 0   ACS, BLS, BPS, and HUD fiscal-year FMR and income limits (published before or soon after the year ends)
    lag 1   BEA price parities and personal income, AHAR PIT and HIC counts, Picture of Subsidized Households, LIHTC

make_supervised() builds one row per state and origin year with the target h calendar years ahead. The horizon is
counted in calendar years, so the 2020 gap is respected: from origin 2019 the 1-year-ahead target (2020) does not exist
and no row is made.

validate_feature_set() enforces the exclusion rules: ACS burden bracket counts never predict a burden rate (circular),
FMR is not paired with ACS rent and HUD median family income is not paired with ACS income (shared source), and
LIHTC, HUD-assisted, and NHPD supply measures are not mixed as separate predictors of each other.
"""

import pandas as pd

KEY = ["state_fips", "year"]

# publication lag in whole years for every feature that may be used
FEATURE_LAG = {
    # ACS and derived from ACS
    "severe_rent_burden_pct": 0, "rent_burden_30_pct": 0, "rent_growth_pct": 0, "income_growth_pct": 0, "rent_income_gap_pp": 0,
    "rent_to_income_pct": 0, "rental_vacancy_rate_pct": 0, "renter_share_pct": 0, "renter_hh_growth_pct": 0,
    "population_growth_pct": 0, "home_value_growth_pct": 0, "median_gross_rent": 0, "median_hh_income": 0, "median_home_value": 0,
    # labor, permits
    "unemployment_rate_pct": 0, "unemployment_rate_change_pp": 0, "permits_per_1000_pop": 0, "permits_5plus_per_1000_pop": 0,
    "permits_5plus_per_1000_renter_hh": 0,
    # BEA
    "per_capita_personal_income_growth_pct": 1, "rpp_all_items": 1, "rpp_rents": 1, "income_rpp_adjusted": 1,
    # HUD policy benchmarks
    "fmr_2br_growth_pct": 0, "fmr_2br": 0, "hud_median_family_income": 0,
    # subsidized and shelter supply
    "psh_units_per_100_renter_hh": 1, "lihtc_units_cum_per_100_renter_hh": 1, "hic_psh_beds_per_10k": 1, "hic_shelter_beds_per_10k": 1,
}

FEATURE_SETS = {
    # ACS dynamics, labor market, and construction pipeline
    "core": ["rent_growth_pct", "income_growth_pct", "rental_vacancy_rate_pct", "renter_share_pct", "renter_hh_growth_pct",
             "population_growth_pct", "home_value_growth_pct", "unemployment_rate_pct", "unemployment_rate_change_pp",
             "permits_5plus_per_1000_renter_hh", "permits_per_1000_pop", "rent_to_income_pct"],
    # core plus price level and subsidized supply
    "extended": ["rent_growth_pct", "income_growth_pct", "rental_vacancy_rate_pct", "renter_share_pct", "renter_hh_growth_pct",
                 "population_growth_pct", "home_value_growth_pct", "unemployment_rate_pct", "unemployment_rate_change_pp",
                 "permits_5plus_per_1000_renter_hh", "permits_per_1000_pop", "rent_to_income_pct", "per_capita_personal_income_growth_pct",
                 "rpp_all_items", "psh_units_per_100_renter_hh", "lihtc_units_cum_per_100_renter_hh", "hic_psh_beds_per_10k"],
}

# groups that share a source; at most one member of each group may appear next to ACS series it derives from
SHARED_SOURCE_PAIRS = [
    ({"fmr_2br", "fmr_2br_growth_pct", "fmr_2br_to_acs_rent_ratio"}, {"median_gross_rent", "rent_growth_pct", "rent_to_income_pct"}),
    ({"hud_median_family_income"}, {"median_hh_income", "income_growth_pct", "income_rpp_adjusted", "rent_to_income_pct"}),
]
BRACKET_COLUMNS = {"rb_total", "rb_30_to_34", "rb_35_to_39", "rb_40_to_49", "rb_50_plus", "rb_not_computed"}
BURDEN_TARGETS = {"severe_rent_burden_pct", "rent_burden_30_pct"}


def validate_feature_set(feature_cols, target):
    cols = set(feature_cols)
    unknown = cols - set(FEATURE_LAG)
    if unknown:
        raise ValueError(f"features without a publication lag in FEATURE_LAG: {sorted(unknown)}")
    if target in BURDEN_TARGETS and cols & BRACKET_COLUMNS:
        raise ValueError(f"burden bracket counts cannot predict {target} (circular): {sorted(cols & BRACKET_COLUMNS)}")
    if target in BURDEN_TARGETS and cols & (BURDEN_TARGETS - {target}) and target != "rent_burden_30_pct":
        raise ValueError("the other burden rate contains the target's own numerator; do not use it as a predictor")
    for a, b in SHARED_SOURCE_PAIRS:
        if cols & a and cols & b:
            raise ValueError(f"shared-source overlap: {sorted(cols & a)} with {sorted(cols & b)}")
    return True


def _shifted(panel, col, years_back):
    """Value of col from years_back calendar years earlier, aligned to the row's year."""
    s = panel[KEY + [col]].copy()
    s["year"] = s["year"] + years_back
    return s.rename(columns={col: f"__{col}"})


def make_supervised(panel, target, horizon, feature_cols, target_lags=2):
    """One row per state and origin year T. Columns:
        origin_year, target_year (= T + horizon), y_target, y_origin (target at T), y_lag1 .. y_lag{target_lags}
        (target at T-1, T-2), and each feature at year T - FEATURE_LAG[feature]."""
    validate_feature_set(feature_cols, target)
    base = panel[KEY + [target]].rename(columns={"year": "origin_year", target: "y_origin"})
    out = base.copy()
    tgt = _shifted(panel, target, -horizon).rename(columns={"year": "origin_year", f"__{target}": "y_target"})
    out = out.merge(tgt, on=["state_fips", "origin_year"], how="left")
    out["target_year"] = out["origin_year"] + horizon
    for k in range(1, target_lags + 1):
        lag = _shifted(panel, target, k).rename(columns={"year": "origin_year", f"__{target}": f"y_lag{k}"})
        out = out.merge(lag, on=["state_fips", "origin_year"], how="left")
    for col in feature_cols:
        s = _shifted(panel, col, FEATURE_LAG[col]).rename(columns={"year": "origin_year", f"__{col}": col})
        out = out.merge(s, on=["state_fips", "origin_year"], how="left")
    out = out.dropna(subset=["y_origin", "y_target"]).reset_index(drop=True)
    return out.sort_values(["origin_year", "state_fips"]).reset_index(drop=True)


def feature_columns(supervised):
    skip = {"state_fips", "origin_year", "target_year", "y_target", "y_origin"}
    return [c for c in supervised.columns if c not in skip]
