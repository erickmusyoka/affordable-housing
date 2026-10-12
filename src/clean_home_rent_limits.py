"""
Clean the HUD HOME rent limits into a county-level table.

Reads:  data/raw/home_rent_limits/HOME_Limits_Rents_2024.xlsx   fiscal year 2024
        data/raw/home_rent_limits/HOME_AllLimits_2019.xlsx      fiscal year 2019 (also has the 30/50/60/80 percent income limits, not used here)
Writes: data/interim/home_rent_limits_county.csv                one row per county and fiscal year

HOME rent limits (24 CFR 92.252): the High HOME rent is the lesser of the fair market rent (FMR) or the rent affordable at 65 percent of area median
income; the Low HOME rent is the rent affordable at 50 percent (also capped by FMR). Columns per bedroom count b = 0 to 4:
    fmr_b, lowrent_b, highrent_b, homerent50_b, homerent65_b
homerent50 and homerent65 are the uncapped rents at 50 and 65 percent of median, so the rent affordable at 60 percent of median (the LIHTC level)
is 1.2 times homerent50 (income scales linearly with the percent of median, with the same bedroom and occupancy assumptions).
Files are by HUD area; areas in New England are towns, so rows are averaged to the county (state FIPS + county code) and n_areas records how many
areas were averaged. Territories are dropped. Column case differs between the two files and is lowercased.
"""

import pandas as pd

from common import FIPS_STATES, INTERIM, RAW

FILES = {2019: "HOME_AllLimits_2019.xlsx", 2024: "HOME_Limits_Rents_2024.xlsx"}
BEDS = range(5)
RENT_COLS = [f"{k}_{b}" for k in ("fmr", "lowrent", "highrent", "homerent50", "homerent65") for b in BEDS]


def load(year, name):
    d = pd.read_excel(RAW / "home_rent_limits" / name, engine="calamine")
    d.columns = [str(c).strip().lower() for c in d.columns]
    d = d.rename(columns={"county": "county_code"}) if "county" in d.columns else d
    missing = [c for c in ["state", "county_code", "fips2010"] + RENT_COLS if c not in d.columns]
    assert not missing, f"{name}: missing columns {missing}"
    assert not d["fips2010"].duplicated().any(), f"{name}: duplicate area codes"
    d["state_fips"] = d["state"].astype(int).astype(str).str.zfill(2)
    d["county_fips"] = d["state_fips"] + d["county_code"].astype(int).astype(str).str.zfill(3)
    d = d[d["state_fips"].isin(FIPS_STATES)].copy()
    for c in RENT_COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    # sanity: the High HOME rent never exceeds the FMR, and the Low HOME rent never exceeds the High HOME rent
    assert (d["highrent_2"] <= d["fmr_2"] + 1).all(), f"{name}: High HOME rent above FMR"
    assert (d["lowrent_2"] <= d["highrent_2"] + 1).all(), f"{name}: Low HOME rent above High HOME rent"
    g = d.groupby(["state_fips", "county_fips"])
    out = g[RENT_COLS].mean().round(1)
    out["n_areas"] = g.size()
    out["year"] = year
    return out.reset_index()


def main():
    t = pd.concat([load(y, f) for y, f in FILES.items()], ignore_index=True)
    assert not t.duplicated(["county_fips", "year"]).any()
    t["rent60_2br"] = 1.2 * t["homerent50_2"]
    t["fmr_to_rent60_2br"] = t["fmr_2"] / t["rent60_2br"]
    t.to_csv(INTERIM / "home_rent_limits_county.csv", index=False)
    for y, g in t.groupby("year"):
        print(f"FY{y}: {len(g)} counties in {g.state_fips.nunique()} states; FMR 2BR median {g.fmr_2.median():.0f}; "
              f"60 percent rent 2BR median {g.rent60_2br.median():.0f}; FMR above the 60 percent rent in {(g.fmr_to_rent60_2br > 1).mean():.0%} of counties")


if __name__ == "__main__":
    main()
