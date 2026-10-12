"""
Clean the ACS 1-year tenure-by-age and population-by-age tables used by the Part 4 stock-flow scenarios.

Reads:  data/raw/acs1_age/acs1_age_state_<year>.csv   (written by fetch-acs-age-data.py; B25007 and B01001 estimates)
Writes: data/interim/acs_age_tenure_state_year.csv    one row per state and year, 2010 to 2024 excluding 2020

Eight householder or population age bands, chosen so they line up with the Cooper Center five-year bands:
    15-24, 25-34, 35-44, 45-54, 55-64, 65-74, 75-84, 85+
For each band b:
    hh_renter_<b>, hh_owner_<b>   households by tenure and age of householder (B25007)
    pop_<b>                       population in the band (B01001, both sexes)
Totals are checked against the ACS panel (renter households and total households must match B25003 exactly).
Rates (renter households per person, headship, renter share) are computed in src/scenarios.py, not here.
"""

import pandas as pd

from common import FIPS_STATES, INTERIM, RAW, check_panel

BANDS = ["15_24", "25_34", "35_44", "45_54", "55_64", "65_74", "75_84", "85plus"]
# B25007 owner rows 003 to 011 and renter rows 013 to 021: 15-24, 25-34, 35-44, 45-54, 55-59, 60-64, 65-74, 75-84, 85+
OWNER = {"15_24": [3], "25_34": [4], "35_44": [5], "45_54": [6], "55_64": [7, 8], "65_74": [9], "75_84": [10], "85plus": [11]}
RENTER = {b: [i + 10 for i in rows] for b, rows in OWNER.items()}
# B01001 male rows 006 to 025 and female rows 030 to 049
MALE = {"15_24": [6, 7, 8, 9, 10], "25_34": [11, 12], "35_44": [13, 14], "45_54": [15, 16], "55_64": [17, 18, 19],
        "65_74": [20, 21, 22], "75_84": [23, 24], "85plus": [25]}
FEMALE = {b: [i + 24 for i in rows] for b, rows in MALE.items()}


def col(group, i):
    return f"{group}_{i:03d}E"


def main():
    frames = []
    for path in sorted((RAW / "acs1_age").glob("acs1_age_state_*.csv")):
        year = int(path.stem.split("_")[-1])
        d = pd.read_csv(path, dtype={"state": str})
        out = pd.DataFrame({"state_fips": d["state"].str.zfill(2), "year": year})
        for b in BANDS:
            out[f"hh_renter_{b}"] = d[[col("B25007", i) for i in RENTER[b]]].sum(axis=1, min_count=len(RENTER[b])).to_numpy()
            out[f"hh_owner_{b}"] = d[[col("B25007", i) for i in OWNER[b]]].sum(axis=1, min_count=len(OWNER[b])).to_numpy()
            out[f"pop_{b}"] = d[[col("B01001", i) for i in MALE[b] + FEMALE[b]]].sum(axis=1, min_count=1).to_numpy()
        # integrity: the age rows must add up to the table's own renter and owner totals
        assert (out[[f"hh_renter_{b}" for b in BANDS]].sum(axis=1) == d[col("B25007", 12)].to_numpy()).all(), f"{year}: renter age rows do not sum to the renter total"
        assert (out[[f"hh_owner_{b}" for b in BANDS]].sum(axis=1) == d[col("B25007", 2)].to_numpy()).all(), f"{year}: owner age rows do not sum to the owner total"
        frames.append(out[out["state_fips"].isin(FIPS_STATES)])
    df = pd.concat(frames, ignore_index=True).sort_values(["state_fips", "year"]).reset_index(drop=True)

    panel = pd.read_csv("data/processed/panel.csv", dtype={"state_fips": str})[["state_fips", "year", "hh_renter", "hh_owner"]]
    m = df.merge(panel, on=["state_fips", "year"], how="left")
    assert (m[[f"hh_renter_{b}" for b in BANDS]].sum(axis=1) == m["hh_renter"]).all(), "renter households differ from the ACS panel (B25003)"
    assert (m[[f"hh_owner_{b}" for b in BANDS]].sum(axis=1) == m["hh_owner"]).all(), "owner households differ from the ACS panel (B25003)"
    check_panel(df, "acs_age_tenure")
    df.to_csv(INTERIM / "acs_age_tenure_state_year.csv", index=False)
    last = df[df.year == df.year.max()]
    print(f"2024 national renter households by age band: " + ", ".join(f"{b} {last[f'hh_renter_{b}'].sum() / 1e6:.2f}m" for b in BANDS))


if __name__ == "__main__":
    main()
