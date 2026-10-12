"""
Clean HUD CHAS state tables (Table 8 and Table 15C) for the four vintages.

Reads:  data/raw/chas/<vintage>/Table8.csv, Table15C.csv and the vintage's data dictionary workbook
Writes: data/interim/chas_state_vintage.csv   one row per state per vintage (204 rows)

The panel year of a vintage is the END year of its five-year window (2009, 2014, 2019, 2022). That is the
proposed rule in CLAUDE.md and is not final. The 2018-2022 window shares 2018 and 2019 with 2015-2019, so those
two rows are not independent (overlaps_previous_vintage = 1). These are four snapshots, not an annual series.

Table 8 (renters by income band by housing cost burden) and Table 15C (renter-occupied units by rent
affordability and household income) have identical column IDs and labels in all four vintages. Each column ID used
below is checked against the vintage's dictionary before use, so a layout change stops the script.

Income bands are shares of HAMFI (HUD area median family income): le30 (extremely low), 30_50, 50_80, 80_100, gt100.
Cost burden is gross housing cost as a share of income: le30, 30_50, gt50, and "not computed" (no or negative income).
Severe burden is gt50. Burden rates exclude the not-computed households from the denominator.

Counts are rounded by the Census Bureau (0 stays 0, 1 to 7 becomes 4, 8 and above to the nearest 5), so
components may not add exactly to totals at the state level.
"""

import re

import pandas as pd

from common import INTERIM, RAW, STATE_FIPS, check_panel, read_excel

VINTAGES = {
    "2005-2009": "CHAS data dictionary 05-09.xlsx",
    "2010-2014": "CHAS data dictionary 10-14.xlsx",
    "2015-2019": "CHAS data dictionary 15-19.xlsx",
    "2018-2022": "CHAS-data-dictionary-18-22.xlsx",
}
BANDS = ["le30", "30_50", "50_80", "80_100", "gt100"]
# Table 8 renter columns: (all, burden le30, burden 30_50, burden gt50, burden not computed) for each band
T8_COLUMNS = {
    "le30": [69, 70, 73, 76, 79],
    "30_50": [82, 83, 86, 89, 92],
    "50_80": [95, 96, 99, 102, 105],
    "80_100": [108, 109, 112, 115, 118],
    "gt100": [121, 122, 125, 128, 131],
}
BURDENS = ["all", "b_le30", "b_30_50", "b_gt50", "b_nc"]
# Table 15C columns (units with complete facilities): rent affordability level, any occupant income, and
# extremely low income renters living in those units
T15_UNITS = {"rent_le_rhud30": 4, "rent_rhud30_50": 25, "rent_rhud50_80": 46, "rent_gt_rhud80": 67}
T15_ELI_IN_AFFORDABLE = 5

BAND_LABEL = {
    "le30": "less than or equal to 30% of HAMFI",
    "30_50": "greater than 30% but less than or equal to 50% of HAMFI",
    "50_80": "greater than 50% but less than or equal to 80% of HAMFI",
    "80_100": "greater than 80% but less than or equal to 100% of HAMFI",
    "gt100": "greater than 100% of HAMFI",
}
BURDEN_LABEL = {
    "all": "All",
    "b_le30": "less than or equal to 30%",
    "b_30_50": "greater than 30% but less than or equal to 50%",
    "b_gt50": "greater than 50%",
    "b_nc": "not computed (no/negative income)",
}


def check_dictionary(vintage, dict_file):
    """Stop if any column ID used below does not describe what this script assumes."""
    book = pd.ExcelFile(RAW / "chas" / vintage / dict_file, engine="calamine")
    t8 = read_excel(RAW / "chas" / vintage / dict_file, sheet_name="Table 8")
    t8.columns = [str(c).strip() for c in t8.columns]
    t8 = t8.set_index("Column Name")
    for band, ids in T8_COLUMNS.items():
        for burden, num in zip(BURDENS, ids):
            row = t8.loc[f"T8_est{num}"]
            assert "Renter" in str(row["Tenure"]), f"{vintage} T8_est{num}: not a renter column"
            assert str(row["Household income"]).strip() == BAND_LABEL[band], f"{vintage} T8_est{num}: income band {row['Household income']!r}"
            assert str(row["Cost burden"]).strip() == BURDEN_LABEL[burden], f"{vintage} T8_est{num}: burden {row['Cost burden']!r}"
            assert str(row["Facilities"]).strip() == "All", f"{vintage} T8_est{num}: facilities split"
    t15 = read_excel(RAW / "chas" / vintage / dict_file, sheet_name="Table 15C")
    t15.columns = [str(c).strip() for c in t15.columns]
    t15 = t15.set_index("Column Name")
    for name, num in T15_UNITS.items():
        row = t15.loc[f"T15C_est{num}"]
        assert "complete kitchen and plumbing" in str(row["Facilities"]), f"{vintage} T15C_est{num}: facilities"
        assert str(row["Household income"]).strip() == "All", f"{vintage} T15C_est{num}: income"
        assert str(row["Number of bedrooms"]).strip() == "All", f"{vintage} T15C_est{num}: bedrooms"
    assert str(t15.loc[f"T15C_est{T15_ELI_IN_AFFORDABLE}", "Household income"]).strip() == BAND_LABEL["le30"]
    print(f"{vintage}: dictionary check passed")


def read_vintage(vintage):
    start, end = (int(x) for x in vintage.split("-"))
    folder = RAW / "chas" / vintage
    t8 = pd.read_csv(folder / "Table8.csv")
    t15 = pd.read_csv(folder / "Table15C.csv")
    for t in (t8, t15):
        t.columns = [c.lower() for c in t.columns]
    t8 = t8[t8["st"] != 72].set_index("st")  # drop Puerto Rico
    t15 = t15[t15["st"] != 72].set_index("st")
    assert len(t8) == 51 and len(t15) == 51, f"{vintage}: expected 51 states, got {len(t8)} and {len(t15)}"

    out = pd.DataFrame(index=t8.index)
    out["state_fips"] = [f"{int(s):02d}" for s in out.index]
    out["chas_vintage"], out["chas_start_year"], out["chas_end_year"] = vintage, start, end
    out["year"] = end
    for band, ids in T8_COLUMNS.items():
        for burden, num in zip(BURDENS, ids):
            out[f"chas_renters_{band}_{burden}"] = t8[f"t8_est{num}"]
    for name, num in T15_UNITS.items():
        out[f"chas_units_{name}"] = t15[f"t15c_est{num}"]
    out["chas_eli_renters_in_units_rent_le_rhud30"] = t15[f"t15c_est{T15_ELI_IN_AFFORDABLE}"]
    return out.reset_index(drop=True)


def add_measures(df):
    g = lambda band, b: df[f"chas_renters_{band}_{b}"]
    for band in BANDS:
        base = g(band, "all") - g(band, "b_nc")
        df[f"chas_severe_burden_pct_{band}"] = 100 * g(band, "b_gt50") / base
        df[f"chas_burden30_pct_{band}"] = 100 * (g(band, "b_30_50") + g(band, "b_gt50")) / base
    base50 = (g("le30", "all") + g("30_50", "all")) - (g("le30", "b_nc") + g("30_50", "b_nc"))
    df["chas_severe_burden_pct_le50"] = 100 * (g("le30", "b_gt50") + g("30_50", "b_gt50")) / base50
    df["chas_burden30_pct_le50"] = 100 * (g("le30", "b_30_50") + g("30_50", "b_30_50") + g("le30", "b_gt50") + g("30_50", "b_gt50")) / base50
    df["chas_renters_le50"] = g("le30", "all") + g("30_50", "all")
    # units whose rent is affordable at that income level (any occupant income), per 100 renter households
    # in the income band. These count affordable units, not units that are affordable AND available
    df["chas_affordable_units_per100_eli"] = 100 * df["chas_units_rent_le_rhud30"] / g("le30", "all")
    df["chas_affordable_units_per100_le50"] = 100 * (df["chas_units_rent_le_rhud30"] + df["chas_units_rent_rhud30_50"]) / df["chas_renters_le50"]
    df["chas_eli_in_affordable_pct_of_eli"] = 100 * df["chas_eli_renters_in_units_rent_le_rhud30"] / g("le30", "all")
    return df


def main():
    frames = []
    for vintage, dict_file in VINTAGES.items():
        check_dictionary(vintage, dict_file)
        frames.append(read_vintage(vintage))
    df = pd.concat(frames, ignore_index=True)
    df["overlaps_previous_vintage"] = (df["chas_vintage"] == "2018-2022").astype(int)
    df = add_measures(df).sort_values(["state_fips", "year"]).reset_index(drop=True)
    check_panel(df, "chas_state_vintage")
    df.to_csv(INTERIM / "chas_state_vintage.csv", index=False)
    print(df.groupby("chas_vintage")[["chas_severe_burden_pct_le50", "chas_affordable_units_per100_eli"]].agg(["mean", "min", "max"]).round(1).to_string())
    print(df[df.state_fips == "21"][["chas_vintage", "year", "chas_renters_le50", "chas_severe_burden_pct_le50", "chas_severe_burden_pct_le30", "chas_affordable_units_per100_eli", "chas_affordable_units_per100_le50"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
