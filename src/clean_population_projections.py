"""
Clean the Cooper Center (University of Virginia) July 2026 state population projections.

Reads:  data/raw/state_population_projections/WCCPS_NPP_V2026_TOTAL.xlsx
        data/raw/state_population_projections/WCCPS_NPP_V2026_AGE_SEX.xlsx
Writes: data/interim/population_projection_state_year.csv   total population by state and year
        data/interim/population_projection_age_sex.csv     population by state, year, sex, and five-year age band

Years in the source: 2020 (Census count) and projections for 2030, 2040, and 2050. There is no 2035. A 2035 row is
added by geometric interpolation between 2030 and 2040 (constant annual growth), with is_interpolated = 1. It is an
interpolation of a projection, not a separate projection, and should be labeled that way in any output. In the age
and sex file the 2035 rows are adjusted so that Male + Female equals Total and the age bands sum to each row's
population, as they do in the source years.

The U.S. total rows (FIPS 00) are dropped. Projected values are unrounded decimals and are kept as is.
"""

import pandas as pd

from common import FIPS_STATES, INTERIM, RAW

FOLDER = RAW / "state_population_projections"
SOURCE_YEARS = (2020, 2030, 2040, 2050)
AGE_BANDS = ["0 to 4", "5 to 9", "10 to 14", "15 to 19", "20 to 24", "25 to 29", "30 to 34", "35 to 39", "40 to 44",
             "45 to 49", "50 to 54", "55 to 59", "60 to 64", "65 to 69", "70 to 74", "75 to 79", "80 to 84", "85+"]
BAND_COLUMNS = {b: "pop_age_" + b.replace(" to ", "_").replace("+", "plus") for b in AGE_BANDS}


def read_table(path, sheet):
    raw = pd.read_excel(path, sheet_name=sheet, header=None, engine="calamine", dtype=str)
    header_row = raw.index[raw[0].astype(str).str.strip() == "FIPS"][0]
    df = raw.iloc[header_row + 1 :].copy()
    df.columns = [str(c).strip() for c in raw.iloc[header_row]]
    df["FIPS"] = df["FIPS"].astype(str).str.strip().str.zfill(2)
    return df[df["FIPS"].isin(FIPS_STATES)].reset_index(drop=True)


def interpolate_2035(df, value_cols, keys):
    a = df[df["year"] == 2030].set_index(keys)[value_cols]
    b = df[df["year"] == 2040].set_index(keys)[value_cols]
    mid = (a * b) ** 0.5  # geometric midpoint, 2035 is 5 of 10 years from 2030 to 2040
    mid = mid.reset_index()
    mid["year"] = 2035
    mid["is_interpolated"] = 1
    return mid


def total_table():
    raw = read_table(FOLDER / "WCCPS_NPP_V2026_TOTAL.xlsx", "Total Population")
    raw.columns = [str(c).split(".")[0] for c in raw.columns]  # "2030.0" becomes "2030"
    long = raw.melt(id_vars=["FIPS"], value_vars=[str(y) for y in SOURCE_YEARS], var_name="year", value_name="population")
    long["year"] = long["year"].astype(int)
    long["population"] = pd.to_numeric(long["population"])
    long = long.rename(columns={"FIPS": "state_fips"})
    long["is_interpolated"] = 0
    mid = interpolate_2035(long, ["population"], ["state_fips"])
    out = pd.concat([long, mid], ignore_index=True).sort_values(["state_fips", "year"]).reset_index(drop=True)
    out["population_source"] = out["year"].map(lambda y: "census_2020" if y == 2020 else ("interpolated" if y == 2035 else "projection"))
    return out


def age_sex_table():
    frames = []
    for year in SOURCE_YEARS:
        df = read_table(FOLDER / "WCCPS_NPP_V2026_AGE_SEX.xlsx", str(year))
        out = pd.DataFrame({"state_fips": df["FIPS"], "year": year, "sex": df["Sex"].str.strip(),
                            "population": pd.to_numeric(df["Total Population"])})
        for band, col in BAND_COLUMNS.items():
            out[col] = pd.to_numeric(df[band]).to_numpy()
        frames.append(out)
    long = pd.concat(frames, ignore_index=True)
    long["is_interpolated"] = 0
    value_cols = ["population"] + list(BAND_COLUMNS.values())
    mid = interpolate_2035(long, value_cols, ["state_fips", "sex"])
    # interpolating each column separately breaks the identities, so restore them: Male + Female equals the
    # interpolated Total, and the age bands in each row sum to that row's population
    total = mid[mid["sex"] == "Total"].set_index("state_fips")["population"]
    is_mf = mid["sex"] != "Total"
    mf_sum = mid[is_mf].groupby("state_fips")["population"].transform("sum")
    mid.loc[is_mf, "population"] = mid.loc[is_mf, "population"] * mid.loc[is_mf, "state_fips"].map(total) / mf_sum
    band_cols = list(BAND_COLUMNS.values())
    mid[band_cols] = mid[band_cols].mul(mid["population"] / mid[band_cols].sum(axis=1), axis=0)
    out = pd.concat([long, mid], ignore_index=True)
    return out.sort_values(["state_fips", "year", "sex"]).reset_index(drop=True)


def main():
    total = total_table()
    assert total["state_fips"].nunique() == 51 and not total.duplicated(["state_fips", "year"]).any()
    total.to_csv(INTERIM / "population_projection_state_year.csv", index=False)

    ages = age_sex_table()
    assert not ages.duplicated(["state_fips", "year", "sex"]).any()
    ages.to_csv(INTERIM / "population_projection_age_sex.csv", index=False)

    # checks: age bands sum to the total, male plus female equals total, and the Total sheet matches the age file
    band_sum = ages[list(BAND_COLUMNS.values())].sum(axis=1)
    print(f"max |age bands - population| across rows: {(band_sum - ages['population']).abs().max():.2f}")
    piv = ages.pivot_table(index=["state_fips", "year"], columns="sex", values="population")
    print(f"max |male + female - total|: {(piv['Male'] + piv['Female'] - piv['Total']).abs().max():.2f}")
    m = total.merge(piv["Total"].reset_index(), on=["state_fips", "year"], how="inner")
    print(f"max |Total sheet - age file total| over {len(m)} state-years: {(m['population'] - m['Total']).abs().max():.2f}")
    print(f"national: {total.groupby('year').population.sum().round(0).astype(int).to_dict()}")
    print(total[total.state_fips == "21"].round(0).to_string(index=False))


if __name__ == "__main__":
    main()
