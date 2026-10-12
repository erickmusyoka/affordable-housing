"""
Clean the HUD LIHTC property database (LIHTCPUB.xlsx) into state tables.

Reads:  data/raw/lihtc/LIHTCPUB.xlsx
Writes: data/interim/lihtc_state_year.csv    projects and units by placed-in-service year, plus cumulative sums
        data/interim/lihtc_expiry_state.csv  units by estimated end of affordability, one row per state
        data/interim/lihtc_projects.csv.gz   one row per project (the 51 geographies), recoded fields for the Part 5 exit model

Placed-in-service year: yr_pis from 1987 to 2024 is used as is. The codes 8888 (not confirmed) and 9999 (year
missing) are replaced by allocation year + 2 when that year is valid (the same convention as NHPD, which uses
allocation year + 32 for the end date), otherwise the project is left out of the year table and counted in the
printed exclusions. Territories (PR, AS, VI, GU, MP) are dropped.

Estimated end of affordability: placed-in-service year + 30 for allocations from 1990 on, + 15 for allocations
before 1990, or placed-in-service year + aff_yrs when aff_yrs is longer than that base. aff_yrs is filled only
for the 5 percent of projects with a period over 30 years. This is an estimate, not a recorded restriction date,
and it is the same rule NHPD applies to its LIHTC records.

nonprog = 1 means the allocating agency no longer monitors the project for LIHTC compliance (extended use
completed, qualified contract, or other). The dictionary says this does not mean the project stopped being
affordable. Cumulative columns are given both with and without those projects.

The two most recent placement years look incomplete (units placed in service fall from 96.8 thousand in 2022 to
80.6 thousand in 2023 and 64.3 thousand in 2024), so is_recent_incomplete is 1 for 2023 and 2024.
"""

import pandas as pd

from common import INTERIM, RAW, STATE_FIPS, check_panel, read_excel

FIRST_YEAR, LAST_YEAR = 1987, 2024
AS_OF = 2025  # expiry windows are counted from the year after this
WINDOWS = {"2030": 2030, "2035": 2035}


def load():
    d = read_excel(RAW / "lihtc" / "LIHTCPUB.xlsx")
    assert len(d) == 55345, f"unexpected row count {len(d)}"
    assert not d["hud_id"].duplicated().any()
    n_all = len(d)
    d = d[d["proj_st"].isin(STATE_FIPS)].copy()
    print(f"rows: {n_all}; dropped {n_all - len(d)} outside the 50 states and DC")
    d["state_fips"] = d["proj_st"].map(STATE_FIPS)

    pis = d["yr_pis"].where(d["yr_pis"].between(FIRST_YEAR, LAST_YEAR))
    alloc = d["yr_alloc"].where(d["yr_alloc"].between(FIRST_YEAR, LAST_YEAR))
    d["pis_year"] = pis.fillna(alloc + 2).clip(upper=LAST_YEAR)
    d["pis_from_alloc"] = pis.isna() & alloc.notna()
    print(f"placed-in-service year: valid {pis.notna().sum()}, filled from allocation year {d['pis_from_alloc'].sum()}, "
          f"unusable (left out of year table) {d['pis_year'].isna().sum()}")

    base = (alloc < 1990).map({True: 15, False: 30})
    base = base.fillna(30)
    longer = d["aff_yrs"].where(d["aff_yrs"] > base)
    d["affordability_years"] = longer.fillna(base)
    d["est_end_year"] = d["pis_year"] + d["affordability_years"]
    d["monitored"] = (d["nonprog"] != 1)
    print(f"li_units > n_units in {(d['li_units'] > d['n_units']).sum()} rows (kept as reported)")
    return d


def year_table(d):
    p = d.dropna(subset=["pis_year"]).copy()
    p["year"] = p["pis_year"].astype(int)
    g = p.groupby(["state_fips", "year"])
    t = g.agg(lihtc_projects=("hud_id", "size"), lihtc_units=("n_units", "sum"), lihtc_li_units=("li_units", "sum"))
    t["lihtc_units_not_monitored"] = p[~p["monitored"]].groupby(["state_fips", "year"])["n_units"].sum()
    t = t.fillna({"lihtc_units_not_monitored": 0})

    full = pd.MultiIndex.from_product([sorted(set(STATE_FIPS.values())), range(FIRST_YEAR, LAST_YEAR + 1)], names=["state_fips", "year"])
    t = t.reindex(full, fill_value=0).reset_index().sort_values(["state_fips", "year"])
    by_state = t.groupby("state_fips")
    t["lihtc_projects_cum"] = by_state["lihtc_projects"].cumsum()
    t["lihtc_units_cum"] = by_state["lihtc_units"].cumsum()
    t["lihtc_li_units_cum"] = by_state["lihtc_li_units"].cumsum()
    t["lihtc_units_cum_monitored"] = t["lihtc_units_cum"] - by_state["lihtc_units_not_monitored"].cumsum()
    t["is_recent_incomplete"] = (t["year"] >= LAST_YEAR - 1).astype(int)
    return t.reset_index(drop=True)


def expiry_table(d):
    p = d.dropna(subset=["est_end_year"]).copy()
    m = p[p["monitored"]]
    rows = []
    for st in sorted(set(STATE_FIPS.values())):
        a, b = m[m["state_fips"] == st], p[p["state_fips"] == st]
        row = {
            "state_fips": st,
            "lihtc_units_all": b["n_units"].sum(),
            "lihtc_units_monitored": a["n_units"].sum(),
            "lihtc_units_monitored_past_est_end": a.loc[a["est_end_year"] <= AS_OF, "n_units"].sum(),
        }
        for label, year in WINDOWS.items():
            w = a[(a["est_end_year"] > AS_OF) & (a["est_end_year"] <= year)]
            row[f"lihtc_units_est_end_by_{label}"] = w["n_units"].sum()
            row[f"lihtc_li_units_est_end_by_{label}"] = w["li_units"].sum()
        rows.append(row)
    t = pd.DataFrame(rows)
    for label in WINDOWS:
        t[f"pct_monitored_est_end_by_{label}"] = 100 * t[f"lihtc_units_est_end_by_{label}"] / t["lihtc_units_monitored"]
    t["as_of_year"] = AS_OF
    return t


PROJECT_KEEP = ["hud_id", "state_fips", "cnty2020", "trct2020", "pis_year", "pis_from_alloc", "yr_alloc", "est_end_year", "affordability_years",
                "n_units", "li_units", "allocamt", "type", "credit", "metro", "dda", "qct", "inc_ceil", "aff_period", "aff_yrs", "nonprog", "nlm_reason"]


def yes(series):
    """The dictionary codes yes as 1 and no as 2; blank means not reported. Returns 1, 0, or NaN."""
    return series.map({1: 1.0, 2: 0.0})


def project_table(d):
    """One row per project with the fields known at placement, recoded, and the outcome. nonprog = 1 means the project is no longer monitored for
    LIHTC compliance; the dictionary says this does not mean it stopped being affordable, and it was reported informally before 2012."""
    p = d[PROJECT_KEEP].copy()
    for c in ["non_prof", "basis", "bond", "fmha_514", "fmha_515", "fmha_538", "home", "cdbg", "tcap", "rentassist", "fha", "rad", "mff_ra",
              "trgt_pop", "trgt_fam", "trgt_eld", "trgt_dis", "trgt_hml", "scattered_site_cd", "resyndication_cd"]:
        p[c] = yes(d[c])
    p["rhs"] = d[["fmha_514", "fmha_515", "fmha_538"]].eq(1).any(axis=1).astype(int)
    # 5-digit county FIPS from the first complete 11-digit tract code among the four census vintages (placeholders such as 37XXXXXXXXX and "." are skipped);
    # the 3-digit cnty2020 field is blank for some projects, so it is not used
    county = pd.Series(pd.NA, index=d.index, dtype="object")
    for col in ["fips2010", "fips2020", "fips2000", "fips1990"]:
        t = d[col].astype(str).str.strip()
        county = county.where(county.notna(), t.where(t.str.fullmatch(r"\d{11}")).str[:5])
    p["county_fips"] = county
    p["tract_fips"] = d["fips2010"].astype(str).str.strip().where(d["fips2010"].astype(str).str.strip().str.fullmatch(r"\d{11}"))
    p["state_fips"] = p["state_fips"].astype(str).str.zfill(2)
    p["n_units"] = p["n_units"].where(p["n_units"] > 0)
    p["li_units"] = p["li_units"].where(p["li_units"] > 0).clip(upper=p["n_units"])
    p["li_share"] = p["li_units"] / p["n_units"]
    p["alloc_pre1990"] = (d["yr_alloc"].where(d["yr_alloc"].between(1987, 2024)) < 1990).astype(float).where(d["yr_alloc"].between(1987, 2024))
    p["age_2024"] = 2024 - p["pis_year"]
    p["left_monitoring"] = (p["nonprog"] == 1).astype(int)
    assert p["hud_id"].is_unique and p["state_fips"].str.len().eq(2).all()
    known = p["county_fips"].notna()
    assert (p.loc[known, "county_fips"].str[:2] == p.loc[known, "state_fips"]).mean() > 0.99, "county FIPS disagrees with the project state"
    print(f"county FIPS found for {known.mean():.1%} of projects")
    assert p["left_monitoring"].sum() == (d["nonprog"] == 1).sum()
    return p.sort_values("hud_id").reset_index(drop=True)


def main():
    d = load()
    proj = project_table(d)
    proj.to_csv(INTERIM / "lihtc_projects.csv.gz", index=False)
    print(f"lihtc_projects: {len(proj)} projects, {int(proj.left_monitoring.sum())} no longer monitored, {proj.pis_year.notna().sum()} with a usable placed-in-service year")
    years = year_table(d)
    # every state has a row for every year 1987 to 2024 by construction, so only check duplicates and totals
    print(f"lihtc_state_year: {len(years)} rows, duplicate keys: {years.duplicated(['state_fips', 'year']).sum()}")
    years.to_csv(INTERIM / "lihtc_state_year.csv", index=False)

    exp = expiry_table(d)
    exp.to_csv(INTERIM / "lihtc_expiry_state.csv", index=False)
    print(f"units placed 1987-2024 (rows with a usable year): {years['lihtc_units'].sum():,.0f}; "
          f"all 51-state units in the file: {d['n_units'].sum():,.0f}")
    print(f"monitored units: {exp['lihtc_units_monitored'].sum():,.0f}; past estimated end but still monitored: "
          f"{exp['lihtc_units_monitored_past_est_end'].sum():,.0f}; ending 2026-2030: "
          f"{exp['lihtc_units_est_end_by_2030'].sum():,.0f}; 2026-2035: {exp['lihtc_units_est_end_by_2035'].sum():,.0f}")
    print(years[(years.state_fips == "21") & (years.year >= 2008)].to_string(index=False))
    print(exp.sort_values("pct_monitored_est_end_by_2035", ascending=False).head(5).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
