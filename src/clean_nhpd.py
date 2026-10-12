"""
Clean the National Housing Preservation Database files into state tables.

Reads:  data/raw/nhpd/All Subsidies.xlsx, data/raw/nhpd/All Properties.xlsx
Writes: data/interim/nhpd_subsidies.csv.gz   cleaned subsidy rows (50 states plus DC)
        data/interim/nhpd_properties.csv.gz  cleaned property rows with per-property expiry fields
        data/interim/nhpd_expiry_state.csv   one row per state, property-level expiring affordability
        data/interim/nhpd_program_state.csv  one row per state and program, program-level sums only

Snapshot: NHPD as downloaded on 2026-10-06. A subsidy is Active if its end date is after the latest NHPD refresh,
so every active subsidy with a valid end date ends after the download date.

Units. NHPD gives no property-level count of assisted units because assisted units overlap between subsidies, and
property TotalUnits counts assisted AND unassisted units. So each property gets a range:
    units_lb = the largest Assisted Units among its active subsidies (capped at TotalUnits)
    units_ub = TotalUnits
Never add Assisted Units across programs. Program sums in nhpd_program_state.csv stay within one program.

Dates. Years after 2100 are placeholders (end dates run to 3033, start dates to 6439) and are treated as missing.
Public Housing, Project Based Vouchers, and Mod Rehab subsidies have no end date by design (they renew or last until
demolition) and are treated as indefinite. Any other active subsidy with a missing end date is "unknown".

Expiry definitions, per active property (all counted by end year, windows end 2030 and 2035):
    last_end   every active subsidy has a known end date and the LATEST is within the window: the property loses all
               affordability. Properties with an indefinite or unknown-end subsidy are not counted here
    first_end  the EARLIEST known end date is within the window: the property loses at least one subsidy. Upper bound
Computed from the subsidy file, not from the property file's own LatestEndDate, which can show dates already past.

NHPD LIHTC end dates are placed-in-service year + 30 (+ 15 for allocations before 1990), the same rule used for the
HUD LIHTC database, and the LIHTC records come from that database. Do not use both as separate inputs.
"""

import pandas as pd

from common import INTERIM, RAW, STATE_FIPS, check_panel, read_excel

SUBS_FILE = RAW / "nhpd" / "All Subsidies.xlsx"
PROPS_FILE = RAW / "nhpd" / "All Properties.xlsx"
INDEFINITE = {"Public Housing", "Project Based Vouchers", "Mod Rehab"}
MAX_YEAR = 2100
WINDOWS = (2030, 2035)


def year_of(series):
    """Four-digit year from a date string such as '2048-01-01 00:00:00', else missing."""
    y = pd.to_numeric(series.astype("string").str.extract(r"^(\d{4})-", expand=False), errors="coerce")
    return y.where(y <= MAX_YEAR)


def load_properties():
    cols = ["NHPDPropertyID", "State", "PropertyStatus", "TotalUnits", "ActiveSubsidies", "CensusTract", "CBSACode"]
    p = read_excel(PROPS_FILE, usecols=cols, dtype=str)
    assert not p["NHPDPropertyID"].duplicated().any()
    p["State"] = p["State"].str.strip().str.upper()
    p["state_fips"] = p["State"].map(STATE_FIPS)
    p["total_units"] = pd.to_numeric(p["TotalUnits"], errors="coerce")
    p["active_subsidies"] = pd.to_numeric(p["ActiveSubsidies"], errors="coerce")
    # 11-digit census tract FIPS (a leading zero may have been lost) and its 5-digit county; kept only when the tract starts with the property's state
    t = p["CensusTract"].astype("string").str.extract(r"^(\d{9,11})(?:\.0)?$", expand=False).str.zfill(11)
    p["tract_fips"] = t.where(t.str[:2] == p["state_fips"])
    p["county_fips"] = p["tract_fips"].str[:5]
    p["cbsa"] = p["CBSACode"].astype("string").str.extract(r"^(\d{5})", expand=False)
    print(f"county FIPS available for {p['county_fips'].notna().mean():.1%} of properties")
    return p.rename(columns={"NHPDPropertyID": "property_id", "PropertyStatus": "property_status"})[
        ["property_id", "state_fips", "property_status", "total_units", "active_subsidies", "tract_fips", "county_fips", "cbsa"]
    ]


def load_subsidies(props):
    s = read_excel(SUBS_FILE, dtype=str)
    assert not s["NHPD Subsidy ID"].duplicated().any()
    out = pd.DataFrame(
        {
            "property_id": s["NHPD Property ID"],
            "subsidy_id": s["NHPD Subsidy ID"],
            "status": s["Subsidy Status"],
            "program": s["Subsidy Name"],
            "start_year": year_of(s["Start Date"]),
            "end_year": year_of(s["End Date"]),
            "assisted_units": pd.to_numeric(s["Assisted Units"], errors="coerce").clip(lower=0),
            "inactive_reason": s["Inactive Status Description"],
        }
    )
    # state comes from the property file (the subsidy file has blanks and a trailing space on KY)
    st = props.set_index("property_id")["state_fips"]
    out["state_fips"] = out["property_id"].map(st)
    sub_state = s["State"].str.strip().str.upper().map(STATE_FIPS)
    both = out["state_fips"].notna() & sub_state.notna()
    print(f"subsidy state disagrees with property state in {(out.loc[both, 'state_fips'] != sub_state[both]).sum()} rows")
    return out[out["state_fips"].notna()].reset_index(drop=True)


def property_fields(subs, props):
    a = subs[subs["status"] == "Active"].copy()
    a["indefinite"] = a["program"].isin(INDEFINITE)
    a["unknown_end"] = ~a["indefinite"] & a["end_year"].isna()
    known = a[~a["indefinite"] & a["end_year"].notna()]
    g = a.groupby("property_id")
    f = pd.DataFrame(
        {
            "n_active_subsidies": g.size(),
            "has_indefinite": g["indefinite"].any(),
            "has_unknown_end": g["unknown_end"].any(),
            "assisted_units_max_active": g["assisted_units"].max(),
        }
    )
    f["first_end_year"] = known.groupby("property_id")["end_year"].min()
    f["last_end_year"] = known.groupby("property_id")["end_year"].max()
    p = props.merge(f, left_on="property_id", right_index=True, how="left")
    p = p[p["state_fips"].notna()]
    p["units_ub"] = p["total_units"]
    p["units_lb"] = p["assisted_units_max_active"].fillna(0).clip(upper=p["units_ub"])
    over = (p["assisted_units_max_active"] > p["units_ub"]).sum()
    print(f"properties whose largest active subsidy exceeds TotalUnits (lower bound capped): {over}")
    return p


def expiry_table(p):
    act = p[p["property_status"] == "Active"].copy()
    rows = []
    for st, g in act.groupby("state_fips"):
        row = {
            "state_fips": st,
            "nhpd_active_properties": len(g),
            "nhpd_active_units_lb": g["units_lb"].sum(),
            "nhpd_active_units_ub": g["units_ub"].sum(),
            "nhpd_props_with_indefinite": int(g["has_indefinite"].sum()),
            "nhpd_props_with_unknown_end": int(g["has_unknown_end"].sum()),
        }
        clean = g[~g["has_indefinite"].fillna(False) & ~g["has_unknown_end"].fillna(False) & g["last_end_year"].notna()]
        for y in WINDOWS:
            last = clean[clean["last_end_year"] <= y]
            first = g[g["first_end_year"] <= y]
            row[f"nhpd_last_end_by_{y}_props"] = len(last)
            row[f"nhpd_last_end_by_{y}_units_lb"] = last["units_lb"].sum()
            row[f"nhpd_last_end_by_{y}_units_ub"] = last["units_ub"].sum()
            row[f"nhpd_first_end_by_{y}_props"] = len(first)
            row[f"nhpd_first_end_by_{y}_units_lb"] = first["units_lb"].sum()
            row[f"nhpd_first_end_by_{y}_units_ub"] = first["units_ub"].sum()
        rows.append(row)
    t = pd.DataFrame(rows)
    for y in WINDOWS:
        t[f"nhpd_pct_last_end_by_{y}_ub"] = 100 * t[f"nhpd_last_end_by_{y}_units_ub"] / t["nhpd_active_units_ub"]
        t[f"nhpd_pct_first_end_by_{y}_ub"] = 100 * t[f"nhpd_first_end_by_{y}_units_ub"] / t["nhpd_active_units_ub"]
    return t.sort_values("state_fips").reset_index(drop=True)


def program_table(subs):
    a = subs[subs["status"] == "Active"]
    rows = []
    for (st, prog), g in a.groupby(["state_fips", "program"]):
        row = {"state_fips": st, "program": prog, "active_subsidies": len(g), "active_assisted_units": g["assisted_units"].sum(),
               "no_end_date_subsidies": int(g["end_year"].isna().sum())}
        for y in WINDOWS:
            row[f"assisted_units_end_by_{y}"] = g.loc[g["end_year"] <= y, "assisted_units"].sum()
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["state_fips", "program"]).reset_index(drop=True)


def main():
    props = load_properties()
    print(f"properties: {len(props)}; with a state in the 50 states plus DC: {props['state_fips'].notna().sum()}")
    subs = load_subsidies(props)
    print(f"subsidies kept: {len(subs)}; status counts: {subs['status'].value_counts().to_dict()}")
    print(f"end year missing or placeholder among active subsidies: {int(((subs['status'] == 'Active') & subs['end_year'].isna()).sum())}")

    p = property_fields(subs, props)
    subs.to_csv(INTERIM / "nhpd_subsidies.csv.gz", index=False)
    p.to_csv(INTERIM / "nhpd_properties.csv.gz", index=False)

    exp = expiry_table(p)
    assert len(exp) == 51 and exp["state_fips"].is_unique
    exp.to_csv(INTERIM / "nhpd_expiry_state.csv", index=False)
    prog = program_table(subs)
    prog.to_csv(INTERIM / "nhpd_program_state.csv", index=False)

    print(f"\nactive properties: {exp['nhpd_active_properties'].sum():,}; units lower bound {exp['nhpd_active_units_lb'].sum():,.0f}; "
          f"upper bound {exp['nhpd_active_units_ub'].sum():,.0f}")
    for y in WINDOWS:
        print(f"by {y}: last subsidy ends {exp[f'nhpd_last_end_by_{y}_units_lb'].sum():,.0f} to {exp[f'nhpd_last_end_by_{y}_units_ub'].sum():,.0f} units; "
              f"first subsidy ends {exp[f'nhpd_first_end_by_{y}_units_lb'].sum():,.0f} to {exp[f'nhpd_first_end_by_{y}_units_ub'].sum():,.0f} units")
    print(f"properties with an indefinite subsidy: {exp['nhpd_props_with_indefinite'].sum():,}; with an unknown-end subsidy: {exp['nhpd_props_with_unknown_end'].sum():,}")
    cols = ["state_fips", "nhpd_active_units_ub", "nhpd_pct_last_end_by_2035_ub", "nhpd_pct_first_end_by_2035_ub"]
    print(exp.sort_values("nhpd_pct_last_end_by_2035_ub", ascending=False)[cols].head(4).round(1).to_string(index=False))
    print(exp.sort_values("nhpd_pct_last_end_by_2035_ub")[cols].head(4).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
