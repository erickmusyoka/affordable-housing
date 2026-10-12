"""Shared helpers for the cleaning scripts. Run scripts from the project root."""

from pathlib import Path

RAW = Path("data/raw")
INTERIM = Path("data/interim")
PROCESSED = Path("data/processed")
ROOT_OUT = Path("outputs")
INTERIM.mkdir(parents=True, exist_ok=True)

# state abbreviation -> state FIPS, 50 states plus DC
STATE_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08",
    "CT": "09", "DE": "10", "DC": "11", "FL": "12", "GA": "13", "HI": "15",
    "ID": "16", "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21",
    "LA": "22", "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27",
    "MS": "28", "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33",
    "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39",
    "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46",
    "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53",
    "WV": "54", "WI": "55", "WY": "56",
}
FIPS_STATES = set(STATE_FIPS.values())


def read_excel(path, **kwargs):
    """Read any Excel file with calamine, which handles xls, xlsx, and xlsb and
    does not choke on the malformed metadata in the 2023 Income Limits file."""
    import pandas as pd

    return pd.read_excel(path, engine="calamine", **kwargs)


def check_panel(df, name, years=None, key=("state_fips", "year")):
    """Print basic integrity checks: duplicate keys, states per year, missing counts."""
    dup = df.duplicated(list(key)).sum()
    per_year = df.groupby("year").size()
    print(f"{name}: {len(df)} rows, {df['year'].nunique()} years, duplicate keys: {dup}")
    off = per_year[per_year != 51]
    print(f"  years without exactly 51 states: {off.to_dict() if len(off) else 'none'}")
    print(f"  missing values per column: {df.isna().sum()[df.isna().sum() > 0].to_dict() or 'none'}")
    assert dup == 0, "duplicate state-year keys"
