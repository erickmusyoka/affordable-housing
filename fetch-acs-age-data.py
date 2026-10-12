# /// script
# requires-python = ">=3.10"
# dependencies = ["requests", "pandas", "python-dotenv"]
# ///
"""
Download ACS 1-year state tables needed for the Part 4 stock-flow scenarios: tenure by age of householder (B25007) and sex by age (B01001).

Setup:
    CENSUS_API_KEY in the git-ignored .env file, or exported

Run:
    uv run fetch-acs-age-data.py

Output:
    data/raw/acs1_age/acs1_age_state_<year>.csv   one file per year, one row per state, estimates only, as returned by the API
Years: 2010 to 2024 excluding 2020 (no standard 1-year release). The variable labels are checked against the API metadata before use.
"""
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ.get("CENSUS_API_KEY")
if not API_KEY:
    sys.exit("Set the CENSUS_API_KEY environment variable first.")

YEARS = [y for y in range(2010, 2025) if y != 2020]
OUT = Path("data/raw/acs1_age")
OUT.mkdir(parents=True, exist_ok=True)

TENURE = [f"B25007_{i:03d}E" for i in range(1, 22)]
SEXAGE = [f"B01001_{i:03d}E" for i in range(1, 50)]

# expected labels, asserted against the API metadata so a changed layout stops the script
TENURE_LABELS = {
    "B25007_003E": "Owner occupied:!!Householder 15 to 24", "B25007_011E": "Owner occupied:!!Householder 85 years and over",
    "B25007_013E": "Renter occupied:!!Householder 15 to 24", "B25007_021E": "Renter occupied:!!Householder 85 years and over",
}
SEXAGE_LABELS = {"B01001_006E": "Male:!!15 to 17", "B01001_025E": "Male:!!85 years and over", "B01001_030E": "Female:!!15 to 17", "B01001_049E": "Female:!!85 years and over"}


def check_labels(year, group, expected):
    meta = requests.get(f"https://api.census.gov/data/{year}/acs/acs1/groups/{group}.json", timeout=60).json()["variables"]
    for var, text in expected.items():
        if text.replace(":", "") not in meta[var]["label"].replace(":", ""):
            sys.exit(f"{year} {group} {var}: label {meta[var]['label']!r} does not contain {text!r}")


def fetch(year, variables):
    r = requests.get(f"https://api.census.gov/data/{year}/acs/acs1", params={"get": "NAME," + ",".join(variables), "for": "state:*", "key": API_KEY}, timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    rows = r.json()
    return pd.DataFrame(rows[1:], columns=rows[0])


for year in YEARS:
    check_labels(year, "B25007", TENURE_LABELS)
    check_labels(year, "B01001", SEXAGE_LABELS)
    a, b = fetch(year, TENURE), fetch(year, SEXAGE)
    if a is None or b is None:
        print(f"{year}: not available, skipped")
        continue
    df = a.merge(b.drop(columns="NAME"), on="state")
    num = [c for c in df.columns if c not in ("NAME", "state")]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    df[num] = df[num].mask(df[num] <= -100_000_000)
    df = df[df["state"] != "72"]            # drop Puerto Rico
    assert len(df) == 51, (year, len(df))
    df.to_csv(OUT / f"acs1_age_state_{year}.csv", index=False)
    print(f"{year}: {len(df)} states saved")
    time.sleep(0.3)
