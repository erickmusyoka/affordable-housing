# /// script
# requires-python = ">=3.10"
# dependencies = ["requests"]
# ///
"""
Download the Cooper Center (University of Virginia) state population projections.

No API key is needed, these are public files. Source page:
    https://www.coopercenter.org/national-population-projections

Run:
    uv run fetch-population-projections.py

Outputs:
    data/raw/state_population_projections/WCCPS_NPP_V2026_TOTAL.xlsx
    data/raw/state_population_projections/WCCPS_NPP_V2026_AGE_SEX.xlsx
    data/raw/state_population_projections/WCCPS_NPP_V2026_METHODOLOGY.docx

The projections cover 2030, 2040, and 2050 only (there is no 2035).
Citation required by the source: University of Virginia, Weldon Cooper Center for Public Service.
(2026). National and 50-State Population Projections.
"""

import sys
from pathlib import Path

import requests

OUT_DIR = Path("data/raw/state_population_projections")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE = "https://www.coopercenter.org/sites/default/files/2026-07"
FILES = [
    "WCCPS_NPP_V2026_TOTAL.xlsx",
    "WCCPS_NPP_V2026_AGE_SEX.xlsx",
    "WCCPS_NPP_V2026_METHODOLOGY.docx",
]
HEADERS = {"User-Agent": "Mozilla/5.0 (research download)"}

# xlsx and docx files are zip archives and start with these bytes
ZIP_MAGIC = b"PK\x03\x04"


def main() -> None:
    failed = False
    for name in FILES:
        try:
            resp = requests.get(f"{BASE}/{name}", headers=HEADERS, timeout=120)
        except requests.RequestException as exc:
            print(f"{name}: network error ({type(exc).__name__})")
            failed = True
            continue

        if resp.status_code != 200 or not resp.content.startswith(ZIP_MAGIC):
            print(f"{name}: HTTP {resp.status_code}, not a valid Office file, not saved")
            failed = True
            continue

        (OUT_DIR / name).write_bytes(resp.content)
        print(f"Saved {name}: {len(resp.content) / 1024:.0f} KB")

    if failed:
        sys.exit("One or more files failed to download.")


if __name__ == "__main__":
    main()
