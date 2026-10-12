# Data Sources and How Each Was Obtained

> **Note for this package.** This is the data log of the full project. This package contains only the cleaned tables (`data/interim/`, `data/processed/`) and the cleaning scripts for the six sources the analysis uses: ACS tenure by age, Cooper Center population projections, HUD CHAS, NHPD, the HUD LIHTC database, and HUD HOME rent limits. The state market panel (`data/processed/panel.csv`) is included precomputed; the scripts that build it from the other sources (ACS, BLS, BEA, fair market rents, income limits, Picture of Subsidized Households, permits, AHAR) and the references to `CLAUDE.md`, `clean_all.py`, `build_panel.py`, and the other cleaners belong to the full project and are not included. Raw downloads are not included; this log says how to obtain each.

This file is the data log for the project. For every source it records where the data came from, exactly how it was obtained, when, which period it covers, and known quirks. Someone starting from an empty `data/raw/` folder should be able to rebuild it from this file.

Rules that apply to everything below:

- Files in `data/raw/` are never edited by hand. All cleaning happens in scripts that read `data/raw/` and write to `data/interim/` or `data/processed/`
- API keys live in a git-ignored `.env` file in the project root and are never committed or printed
- "Obtained" dates come from file timestamps (the day the file was saved to disk). Where a source publishes its own release date, that is noted separately
- Page URLs below were opened and confirmed on 2026-10-06. Link text inside HUD and NHPD pages can change, so if a link has moved, search the page for the name quoted here
- Download steps for HUD sources are manual. Scripted sources are marked as such
- Read Excel files with `pd.read_excel(..., engine="calamine")` (`python-calamine` is a project dependency). The default `openpyxl` engine fails on `Section8_IncomeLimits_2023.xlsx` because of a malformed date in its metadata, and calamine also reads the `.xls` and `.xlsb` files

## Summary

| Source | Folder | Method | Period covered | Obtained |
|---|---|---|---|---|
| Census ACS 1-year | `data/raw/acs1/` | Script `fetch-acs-data.py` | 2010 to 2019, 2021 to 2024 | 2026-10-05 |
| Census Building Permits | `data/raw/bps/` | Manual or short script | Calendar years 2010 to 2025 | 2026-10-05 |
| HUD LIHTC database | `data/raw/lihtc/` | Manual | Placed in service through 2024 | 2026-10-05 |
| HUD AHAR PIT and HIC | `data/raw/ahar/` | Manual | 2007 to 2024 | 2026-10-05 |
| HUD Income Limits | `data/raw/hud_income_limits/` | Manual | Fiscal years 2010 to 2026 | 2026-10-05 |
| HUD Fair Market Rents | `data/raw/fair_market_rents/` | Manual | Fiscal years 2010 to 2026 | 2026-10-05 |
| HUD CHAS | `data/raw/chas/` | Manual | Four 5-year windows, 2005-2009 to 2018-2022 | Zips on disk by 2026-10-05; extracted 2026-10-06 |
| HUD Picture of Subsidized Households | `data/raw/picture_subsidized_households/` | Manual | 2009 to 2025 | 2026-10-05 |
| BLS unemployment and CPI via FRED | `data/raw/fred/` | Script `fetch-fred-data.py` | 2010 to 2025 | 2026-10-06 |
| BEA price parities and income | `data/raw/bea/` | Script `fetch-bea-data.py` | 2010 to 2025 (parities to 2024) | 2026-10-06 |
| National Housing Preservation Database | `data/raw/nhpd/` | Manual, free account | Snapshot at download | 2026-10-06 |
| State population projections (Cooper Center) | `data/raw/state_population_projections/` | Script `fetch-population-projections.py` | 2020 Census base; projections 2030, 2040, 2050 | 2026-10-06 |
| HUD HOME rent limits | `data/raw/home_rent_limits/` | Manual, through the browser (HUD's site blocks scripts) | Fiscal years 2019 and 2024 | 2026-10-07 |
| Census ACS 1-year, tenure and population by age | `data/raw/acs1_age/` | Script `fetch-acs-age-data.py` | 2010 to 2019, 2021 to 2024 | 2026-10-07 |

Every planned source is now on disk.

Environment variables needed for the scripts: `CENSUS_API_KEY`, `FRED_API_KEY`, `BEA_API_KEY`. The population projections script needs no key. Run scripts with `uv run <script>` from the project root.

---

## 1. Census ACS 1-year (scripted)

- **Script:** `fetch-acs-data.py`, run with `uv run fetch-acs-data.py`
- **Key:** a free Census Bureau API key, stored as `CENSUS_API_KEY` in `.env`
- **Endpoint:** `https://api.census.gov/data/<year>/acs/acs1` with `for=state:*`
- **Years:** 2010 to 2025 excluding 2020. 2025 was not available when run, so the panel is 2010 to 2019 and 2021 to 2024 (14 years, 714 rows, 51 per year). 2020 has no standard 1-year release
- **Variables (estimate and margin of error each):** median gross rent B25064_001, median household income B19013_001, gross rent as a percentage of income B25070_001 and 007 to 011, tenure B25003_001 to 003, vacancy status B25004_001 to 003, median home value B25077_001, population B01003_001
- **Outputs:** `data/raw/acs1/acs1_state_<year>.csv` (API response as returned) and `data/processed/acs1_state_panel.csv` (stacked, with derived burden, vacancy, renter share, and growth columns)
- **Reference period:** calendar-year survey estimates
- **Quirks:**
  - Puerto Rico is dropped; 50 states plus DC are kept
  - API sentinel values at or below -100,000,000 are converted to missing
  - 2021 year-over-year growth is blanked because 2020 is missing and the change would span two years
  - Small states (Wyoming, Vermont, DC) have wide margins of error; the margin columns are kept
  - Spot-check done 2026-10-07: Kentucky 2023 median gross rent on data.census.gov, table B25064, ACS 1-year, is 929 with a margin of error of 12, matching the panel exactly (the ACS 5-year product shows 933 with a margin of 5, a different estimate, not an error)

## 2. Census Building Permits Survey

- **Landing page:** https://www.census.gov/construction/bps/
- **Files:** https://www2.census.gov/econ/bps/State/ holds state-level text files named `st` + 2-digit year + 2-digit month + a letter. `c` is the current month, `y` is year-to-date
- **What to take:** the December year-to-date file for each year, `st1012y.txt` through `st2512y.txt` (16 files). Save each as `data/raw/bps/BPS_State_<year>.txt`. The site has no annual-total file, so December year-to-date is the annual figure
- **Short script that reproduces the download:**

```python
from pathlib import Path
import requests

out = Path("data/raw/bps")
out.mkdir(parents=True, exist_ok=True)
base = "https://www2.census.gov/econ/bps/State"
headers = {"User-Agent": "Mozilla/5.0 (research download)"}

for year in range(2010, 2026):
    name = f"st{year % 100:02d}12y.txt"
    resp = requests.get(f"{base}/{name}", headers=headers, timeout=60)
    resp.raise_for_status()
    (out / f"BPS_State_{year}.txt").write_bytes(resp.content)
```

- **Reference period:** calendar year. 2024 and 2025 are slightly provisional because Census revises permit counts afterward
- **Layout (checked on the 2010 and 2025 files, which match):** comma-separated, two header rows then a blank line. Columns: survey date, state FIPS, region, division, state name, then 1-unit, 2-units, 3-4 units, and 5+ units, each with buildings, units, and value. The same four groups repeat as "rep" columns; the meaning of the "rep" set is unconfirmed, so use the first set only. The other years are assumed to match; the parser should assert the header

## 3. HUD LIHTC property database

- **Page:** https://www.huduser.gov/portal/datasets/lihtc.html
- **What to take:** the full LIHTC database download, `LIHTCPUB.ZIP`. It was saved as `lihtcpub.zip` and then extracted in place into `data/raw/lihtc/`
- **Contents:** `LIHTC Data Dictionary 2024.pdf`, `LIHTCPUB.xlsx`, `LIHTCPUB_BIN.xlsx` (not yet examined), and `LIHTCPUB.accdb`. The zip has no CSV
- **Reference period:** projects placed in service 1987 to 2024 (55,345 projects, about 3.9 million units). 2025 placements are added in spring 2027. The files inside the zip are dated 2025-05-27 (Access), 2026-04-15 (dictionary), and 2026-04-30 (xlsx)
- **Quirks:**
  - `LIHTCPUB.accdb` is 79 MB, duplicates the xlsx, and is git-ignored. Everything else is committed. If you need it, it comes from the same zip
  - Projects can overlap with Section 8 and voucher-supported units. Never add LIHTC units to Picture of Subsidized Households units, and use NHPD to de-duplicate
  - `LIHTCPUB.xlsx` has 55,345 rows and 80 lowercase columns. Territories (PR, AS, VI, GU, MP) must be dropped. `yr_pis` uses 8888 (not confirmed) and 9999 (year missing) as sentinels; `yr_alloc` also uses 9999
  - `nonprog` = 1 on 20 percent of rows means the project is no longer monitored for LIHTC compliance (extended use ended, qualified contract, or other), not necessarily that it stopped being affordable. A cumulative unit sum by placement year overstates the currently monitored supply
  - `aff_yrs` is filled only when the affordability period exceeds 30 years (5 percent of rows); otherwise assume 30
  - The dictionary lists placed-in-service years 1987 to 2023, and units placed in service fall in 2023 and 2024 (96,805, 80,625, 64,269 for 2022 to 2024), which looks like a reporting lag. `LIHTCPUB_BIN.xlsx` is a building-level address list with no unit counts and is not needed

## 4. HUD AHAR: Point-in-Time and Housing Inventory Count

- **Page:** https://www.huduser.gov/portal/datasets/ahar.html
- **What to take:** the link named "Point-in-Time (PIT) and Housing Inventory Count (HIC) data files", then download **PIT Counts by State** and **HIC Counts by State**, both 2007 to latest
- **Files:** `data/raw/ahar/2007-2024-PIT-Counts-by-State.xlsb` and `2007-2024-HIC-Counts-by-State.xlsx`
- **Reference period:** one night in January of each year, 2007 to 2024. The AHAR page also lists a 2025 report; the files on disk end at 2024, so check for a newer file before relying on 2024 as the latest
- **Quirks:**
  - The PIT file is binary `.xlsb`; calamine reads it
  - Each year is a separate sheet (2007 to 2024). Column names do not include the year (`Overall Homeless`, `Sheltered Total Homeless`, `Unsheltered Homeless`, and so on), but the sheets widen from 28 columns in 2007 to 1,306 in 2024, and each has a `Total` row and, in 2024, footnotes to drop
  - Unsheltered counts are incomplete in 2021 (COVID) and also in 2024, where a footnote says 22 CoCs did not conduct an unsheltered count (AL, AR, CA, CO, FL, MD, NJ, OR, TX, VI)
  - HIC sheets have a two-row header
  - PIT undercounts people who are doubled up. Use it for validation, not as the main target

## 5. HUD Income Limits

- **Page:** https://www.huduser.gov/portal/datasets/il.html
- **What to take:** for each fiscal year 2010 to 2026, only the first Excel link, "Data for Section 8 Income Limits" (17 files; there is no combined file). Skip the Section 221(d)(3) BMIR, 235 and 236 files, the national and non-metro very low income limits, and the ESG 30 percent limits
- **Files:** `data/raw/hud_income_limits/Section8_IncomeLimits_<year>`, `.xls` for 2010 to 2014 and `.xlsx` for 2015 to 2026 (reading `.xls` needs `xlrd`)
- **Reference period:** HUD fiscal year. The effective date of each year should be recorded when the data is cleaned
- **Quirks:** data is at county or HUD metro area level (about 4,765 rows a year) and has no population column, so aggregate to state with the same-year FMR file's population column, joined on the 10-digit FIPS (state is the first two digits). Column names change across years: `median<year>` throughout, but the extremely low income limit is `l30_4` in 2010 to 2013 and `ELI_4` from 2014, with a jump in the 4-person mean from $18,827 to $24,024 between 2013 and 2014 that is a change in HUD's definition, not in incomes (explained in the next bullet). Identifier columns change three times and Puerto Rico and territories are included. The fiscal year 2024 methodology change affects the limits, so use median family income for trends. HUD median family income is built from ACS, so do not use it beside ACS income as a separate predictor
- **Why the extremely low income limit jumps in FY2014 (resolved 2026-10-07):** HUD's FY2014 Income Limits briefing material (read 2026-10-07, section D) explains it: the Consolidated Appropriations Act, 2014 redefined extremely low income as the greater of the HHS poverty guideline for the family size or 30 percent of area median (computed as 30/50ths of the very low income limit), capped at the very low income limit, with Puerto Rico and the territories excluded; HUD says all areas except 13 changed for at least one family size. The data agree: in FY2014, 3,506 of 4,767 areas (73.5 percent) have the same 4-person value of $23,850 (73.5 percent again at $24,250 in FY2015), against a largest share of 6.6 percent in FY2013, and the mean jumps from $18,827 to $24,024 (median 17,950 to 23,850). So `il_30pct_4p` (FY2010 to 2013) and `il_eli_4p` (FY2014 on) are different definitions and must stay separate; do not read the jump as a change in incomes. The FY2014 briefing is saved as `data/raw/hud_income_limits/IncomeLimits_FY14_BriefingMaterial.pdf` (195,901 bytes, 41 pages; source https://www.huduser.gov/portal/datasets/il/il14/IncomeLimitsBriefingMaterial_FY14_Rev.pdf, saved through the browser because HUD's site blocks scripted requests)

## 6. HUD Fair Market Rents

- **Page:** https://www.huduser.gov/portal/datasets/fmr.html
- **What to take:** the **County Level Data File** (Excel) for each fiscal year 2010 to 2026. For older years such as 2010, take Final Data, County Level Data File (skip Revised FMRs for 2 Areas and Proposed Data). For newer years that list several effective dates, take County Level Data under the most recent effective date. Skip Small Area FMRs, ERAP FMRs, comparison rent adjustment factors, ZCTA files, unadjusted rents, and PDFs
- **Files:** `data/raw/fair_market_rents/FMR_<year>`, `.xls` for 2010 to 2015 and `.xlsx` for 2016 to 2026. The optional 2027 file was not downloaded
- **Reference period:** HUD fiscal year, so it does not equal the calendar year. Record the effective date when cleaning
- **Quirks:** county level (about 4,765 rows a year), so aggregate to state with the file's own population column, whose name changes with the vintage (`pop2000`, `pop2010`, `pop2017`, `pop2020`, `pop2023`; find it with `^pop\d{4}$`). It is total population including Puerto Rico and territories, not renter households. Rent columns are `fmr0` to `fmr4` in 2010 to 2017 and `fmr_0` to `fmr_4` from 2018, and the FIPS column is named `FIPS`, `fips2010`, or `fips`. FMR is partly derived from ACS rents, so treat it as a policy benchmark, not a co-predictor with ACS rent

## 7. HUD CHAS (Comprehensive Housing Affordability Strategy)

- **Page:** https://www.huduser.gov/portal/datasets/cp.html
- **What to take:** set Geographic Summary Level to **States** and File Type to **csv**. Each download is a zip of all tables for one vintage. Take four vintages: 2005-2009, 2010-2014, 2015-2019, and 2018-2022. Skip 3-year vintages and the other overlapping 5-year vintages
- **Files:** each zip was extracted into `data/raw/chas/<vintage>/` (state-level table CSVs `Table1.csv` to `Table18C.csv`; 2005-2009 has no Table 6)
- **Data dictionaries:** the 2005-2009, 2010-2014, and 2015-2019 zips each included a dictionary workbook and a readme. The 2018-2022 zip did not, so it was downloaded separately from https://www.huduser.gov/portal/datasets/cp/CHAS-data-dictionary-18-22.xlsx and saved into `data/raw/chas/2018-2022/`. HUD's documentation page, https://www.huduser.gov/portal/datasets/cp/CHAS/data_doc_chas.html, lists every dictionary
- **Reference period:** five-year ACS windows. The 2018-2022 vintage was released in December 2025. The proposed rule is panel year = window end year; this is not final. The 2018-2022 window shares 2018 and 2019 with 2015-2019, so those two vintages are not independent. The exact download date was not recorded because the zips were extracted and removed; file dates inside reflect HUD's release dates
- **Dates (recovered 2026-10-07 from git history and file timestamps):** the four vintage zips were first committed on 2026-10-05, so they were downloaded on or before that day (the exact day was not recorded). `2005-2009` and `2010-2014` were extracted on 2026-10-06 at 15:09. The 2018-2022 zip had no dictionary; `CHAS-data-dictionary-18-22.xlsx` was downloaded separately on 2026-10-06 at 15:39. HUD's own release dates survive in the extracted files' timestamps: 2005-2009 tables January 2012, 2010-2014 May 2017, 2015-2019 August to September 2022, 2018-2022 December 2025
- **Quirks:**
  - Each state table has 52 rows: 50 states, DC, and Puerto Rico (`ST` = 72), which must be dropped
  - Column header case differs: `NAME` and `ST` in 2005-2009, `name` and `st` later
  - Table 8 (cost burden by income band and tenure) and Table 15C (renter affordability by income band) have identical column IDs and labels in all four vintages, checked against the dictionaries. Do not assume this for other tables
  - Dollar values are not inflation-adjusted across vintages
  - Counts are rounded by the Census Bureau (0 stays 0, 1 to 7 becomes 4, 8 or more rounds to the nearest 5), so state sums can differ slightly

## 8. HUD Picture of Subsidized Households

- **Page:** https://www.huduser.gov/portal/datasets/assthsg.html
- **What to take:** the **2009-2025** tab, then the **State** file for each year (17 files). Skip the U.S. total, county, city, CBSA, PHA, project, census tract, congressional district, and zipcode files
- **Files:** `data/raw/picture_subsidized_households/PSH_State_<year>.xlsx`, 2009 to 2025
- **Reference period:** the year of each release, 2009 to 2025
- **Quirks:**
  - Each row is a program type within a state. Use the row labeled "Summary of All HUD Programs" for state totals and do not sum program rows. All 51 states are present in every year, along with territories
  - 2025 has 752 rows because Public Housing and Housing Choice Voucher rows repeat with sub-program splits (TBV, PBV, MTW), so summing programs would double count
  - Program labels changed: 2009 to 2013 list eight programs including LIHTC; from 2014 the labels are `Project Based Section 8`, `202/PRAC`, `811/PRAC`, `S236/BMIR`, `RentSup/RAP`, and so on. The state column is `states` or `States` (`"AL Alabama"`). Reference date is 31 December in 2015 to 2025; for 2009 to 2014 it is not in the files
  - **Dictionaries downloaded 2026-10-07** from the Documentation tab of the 2009-2025 data (https://www.huduser.gov/portal/datasets/assthsg.html), saved as `data/raw/picture_subsidized_households/PSH_dictionary_2009-2014.pdf` (source `dictionary-2008.pdf`, 53,251 bytes, 7 pages) and `PSH_dictionary_2025.pdf` (source `dictionary_2025.pdf`, 236,009 bytes, 6 pages). HUD's site sits behind a bot challenge, so a script cannot fetch them; they were saved through the browser. Other years' dictionaries (2015 to 2024) are on the same tab
  - **2013 to 2014 break (resolved as a documented break):** HUD dictionaries read 2026-10-07: the definition of total_units is identical in the 2009-2014 and 2025 dictionaries (units under contract for federal subsidy and available for occupancy), so the measure did not change; the program categories did. By program the 2013 to 2014 national change is entirely in the multifamily programs: vouchers 2,349 to 2,378 thousand and public housing 1,092 to 1,083 thousand are stable, while Section 8 NC/SR + Multi-Family Other + Section 236 (1,602 thousand in 2013) become Project Based Section 8, 202/PRAC, 811/PRAC, S236/BMIR, and RentSup/RAP (1,428 thousand in 2014), a fall of 174 thousand units (11 percent) in a series that was flat before (1,608 to 1,602 thousand, 2012 to 2013) and after (1,428 to 1,432 thousand, 2014 to 2015). The dictionaries do not say whether this is reclassification or loss, so it is treated as a break in the multifamily series
  - Never add to LIHTC counts

## 9. BLS unemployment and CPI through FRED (scripted)

- **Script:** `fetch-fred-data.py`, run with `uv run fetch-fred-data.py`
- **Key:** a free FRED API key from https://fred.stlouisfed.org/docs/api/api_key.html, stored as `FRED_API_KEY` in `.env`
- **Series:** 51 state unemployment rates, named as the state abbreviation plus `UR` (for example `KYUR`, `DCUR`; seasonally adjusted), and two national CPI series: shelter `CUUR0000SAH1` and rent of primary residence `CUUR0000SEHA`
- **Method:** FRED computes the calendar-year average of the monthly values (`frequency=a`, `aggregation_method=avg`), starting 2010
- **Outputs:** `data/raw/fred/<series>.csv`, `data/processed/bls_state_unemployment.csv` (816 rows, 51 per year, 2010 to 2025), `data/processed/bls_national_cpi.csv`
- **Quirks:** FRED returns no annual average for an incomplete year, so 2026 is dropped. CPI is national and context only; there is no official state CPI

## 10. BEA Regional Price Parities and personal income (scripted)

- **Script:** `fetch-bea-data.py`, run with `uv run fetch-bea-data.py`
- **Key:** a free BEA API key from https://apps.bea.gov/API/signup/, stored as `BEA_API_KEY` in `.env`
- **Tables (line codes confirmed against the API):** `SARPP` line 1 (price parity, all items), `SARPP` line 3 (price parity, services: rents), `SAINC1` line 1 (personal income, thousands of dollars), `SAINC1` line 3 (per capita personal income). `GeoFips=STATE`, all years
- **Outputs:** `data/raw/bea/<table>_line<n>.csv` and `data/processed/bea_state_panel.csv` (51 states, 2010 to 2025)
- **Quirks:** price parities are U.S. = 100 levels, not growth rates, and are missing for 2025 because BEA had not released that year. Line 3 of `SARPP` is rents only, not general housing

## 11. National Housing Preservation Database

- **Site:** https://preservationdatabase.org. A free account is required, and you must be logged in to download
- **What to take:** on the data page, open the dropdown beside "Download Complete Database", choose the file, and click **Download Complete Database** (not "Export Filtered Grid", which exports only the rows on screen). Leave the Subsidy Status and Program filters blank. Two downloads:
  - **All Subsidies (xlsx)**, saved as `data/raw/nhpd/All Subsidies.xlsx`
  - **All Properties (xlsx)**, saved as `data/raw/nhpd/All Properties.xlsx`
  - The dropdown also offers Active and Inconclusive, Active only, Inactive only, and per-state files. They were not used because the two above contain all of that
- **Reference period:** a snapshot of the database at download, 2026-10-06. The site shows no version date that was recorded
- **Contents:** All Subsidies has 210,509 rows, one per subsidy. All Properties has 122,174 rows and 271 columns, one per property
  - Summing assisted units over subsidies double counts (7.29 million units over active subsidies). Property total units for active properties (5.50 million) is an upper bound on assisted units, not a count of them
  - State has 59 values including territories, odd codes, blanks, and `KY ` with a trailing space
  - Some dates are placeholders (end dates to 3033, start dates to 6439); treat years after 2100 as missing
  - Public Housing, Project Based Vouchers, and Mod Rehab subsidies have no end dates
  - The two files are large and slow to parse (about four minutes to read both with pandas), so read once and save an interim file
  - The data dictionary is saved as `data/raw/nhpd/Data-Dictionary-August-2026.pdf` (99 pages, from the site's Documentation tab, downloaded 2026-10-06)
  - Property `Total Units` counts assisted and unassisted units, so it is not an assisted-unit count; NHPD gives no property-level assisted count because assisted units overlap between subsidies
  - LIHTC end dates are computed as placed-in-service year + 30 (or + 15 for allocations before 1990), not actual restriction end dates, and LIHTC records come from the same HUD database as Section 3

## 12. State population projections (scripted)

- **Page:** https://www.coopercenter.org/national-population-projections (Weldon Cooper Center, University of Virginia, "National Population Projections")
- **Script:** `fetch-population-projections.py`, run with `uv run fetch-population-projections.py`. No API key is needed because the files are public. It checks that each download is a valid Office file before saving
- **What it takes:** the Total Population and Age and Sex Excel files, and the methodology document. Direct links, confirmed on the page on 2026-10-06:
  - https://www.coopercenter.org/sites/default/files/2026-07/WCCPS_NPP_V2026_TOTAL.xlsx
  - https://www.coopercenter.org/sites/default/files/2026-07/WCCPS_NPP_V2026_AGE_SEX.xlsx
  - https://www.coopercenter.org/sites/default/files/2026-07/WCCPS_NPP_V2026_METHODOLOGY.docx
- **Saved to:** `data/raw/state_population_projections/` (TOTAL 12 KB, AGE_SEX 170 KB, METHODOLOGY 871 KB). The files say they were released on July 15, 2026
- **Reference period:** projections for 2030, 2040, and 2050, released July 2026, benchmarked to the 2020 Decennial Census and the Census Bureau's 2025 population estimates. Covers the 50 states and DC. **There is no 2035 projection**, so the 2035 scenario must interpolate between 2030 and 2040
- **Citation required by the source:** University of Virginia, Weldon Cooper Center for Public Service. (2026). National and 50-State Population Projections
- **Layout:** `WCCPS_NPP_V2026_TOTAL.xlsx` has one sheet with FIPS, name, and population for 2020, 2030, 2040, 2050 (52 rows: the U.S., 50 states, DC). `WCCPS_NPP_V2026_AGE_SEX.xlsx` has one sheet per year (2020, 2030, 2040, 2050), with Total, Male, and Female rows per geography across 18 five-year age bands (0 to 4 up to 85+). Both have three title rows above the header row. The 2020 values are Census counts; projected values are unrounded decimals. The U.S. rows (FIPS `00`) must be dropped. The age detail is based on the 2020 Census; the methodology document has not been read yet
- **Methodology (document read 2026-10-07):** inputs are the 2010 and 2020 Censuses (total and by age; 2020 also by sex) and the Census Bureau's July 1, 2025 estimate (**total population only**). The 2030 total is a trend extrapolation: each state's 2025 estimate grown at the average of its 2010-2025 and 2020-2025 exponential growth rates. Age detail comes from the Hamilton-Perry cohort-change method using 2010-2020 cohort change ratios (which embed mortality and migration of that decade) applied to the 2020 Census cohorts; under-65 groups are rescaled to the state total and the 65-and-older groups are held at their cohort-aged levels. 2040 and 2050 repeat the cohort method with a national control total and state growth set to the average of the exponential and cohort-method growth, raked to the national total. Sex follows 2020 age-specific ratios. There is no explicit migration or fertility assumption beyond the 2010-2020 ratios, and no uncertainty range

## 13. Census ACS 1-year tenure by age of householder and population by age (scripted)

- **Why:** the Part 4 stock-flow scenarios need renter households by age of householder to apply to the Cooper Center population by age. The original ACS pull has only totals
- **Tables:** B25007 (tenure by age of householder, estimates 001 to 021) and B01001 (sex by age, estimates 001 to 049), ACS 1-year, `for=state:*`. Endpoint `https://api.census.gov/data/<year>/acs/acs1`
- **Script:** `fetch-acs-age-data.py`, run with `uv run fetch-acs-age-data.py`. Needs `CENSUS_API_KEY` in `.env`. The API allows 50 variables per call, so each year takes two calls that are merged. The script checks the label of the first and last row of each age block against the API metadata for every year and stops if the layout differs (2010 labels have no colons, which the check allows for)
- **Saved to:** `data/raw/acs1_age/acs1_age_state_<year>.csv`, 2010 to 2019 and 2021 to 2024 (2020 has no standard 1-year release), 51 rows each after dropping Puerto Rico. Estimates only, no margins of error. Obtained 2026-10-07
- **Cleaner:** `src/clean_acs_age.py` writes `data/interim/acs_age_tenure_state_year.csv` (714 rows) with renter households, owner households, and population in eight bands: 15-24, 25-34, 35-44, 45-54, 55-64, 65-74, 75-84, 85+. B25007's 55-59 and 60-64 rows are combined, as are the matching B01001 rows. The cleaner asserts that the age rows add up to the table's own renter and owner totals and that those equal ACS B25003 (`hh_renter`, `hh_owner` in the panel) in all 714 state-years (they do)
- **Quirks:** the 1-year estimates for small states are noisy; the scenarios pool 2022 to 2024 for the base and 2014 to 2016 for the historical comparison. Population by age here (ACS) is not the same as the Cooper Center age detail (2020 Census base), which is why the scenarios apply Cooper growth *factors* to ACS levels instead of mixing the two as levels

## 14. HUD HOME rent limits (manual, through the browser)

- **Page:** https://www.huduser.gov/portal/datasets/HOME-Rent-limits.html (HUD USER, "HUD HOME Rent Limits"; data are available from FY1998, annual workbooks "All Rent and Income Limits" from FY2012 on). Linked from HUD Exchange's CPD Income and Rent Limits page. The site sits behind a bot challenge, so a script cannot fetch the files; they were saved through the browser on 2026-10-07
- **Files saved:** `data/raw/home_rent_limits/HOME_Limits_Rents_2024.xlsx` (850,480 bytes, source `.../home-datasets/files/HOME_Limits_Rents_2024.xlsx`) and `HOME_AllLimits_2019.xlsx` (1,718,572 bytes, source `.../home-datasets/files/HOME_AllLimits_2019.xlsx`). Other years (FY2012 to FY2026) are on the same page; only these two are held
- **Rule (24 CFR 92.252, from the page):** the High HOME rent is the lesser of the fair market rent or the rent affordable at 65 percent of area median income; the Low HOME rent is the rent affordable at 50 percent, also capped by the fair market rent. Rents are given by bedroom count 0 to 6; the cleaner keeps 0 to 4
- **Layout:** one row per HUD area (4,764 in FY2024, 4,769 in FY2019), no duplicate area codes. Key columns: `fips2010` (9-digit area code with the leading zero dropped), `state`, `county` (numeric codes), `fmr_b`, `lowrent_b`, `highrent_b`, `homerent50_b`, `homerent65_b`. Column case differs between the two files (`HomeRent50_` in 2019, `homerent50_` in 2024). The FY2019 file also carries the 30, 50, 60, and 80 percent income limits by household size; the FY2024 file does not
- **Cleaning (`src/clean_home_rent_limits.py`):** areas are averaged to the county (New England areas are towns), territories are dropped, and the cleaner asserts that the High HOME rent never exceeds the FMR and the Low HOME rent never exceeds the High HOME rent. Output `data/interim/home_rent_limits_county.csv` (3,143 counties for FY2019, 3,144 for FY2024), plus `rent60_2br` = 1.2 times the 50 percent HOME rent (the rent affordable at 60 percent of median, the LIHTC level) and `fmr_to_rent60_2br` = FMR divided by that rent. FMR exceeds the 60 percent rent in 17 percent of counties in FY2019 and 26 percent in FY2024
- **Used for:** the market-rent-versus-restricted-rent test in `notebooks/05_preservation/03_rent_gap`. HOME income limits are the Section 8 income limits already held (HOME uses them), so they are not a separate source

## Cleaned tables

The raw files above are turned into state-year tables in `data/interim/` by the scripts in `src/` (one cleaner per source, run all with `uv run src/clean_all.py`). The tables, grains, and every cleaning decision are in `CLAUDE.md` Section 13. The cleaners read `data/raw/` only and never change it. Mapping:

| Raw source | Interim table or tables |
|---|---|
| Census Building Permits Survey | `bps_state_year.csv` |
| HUD AHAR PIT and HIC | `ahar_pit_state_year.csv`, `ahar_hic_state_year.csv` |
| HUD Fair Market Rents | `fmr_county_year.csv`, `fmr_state_year.csv` |
| HUD Income Limits (weights from the FMR county table) | `income_limits_state_year.csv` |
| HUD Picture of Subsidized Households | `psh_state_year.csv` |
| HUD LIHTC database | `lihtc_state_year.csv`, `lihtc_expiry_state.csv` |
| HUD CHAS | `chas_state_vintage.csv` |
| National Housing Preservation Database | `nhpd_subsidies.csv.gz`, `nhpd_properties.csv.gz`, `nhpd_expiry_state.csv`, `nhpd_program_state.csv` |
| Cooper Center population projections | `population_projection_state_year.csv`, `population_projection_age_sex.csv` |
| Census ACS tenure and population by age | `acs_age_tenure_state_year.csv` |
| HUD HOME rent limits | `home_rent_limits_county.csv` |

The ACS, FRED, and BEA scripts already write state-year tables to `data/processed/`, so they have no separate cleaner.

The cleaned tables and the ACS, FRED, and BEA tables are merged by `src/build_panel.py` into `data/processed/panel.csv` (51 states by 2010 to 2025, with a column dictionary in `panel_dictionary.csv` that records the source, unit, and reference period of every column) and `state_snapshot.csv` for the sources that are not annual.

---

## Large files and git

The data folder is committed so that manual downloads survive a change of machine. The two NHPD files (87 MB and 38 MB) are committed even though the larger one is over GitHub's 50 MB warning threshold. `data/raw/lihtc/LIHTCPUB.accdb` (79 MB, duplicates the xlsx) is git-ignored, along with `.env` and `.DS_Store`. The cleaned `data/interim/` tables (9 MB) are committed too, so they can be used without rerunning the pipeline.
