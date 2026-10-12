# Affordable Rental Housing: Units at Stake by 2035

> How many affordable rental units in the United States could be lost, or newly needed, by 2035; in which states; and what it would take to hold the line.

| | |
|---|---|
| **Answer** | **0.90 to 1.03 million units** are at stake by 2035 (885,000 to 1.05 million counting sampling error); under baseline growth, 62 to 67 percent of that is existing housing that could be lost rather than new demand |
| **Coverage** | 50 states and DC; data from 2010 to 2025 (scenarios to 2035) |
| **Methods** | Logistic regression, penalized logistic regression with state effects, cohort life tables, demographic projection, composite index (and, in the full study, fixed-effects panels, a Bayesian state-space model, and machine learning that did not win) |
| **Stack** | Python 3.12+, [uv](https://docs.astral.sh/uv/), pandas, statsmodels, linearmodels, scikit-learn |
| **Start with** | [`REPORT_SHORT.md`](REPORT_SHORT.md): the full write-up, with code, output, and figures |

**Contents:** [Why](#why-this-project) | [Result](#headline-result) | [Quick start](#quick-start) | [Data](#data) | [Exploratory analysis](#exploratory-analysis) | [Models](#models) | [Results](#results-in-pictures) | [Try it](#try-it-in-five-minutes) | [Layout](#repository-layout) | [Checks](#reproducibility-and-checks) | [Limits](#limits) | [Next steps](#next-steps)

## Why this project

Affordable rental housing gets scarcer through two forces. **Demand grows:** more extremely low income renter households need a home. **Supply leaks:** subsidies and rules that keep existing units affordable run out, and nobody is required to renew them. Most public counts stop at "this many units have an expiration date by 2030 or 2035". That says what is scheduled, not what will actually be lost.

This project asks the harder question, using what has already happened. When a federal subsidy ended in the past, which properties got a new one? When LIHTC projects reached the end of their 30-year clock, which dropped off the state agency's list? It learns those patterns from 12,822 past lapses and 7,315 LIHTC projects, projects them onto today's expiring stock, adds the growth in need from a demographic projection, and totals the result by state.

Two commitments shape everything else. **Honesty about noise:** the rent-burden series is a survey sample, and about three quarters of its year-to-year movement is sampling error, so every model is tested against "next year equals this year" on data it has not seen. **Honesty about causation:** the models say what is likely to happen, where, and how big; they do not say which intervention works. The planning arithmetic at the end is arithmetic, with no costs.

## Headline result

<p align="center"><img src="outputs/figures/09_report/91_national_pieces.png" alt="Figure 1. The four pieces of the national total, in thousand units by 2035 (bar" width="760"></p>

<p align="center"><em>Figure 1. The four pieces of the national total, in thousand units by 2035 (bar: low to high estimate).</em></p>


| Piece | Method | Units at stake by 2035 |
|---|---|---|
| Growth in need | Demographic projection with age-specific tenure rates | about 341,000 extra affordable units to hold today's ratio |
| HOME and HUD-program subsidies lapsing | Logistic regression on property size and program | 295,000 to 353,000 units not rescued (an upper bound) |
| LIHTC projects leaving the monitoring list | Ridge-penalized logistic regression with state fixed effects | 236,000 to 297,000 units |
| HUD-insured mortgages ending early | Cohort life table | 28,000 to 42,000 units |
| **Total** | | **0.90 to 1.03 million** |

By units, Texas, Florida, and North Carolina lead (growth-driven); California, New York, Pennsylvania, and Ohio lead where subsidies lapse; Maryland, Michigan, and Virginia where LIHTC projects leave monitoring. Per extremely low income renter, Idaho, South Dakota, and Utah are highest.

<p align="center"><img src="outputs/figures/06_combined/61_units_at_stake.png" alt="Figure 2. Units at stake by 2035 for the 15 largest states, by source." width="760"></p>

<p align="center"><em>Figure 2. Units at stake by 2035 for the 15 largest states, by source.</em></p>


## Quick start

**Requirements:** Python 3.12 or newer and [uv](https://docs.astral.sh/uv/getting-started/installation/) (`brew install uv`, or `curl -LsSf https://astral.sh/uv/install.sh | sh`). About 15 MB of data are included; no downloads or API keys are needed.

```bash
cd SUBMISSION            # the folder that contains this README
uv sync                  # creates .venv and installs the locked dependencies
uv run examples/04_need_index.py     # first run: prints the highest-need states and how stable the ranking is
```

**Run everything** (all eighteen notebooks, rebuilt from the cleaned tables in `data/`):

```bash
uv run src/run_short_report.py        # about 14 minutes; one line per notebook, "ok" or the error
uv run src/run_short_report.py all    # also re-runs the six cleaning scripts (needs the raw downloads, see DATA.md)
```

**Run one notebook** or **one example:**

```bash
uv run src/notebook_tools.py notebooks/05_preservation/01_preservation_model.py   # writes the executed .ipynb next to the .py
uv run examples/06_rescue_model.py
```

Run one copy at a time: the notebooks write to `outputs/`. Edit the `.py` file, not the `.ipynb`; the `.ipynb` is regenerated from it on every run.


## How it fits together

<p align="center"><img src="outputs/figures/01_eda/flow-diagram.png" alt="Figure 3. From raw downloads to units at stake by state." width="360"></p>

<p align="center"><em>Figure 3. From raw downloads to units at stake by state.</em></p>



## Data

Every source has a cleaning script that reads `data/raw/` and writes a plain table to `data/interim/` or `data/processed/`, with the reference period of each source preserved. [`DATA.md`](DATA.md) records where each file came from, when, and its quirks.

| Source | Used for | Period |
|---|---|---|
| Census ACS 1-year (rent, income, burden, households, vacancy) | The state panel and the rent-burden target | 2010 to 2024, no 2020 release |
| Census ACS tenure and population by age | Renter rates by age for the demand scenarios | 2010 to 2024 |
| Weldon Cooper Center population projections | Population by age to 2035 (2035 interpolated) | 2020 to 2050 |
| HUD CHAS | Extremely low income renters; rent-affordable units | Four 5-year windows, 2005-09 to 2018-22 |
| National Housing Preservation Database (NHPD) | Subsidy history and expiry dates; the insured-mortgage pathway | Snapshot, October 2026 |
| HUD LIHTC database | LIHTC projects, placement year, monitoring status | Placed in service 1987 to 2024 |
| HUD HOME rent limits and fair market rents | County gap between market and restricted rent (tested, no effect) | FY2019 and FY2024 |
| NLIHC The Gap, state table | Independent benchmark for our state measures (never a model input) | Table as displayed 2026-10-11 |
| FHFA county house price index | Price appreciation (tested, no effect) | 1975 to 2025 |


The cleaned tables share one key (`state_fips`, a two-character string such as `"06"` for California), so any table joins to any other.

## Exploratory analysis

Before choosing models, five notebooks (`notebooks/01_eda/`) check what the data can support. Four findings drive the modeling.

1. **Most of the burden is which state you are in** (73.5 percent of the variance), 12.0 percent is common year shocks, and 14.4 percent is left over, a mix of real movement and noise.
2. **The survey noise is large in small states.** The median 90 percent margin of error is 1.07 points, about 2.8 in Wyoming, Vermont, and Delaware and 0.5 in California and Texas; sampling error alone explains about 76 percent of the variance of year-to-year changes.
3. **That sets a ceiling.** An oracle that knew every state's true burden would still miss the next published value by 0.61 points against 1.09 for "next year equals this year", so no model can gain more than 44 percent over that benchmark.
4. **Rankings are stable, changes are not.** States keep their place (rank correlation 0.82 between 2010-12 and 2022-24); the median burden fell to 22.1 percent by 2019, jumped to 25.1 in 2021, and sits at 24.4 in 2024, driven by rent growth.

<p align="center"><img src="outputs/figures/01_eda/02_three_states_noise_bands.png" alt="Figure 4. Severe burden with its 90 percent margin of error shaded" width="820"></p>

<p align="center"><em>Figure 4. Severe burden with its 90 percent margin of error shaded: in Wyoming the band is as wide as the swings; in California it is a sliver.</em></p>

<p align="center"><img src="outputs/figures/01_eda/03_indexed_rent_income.png" alt="Figure 5. Rent and household income in the median state, 2010 = 100" width="640"></p>

<p align="center"><em>Figure 5. Rent and household income in the median state, 2010 = 100: both rose about 62 percent; rent outgrew income in 38 of 51 states.</em></p>


## Models

Short labels such as "rescue model" name a use of a standard method. The report's Section 2 is the full register of what was tried, with evidence; this table is the part this package contains.

| Model (standard name) | What it does | Result |
|---|---|---|
| Composite indicator (z-score aggregation) | Scores every state on seven need measures, with rank intervals | Florida, Nevada, California, Arizona, Oregon lead; the top eight ranks are stable |
| Demographic projection with age-specific tenure rates | Scales renter households by the projected population of each age group; three scenarios | ELI renters 10.9 million (2023) to 11.6 million (2035) in the baseline; the direction depends on whether renting keeps falling |
| Logistic regression | Chance that a property whose subsidy ended is re-subsidized, from size and program | AUC 0.82 on a time split; one variable (size) does the work |
| Ridge-penalized logistic regression with state fixed effects and a cohort trend | Chance a LIHTC project leaves the monitoring list | State predicts it (AUC 0.79), the project does not (0.52 across states) |
| Cohort life table | Insured mortgages that end before their scheduled date | 28,000 to 42,000 units the date-based list misses |
| Logistic regression with state effects (tests) | Does a market-rent gap, house price growth, or nonprofit sponsor explain outcomes? | No effect, within states or out of sample |
| Bootstrap with random outcomes (resample, refit, rescore, draw each outcome) | How uncertain is the total, and what does it depend on? | 885,000 to 1.05 million with sampling error; the growth scenario moves it -35 to +29 percent |
| Hindcast and real-time recalibration tests | Are the totals right, and can a model keep up with the post-2021 rise in rescues? | Totals right to about 1 percent; state differences in rescue shares not explained; no real-time method anticipates the rise |

**Tried in the full study and not included here** (each has a verdict and evidence in the report): fixed-effects panels and the Mundlak test, a Bayesian state-space forecast (0.89 of the persistence error at one year), shrinkage and dynamic-panel forecasts, Elastic Net, random forest, gradient boosting, and SHAP. No machine learning model beat the simple econometric ones in any of 27 cases.

<p align="center"><img src="outputs/figures/09_report/92_one_year_scorecard.png" alt="Figure 6. Every model tried for the one-year severe-burden forecast, against "next year equals this year" (below 1 is better). Econometric models in blue, machine learning in orange." width="760"></p>

<p align="center"><em>Figure 6. Every model tried for the one-year severe-burden forecast, against "next year equals this year" (below 1 is better). Econometric models in blue, machine learning in orange.</em></p>


## Results in pictures

<p align="center"><img src="outputs/figures/01_eda/04_ahni_tile_map.png" alt="Figure 7. Need index by state, 2022-24 (darker means more need). The top eight ranks hold within two places in 99.8 percent of resamples of the survey error." width="720"></p>

<p align="center"><em>Figure 7. Need index by state, 2022-24 (darker means more need). The top eight ranks hold within two places in 99.8 percent of resamples of the survey error.</em></p>

<p align="center"><img src="outputs/figures/04_scenarios/41_national_scenarios.png" alt="Figure 8. Extremely low income renter households under three scenarios" width="640"></p>

<p align="center"><em>Figure 8. Extremely low income renter households under three scenarios: the spread comes from assumptions about renting, not demographics.</em></p>

<p align="center"><img src="outputs/figures/05_preservation/51_rescue_by_size.png" alt="Figure 9. Share of properties re-subsidized within three years, by size" width="640"></p>

<p align="center"><em>Figure 9. Share of properties re-subsidized within three years, by size: under 1 percent at ten units or fewer, 28 percent at 200 or more.</em></p>

<p align="center"><img src="outputs/figures/05_preservation/52_exit_by_state.png" alt="Figure 10. LIHTC projects placed 1993 to 1995 that have left monitoring, by state" width="640"></p>

<p align="center"><em>Figure 10. LIHTC projects placed 1993 to 1995 that have left monitoring, by state: from 0 percent (Connecticut) to 99 percent (Maryland).</em></p>

<p align="center"><img src="outputs/figures/08_actions/81_lever_grid.png" alt="Figure 11. Affordable units per 100 ELI renters in 2035 under combinations of preservation and new production (2023" width="640"></p>

<p align="center"><em>Figure 11. Affordable units per 100 ELI renters in 2035 under combinations of preservation and new production (2023: 54). Planning arithmetic, no costs.</em></p>


## Try it in five minutes

Fourteen short scripts in [`examples/`](examples/) run one model each, print the numbers quoted in the report, and take seconds to a minute. Three to start with:

```bash
uv run examples/04_need_index.py       # builds the need index and tests its stability
uv run examples/06_rescue_model.py     # which expiring properties get a new subsidy
uv run examples/09_units_at_stake.py   # adds the four pieces by state
```

For example, the last one prints (the first lines):

```text
national total, units by 2035 (low to high):
  growth need (baseline)                    341,444 to    341,444
  HOME and HUD subsidies not rescued        294,898 to    352,711
  LIHTC leaving monitoring                  235,693 to    296,639
  insured mortgages ending early             27,810 to     42,181
  total                                     899,845 to  1,032,975
```

To work with the results directly:

```python
import pandas as pd

t = pd.read_csv("outputs/tables/units_at_stake_state.csv")
print(t.sort_values("stake_mid", ascending=False)[["state", "stake_low", "stake_high", "lead"]].head(5).round(0))
#   state  stake_low  stake_high    lead
#      TX   101555.0    109084.0  growth
#      FL    72588.0     79762.0  growth
#      NC    48373.0     53190.0  growth
#      NY    37511.0     47484.0    nhpd
#      CA    38496.0     46094.0    nhpd
```

Or load a cleaned table and the model code from `src/`:

```python
import sys; sys.path.insert(0, "src")
import preservation as P

events = P.build_events()          # one row per property whose last subsidy ended, 2011 to 2023
print(len(events), int(events.resub_3y.sum()))     # 12822 events, 189 re-subsidized within three years
```


## Notebooks at a glance

| Notebook | Question | Result |
    |---|---|---|
    | `01_eda/01_coverage` | Which data exist for which states and years? | Complete for all 51 states except 2020, 2025, and the CHAS snapshots |
    | `01_eda/02_signal_vs_noise` | How much of the year-to-year movement in rent burden is real? | About 76 percent of changes is sampling error; this limits any forecast |
    | `01_eda/03_trends` | How did rent, income, and burden move 2010 to 2024? | Rent and income rose about 62 percent in the median state; burden fell to 2019, jumped in 2021 |
    | `01_eda/04_need_today_ahni` | Where is need greatest today? | Need index with rank intervals; Florida, Nevada, California, Arizona, Oregon lead |
    | `01_eda/05_relationships` | What moves with the burden and with rent growth? | Burden change is mostly mean reversion; rent growth has modest predictors |
    | `04_scenarios/01_stock_flow` | How many more households will need affordable housing? | Scenarios, not forecasts: 10.9 million ELI renters (2023) to 11.6 million (2035) in the baseline |
    | `05_preservation/01_preservation_model` | Which lapsing HOME and HUD subsidies are rescued? | AUC 0.82 on a time split; forest and boosting do no better |
    | `05_preservation/02_lihtc_exit` | What happens when the LIHTC clock runs out? | Driven by the state, not the project; AUC 0.80 on a cohort split |
    | `05_preservation/03_rent_gap`, `06_price_growth` | Does a market incentive, house price growth, or sponsor type explain it? | No (negative results) |
    | `05_preservation/04_mortgage_exit` | What does the date-based expiry list miss? | 28,000 to 42,000 units of insured mortgages that end early |
    | `05_preservation/07_state_hindcast`, `08_rescue_drift` | Does the rescue model get the totals right, and what if the post-2021 rescue rate persists? | Totals right to about 1 percent; state rescue shares not explained; no real-time method anticipates the rise; if it persists the total is about 5 percent lower |
| `06_combined/01_units_at_stake` | How big is the task, by state? | 0.90 to 1.03 million units by 2035 |
| `06_combined/02_uncertainty_and_sensitivity` | How sure are we of the total? | 885,000 to 1.05 million with sampling error; the growth scenario moves it -35 to +29 percent |
| `06_combined/03_external_benchmarks` | Do we agree with independent figures? | State measures agree with NLIHC (r 0.995 and 0.75); published expiring counts fall inside or near ours |
    | `07_homelessness/01_ahar_context` | Context from the homelessness count | Descriptive only; annual changes are not explained by rents |
    | `08_actions/01_what_can_be_done` | What could be done? | Planning arithmetic: targeting trade-off, capacity, and levers |
    
## Repository layout

```text
REPORT_SHORT.md        the full write-up: code, output, and figures (also exported as report_short.pdf)
README.md              this file
DATA.md                where each source came from, its reference period, its quirks
examples/              fourteen runnable scripts, one per model
notebooks/             eighteen notebooks: .py source (edit this) and executed .ipynb
  01_eda/  04_scenarios/  05_preservation/  06_combined/  07_homelessness/  08_actions/
src/                   model code (ahni, scenarios, preservation, lihtc_exit, mortgage_exit,
                       combined, actions), helpers, six cleaners, and run_short_report.py
data/interim/          cleaned tables (the notebooks read these)
data/processed/        the state-year panel and its dictionary
data/raw/chas/         two small raw HUD tables read directly by one notebook
outputs/tables/        result tables written by the notebooks
outputs/figures/       figures, grouped by part
fetch-*.py             API download scripts (need a free Census key)
pyproject.toml, uv.lock   locked dependencies
```

Notebook numbers keep the full project's numbering, so the gaps (no `02_econometrics` or `03_ml_bakeoff`) are parts of the full study that this package leaves out; the Models section above says which.


## What this package leaves out, and why

| Not included | What it was | Why it is out |
    |---|---|---|
    | State-level forecasts (Bayesian state-space, fixed-effects and shrinkage models) | One to three year forecasts of state rent burden and rent growth | They work modestly (0.89 of "same as last year" at one year; noise caps the gain) and the combined result does not use them |
    | Machine-learning bake-off and SHAP for state forecasts | Elastic Net, random forest, boosting against the econometric models | None of 27 cases beat the simple models; the rescue notebook keeps the comparison at property level |
    | Explanatory econometrics | Fixed effects, dynamics, cross-state dependence | Context for the full study, not needed for the combined result |
    | Richer-feature check and policy-flag profiles | Tract context, location, extra NHPD fields; five yes/no flags by state | No gain, and superseded by the combined estimate |
    | Raw downloads (about 280 MB) | NHPD, LIHTC, CHAS, rent limits, and others | Size; cleaned tables are included and `DATA.md` says how to re-download |
    
## Reproducibility and checks

- **Raw data is immutable.** Cleaning scripts only read `data/raw/` and write elsewhere; every script can be rerun.
- **Uncertainty is quantified, not asserted.** The headline range is also run through 1,000 resamples of the training data; the growth assumption moves the total by -35 to +29 percent, the largest modeling choice (the rescue rate staying at its post-2021 level) by 5 percent and the rest by 3 percent or less (`notebooks/06_combined/02_uncertainty_and_sensitivity`).
- **External benchmarks.** Our state measures agree with NLIHC's independently built Gap table (ELI renter households r = 0.995; homes per 100 rank r = 0.75) and published expiring-unit counts fall inside or near ours (`notebooks/06_combined/03_external_benchmarks`).
- **Every example reproduces the number it is quoted for** in the report (for example the LIHTC projection of 235,693 to 296,639 units and the total of 899,845 to 1,032,975).
- **Forecast tests never use random splits.** Models are fitted only on what was known at the time, judged against persistence, and every result is also shown without 2021 and without 2021 to 2023.
- **Leakage was hunted and fixed:** a later subsidy leaked into the first rescue features (AUC 0.999), extra NHPD fields turned out to record the outcome (AUC 0.94 from "is it filled in" flags), and two rent-gap results were artifacts of a poor county match and of mixing two rent-limit vintages. The report describes each.
- **Seeds are fixed** for resampling and sampling steps, so reruns reproduce the tables to rounding noise.


## Limits

A table of the questions a reviewer could raise, what was done about each, and what remains is in the report (Section 8, Threats to validity).

- The scenarios are conditional on a population projection that extends 2010-2025 growth trends and on renter behavior; they are not forecasts, and 2035 is interpolated.
- The rescue model is conditional on a lapse, so its total is an upper bound on loss; it has 58 rescued events in training and covers HOME and HUD-program subsidies (41 percent of expiring units). A 2019-2023 hindcast shows its totals are right in aggregate but that it does not explain differences in rescue shares between states, and the rescue rate rose after 2021 (no real-time model anticipated it; if the higher rate persists the total is about 5 percent lower).
- The LIHTC outcome is leaving the monitoring list, not losing affordability, and the state effect mixes policy and reporting practice.
- The combined total is an order of magnitude: pieces can overlap, and sampling error is small next to the choice of growth scenario, which moves the total by about a third either way.
- Everything here is association. The planning arithmetic has no costs and assumes an action succeeds.

## Next steps

1. Dated LIHTC exits and extended-use periods from state agencies, to support survival (hazard) models.
2. As-of-event owner, manager, and inspection history, to test for-profit against nonprofit ownership without survivorship bias.
3. Causal designs where state policy varies over time (for example difference-in-differences on qualified-contract rules), which needs dated policy data.
4. An availability-adjusted affordable-unit gap (CHAS Table 14B) to match the standard national measure.
5. A refresh with 2025 data, and cost data for an investment index kept separate from need.

Data providers: U.S. Census Bureau, HUD, the National Low Income Housing Coalition and Public and Affordable Housing Research Corporation (NHPD), the Weldon Cooper Center, BLS, BEA, and FHFA.
