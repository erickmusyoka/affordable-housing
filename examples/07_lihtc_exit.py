"""
Example 7: when the LIHTC clock runs out, does a project leave the agency's monitoring list? One effect per state plus a cohort trend.

Run from the project root:   uv run examples/07_lihtc_exit.py
Sample: 30-year allocations placed in service 1990-1995 (observed at ages 29 to 34). Train on 1990-92, test on 1993-95, then project 1996-2005.
"""
import sys
sys.path.insert(0, "src")
import pandas as pd

import ahni
import lihtc_exit as L

projects = L.load_projects()                                     # one row per LIHTC project, with placement year and left_monitoring (1 = no longer monitored)
sample = projects[projects.pis_year.between(1990, 1995)]
print(f"{len(sample):,} projects placed 1990-95; {100 * sample.left_monitoring.mean():.0f} % have left the monitoring list")
print("share that left, by placement year:", (100 * sample.groupby("pis_year").left_monitoring.mean()).round(0).astype(int).to_dict())

ev = L.cohort_split_eval(projects, C=0.3)                        # train 1990-92, test 1993-95
print(f"\nout of sample: project AUC {ev['auc']:.2f}; state-level error {ev['state_mae']:.3f} against {ev['state_mae_national_trend']:.3f} for a national trend")

# the fit used for the projection: one effect per state plus a straight-line cohort trend, trained on all of 1990-95
model = L.StateLogit(C=0.3).fit(sample)
future = L.project_future(projects, model)                       # projects placed 1996-2005, scored as the 1993 cohort behaved at the same age (about 31)
cell = future.groupby(["state_fips", "pis_year"]).agg(p31=("p31", "first"), observed=("left_monitoring", "mean"))
future = future.join(cell.rename(columns={"observed": "obs_cell"})[["obs_cell"]], on=["state_fips", "pis_year"])
# some projects in each state-and-year cell have already left, so only the remaining share can still leave
future["remaining_share"] = ((future.p31 - future.obs_cell) / (1 - future.obs_cell)).clip(lower=0, upper=1)
monitored = future[future.left_monitoring == 0].copy()                                    # still on the list today
monitored["expected_units"] = monitored.n_units * monitored.remaining_share
total, expected = monitored.n_units.sum(), monitored.expected_units.sum()
print(f"\n{len(monitored):,} projects placed 1996-2005 are still monitored ({total:,.0f} units); expected to leave by about age 31: {expected:,.0f} units ({100 * expected / total:.0f} %)")

# the cohort test over-predicted the level, so a calibration-adjusted variant scales the predicted shares down
scale = ev["mean_obs"] / ev["mean_pred"]
adj = (monitored.n_units * (((future.loc[monitored.index, "p31"] * scale) - monitored.obs_cell) / (1 - monitored.obs_cell)).clip(lower=0, upper=1)).sum()
print(f"calibration-adjusted: {adj:,.0f} units, so the range is {adj:,.0f} to {expected:,.0f}")

by_state = monitored.groupby("state_fips").agg(units=("n_units", "sum"), leaving=("expected_units", "sum"))
share = (100 * by_state.leaving / by_state.units).sort_values()
print("lowest share leaving:", ", ".join(f"{ahni.ABBR[s]} {v:.0f}%" for s, v in share.head(3).items()), "| highest:", ", ".join(f"{ahni.ABBR[s]} {v:.0f}%" for s, v in share.tail(3)[::-1].items()))
