"""
Example 14: do our numbers agree with independent published ones? State measures against NLIHC's The Gap.

Run from the project root:   uv run examples/14_external_benchmarks.py
Reads data/interim/nlihc_gap_state.csv (NLIHC's state table, saved from https://nlihc.org/gap on 2026-10-11) and our own result tables.
"""
import pandas as pd

nlihc = pd.read_csv("data/interim/nlihc_gap_state.csv", dtype={"state_fips": str})
ours = pd.read_csv("outputs/tables/scenarios_state.csv", dtype={"state_fips": str})
ours = ours[(ours.scenario == "baseline") & (ours.year == 2023)][["state_fips", "eli_renters", "afford_units"]]
d = nlihc.merge(ours, on="state_fips")
d["ours_per100"] = 100 * d.afford_units / d.eli_renters          # affordable units per 100 ELI renters (NLIHC counts affordable AND available)

print(f"ELI renter households: ours {d.eli_renters.sum() / 1e6:.2f} million, NLIHC {d.nlihc_eli_renters.sum() / 1e6:.2f} million")
print(f"  correlation across {len(d)} states: {d.eli_renters.corr(d.nlihc_eli_renters):.3f}")
print(f"affordable homes per 100 ELI renters, rank correlation with NLIHC: {d.ours_per100.corr(d.nlihc_affordable_available_per100, method='spearman'):.2f}")
low = lambda col: ", ".join(d.nsmallest(5, col).state)
print(f"  five lowest, ours: {low('ours_per100')}")
print(f"  five lowest, NLIHC: {low('nlihc_affordable_available_per100')}")

# published expiring-unit counts against ours (same database, different snapshots and definitions)
e = pd.read_csv("data/interim/nhpd_expiry_state.csv")
print(f"\nfederally assisted units expiring in five years: NLIHC and PAHRC 374,497; ours (last subsidy ends 2026-2030) "
      f"{e.nhpd_last_end_by_2030_units_lb.sum():,.0f} to {e.nhpd_last_end_by_2030_units_ub.sum():,.0f}")
