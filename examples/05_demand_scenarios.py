"""
Example 5: how many extremely low income (ELI) renter households will there be in 2035? Three scenarios.

Run from the project root:   uv run examples/05_demand_scenarios.py
Renter households by age of householder, grown with the projected population of the same age; ELI share from CHAS.
"""
import sys
sys.path.insert(0, "src")
import pandas as pd

import scenarios as S

acs, cooper, chas, panel = S.load_inputs()
drift = S.historical_drift(acs, chas)                 # how renter rates and the ELI share actually drifted, 2015 to 2023, state by state
print("share of states where renter rates fell:", f"{100 * (drift.g_rate < 0).mean():.0f} %", "| ELI share fell:", f"{100 * (drift.g_eli < 0).mean():.0f} %")

result = S.project(acs, cooper, chas, years=(2030, 2035), drift=drift)
national = result.groupby(["scenario", "year"])[["renter_hh", "eli_renters"]].sum() / 1e6
print("\nnational totals, millions")
print(national.round(2).unstack("scenario").to_string())

base = result[(result.scenario == "baseline")].pivot(index="state_fips", columns="year", values="eli_renters")
change = (100 * (base[2035] / base[2023] - 1)).sort_values()
names = panel.groupby("state_fips").state_abbr.first()
print("\nbaseline change in ELI renters, 2023 to 2035:")
print("  largest rises: ", ", ".join(f"{names[s]} {v:+.0f}%" for s, v in change.tail(4)[::-1].items()))
print("  largest falls: ", ", ".join(f"{names[s]} {v:+.0f}%" for s, v in change.head(3).items()))

h = S.hindcast(acs)                                   # test of the mechanism on the past: does age structure alone explain 2015 to 2023?
print(f"\nhindcast, mean growth 2015 to 2023: actual {h.actual_growth_pct.mean():.2f}% against {h.age_rates_growth_pct.mean():.2f}% predicted from age structure")
