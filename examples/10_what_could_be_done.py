"""
Example 10: planning arithmetic. What would it take to keep the number of affordable units per 100 ELI renters at its 2023 level?

Run from the project root:   uv run examples/10_what_could_be_done.py
This is arithmetic on the model outputs, not a forecast of what any policy achieves, and it has no costs in it.
"""
import sys
sys.path.insert(0, "src")
import pandas as pd

import actions as A
import combined

stake = combined.units_at_stake()
sc = pd.read_csv("outputs/tables/scenarios_state.csv", dtype={"state_fips": str})
base = sc[sc.scenario == "baseline"]
stock = base[base.year == 2023].afford_units.sum()               # rent-affordable units in 2023 (CHAS), held fixed
eli_2023 = base[base.year == 2023].eli_renters.sum()
eli_2035 = base[base.year == 2035].eli_renters.sum()

loss_low = stake.nhpd_unrescued_lb.sum() + stake.lihtc_leaving_adj.sum() + stake.mortgage_lb.sum()
loss_high = stake.nhpd_unrescued_ub.sum() + stake.lihtc_leaving.sum() + stake.mortgage_ub.sum()
ratio_2023 = 100 * stock / eli_2023
print(f"2023: {stock / 1e6:.2f} million rent-affordable units for {eli_2023 / 1e6:.2f} million ELI renters = {ratio_2023:.1f} per 100")
print(f"2035, nothing lost and nothing built:  {100 * stock / eli_2035:.1f} per 100 (more renters, same units)")
print(f"2035, preservation losses taken out in full: {100 * (stock - loss_high) / eli_2035:.1f} to {100 * (stock - loss_low) / eli_2035:.1f} per 100")

need = [0.01 * ratio_2023 * eli_2035 - stock + loss for loss in (loss_low, loss_high)]
print(f"units to preserve or add by 2035 to hold {ratio_2023:.1f}: {need[0]:,.0f} to {need[1]:,.0f} ({need[0] / 12:,.0f} to {need[1] / 12:,.0f} a year)")

# the lever grid: share of the expected losses prevented (rows) against new units added per year (columns), 2035 ratio per 100 ELI renters
grid = A.lever_grid(stock, eli_2035, (loss_low + loss_high) / 2)
print("\naffordable units per 100 ELI renters in 2035:")
print(grid.round(1).to_string())
