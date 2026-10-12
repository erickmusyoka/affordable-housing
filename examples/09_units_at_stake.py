"""
Example 9: put the four pieces together by state, and ask what it would take to hold the line.

Run from the project root:   uv run examples/09_units_at_stake.py
Needs the result tables written by the notebooks (outputs/tables/), which `uv run src/run_short_report.py` rebuilds.
"""
import sys
sys.path.insert(0, "src")
import pandas as pd

import combined

t = combined.units_at_stake()                                   # one row per state: growth need, subsidies lapsing, LIHTC leaving, mortgages ending
parts = {"growth need (baseline)": "growth_need_baseline", "HOME and HUD subsidies not rescued": ("nhpd_unrescued_lb", "nhpd_unrescued_ub"),
         "LIHTC leaving monitoring": ("lihtc_leaving_adj", "lihtc_leaving"), "insured mortgages ending early": ("mortgage_lb", "mortgage_ub")}
print("national total, units by 2035 (low to high):")
for name, col in parts.items():
    lo, hi = (t[col].sum(), t[col].sum()) if isinstance(col, str) else (t[col[0]].sum(), t[col[1]].sum())
    print(f"  {name:38s} {lo:>10,.0f} to {hi:>10,.0f}")
print(f"  {'total':38s} {t.stake_low.sum():>10,.0f} to {t.stake_high.sum():>10,.0f}")
pres = (t.nhpd_unrescued_lb + t.lihtc_leaving_adj + t.mortgage_lb).sum() / t.stake_low.sum(), (t.nhpd_unrescued_ub + t.lihtc_leaving + t.mortgage_ub).sum() / t.stake_high.sum()
print(f"preservation share of the total: {100 * min(pres):.0f} to {100 * max(pres):.0f} percent")

print("\nlargest states (units):")
print(t.nlargest(5, "stake_mid")[["state", "stake_low", "stake_high", "lead"]].round(0).to_string(index=False))
print("\nhighest per 100 ELI renters:")
print(t.nlargest(4, "stake_per100_eli")[["state", "stake_per100_eli"]].round(1).to_string(index=False))
