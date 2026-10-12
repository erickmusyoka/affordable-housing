"""
Example 8: insured mortgages that end early. A cohort life table, with no fitted parameters.

Run from the project root:   uv run examples/08_insured_mortgages.py
NHPD records the scheduled end of each HUD-insured loan, but many loans terminate earlier, so the date-based expiry list misses them.
"""
import sys
sys.path.insert(0, "src")
import ahni
import mortgage_exit as M

subsidies, properties = M.load()

# 1. What happens when an insured mortgage terminates? Most properties keep another subsidy.
outcomes = M.outcome_table(subsidies)
print(f"properties that lost an insured mortgage: {len(outcomes):,}; had other subsidies: {100 * outcomes.other_subsidy_ever.mean():.0f} %")
rescue, n = M.rescue_rate(outcomes)
print(f"of {n:,} with no other active subsidy, {100 * rescue:.1f} % hold a new insured record (a refinance)")

# 2. The life table: of the loans that started in year s, what share has already terminated?
curve = M.cohort_curve(subsidies)
print("\nshare of insured loans terminated, by start year:", {y: f"{100 * curve.share_terminated[y]:.0f}%" for y in (2005, 2010, 2012, 2015, 2018, 2020)})

# 3. The loans the date-based list misses: insured loan is the only active subsidy, scheduled end after 2035
exposed = M.exposed_properties(subsidies, properties)
print(f"\n{len(exposed)} properties ({exposed.units.sum():,.0f} units) have an insured loan as their only active subsidy and a scheduled end that is not already on the 2026-2035 list")

# 4. Project forward: a loan that started in year s is observed at age 2026 - s; by 2035 it is nine years older, so read its
#    chance of having ended from the cohort that started nine years earlier (7 and 11 years give the low and high cases).
table, info = M.exposure_by_state(subsidies, properties)
total = table[["mortgage_unrescued_low", "mortgage_unrescued_central", "mortgage_unrescued_high"]].sum()
print(f"units expected to lose the mortgage and not be rescued by 2035: low {total.iloc[0]:,.0f}, central {total.iloc[1]:,.0f}, high {total.iloc[2]:,.0f}")
print("largest (central):", ", ".join(f"{ahni.ABBR[s]} {v:,.0f}" for s, v in table.mortgage_unrescued_central.nlargest(4).items()))
