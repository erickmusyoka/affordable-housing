"""
Example 2: what moves with the rent burden? Pooled regression against two-way fixed effects.

Run from the project root:   uv run examples/02_fixed_effects_regression.py
Same specification as the explanatory notebook: severe burden on log rent, log income, vacancy, unemployment, renter share,
last year's multifamily permits, and HUD-assisted units, with standard errors clustered by state.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, PooledOLS

panel = pd.read_csv("data/processed/panel.csv", dtype={"state_fips": str})

# last year's 5+ unit permits per 1,000 residents (what builders had started before the survey year)
permits = panel[["state_fips", "year", "permits_5plus", "population_rate_base"]].copy()
permits["permits_lag1"] = 1000 * permits["permits_5plus"] / permits["population_rate_base"]
permits["year"] += 1
p = panel.merge(permits[["state_fips", "year", "permits_lag1"]], on=["state_fips", "year"], how="left")
p = p[p["acs_available"] == 1]

df = pd.DataFrame({
    "state_fips": p["state_fips"], "year": p["year"], "burden": p["severe_rent_burden_pct"],
    "ln_rent": np.log(p["median_gross_rent"]), "ln_income": np.log(p["median_hh_income"]),
    "vacancy": p["rental_vacancy_rate_pct"], "unemployment": p["unemployment_rate_pct"], "renter_share": p["renter_share_pct"],
    "permits_lag1": p["permits_lag1"], "assisted_per_100": p["psh_units_per_100_renter_hh"],
}).dropna().set_index(["state_fips", "year"])
X = ["ln_rent", "ln_income", "vacancy", "unemployment", "renter_share", "permits_lag1", "assisted_per_100"]
print(f"{len(df)} state-years, {df.index.get_level_values(0).nunique()} states\n")

clustered = dict(cov_type="clustered", cluster_entity=True)          # errors may be correlated within a state
pooled = PooledOLS(df["burden"], sm.add_constant(df[X])).fit(**clustered)
twoway = PanelOLS(df["burden"], df[X], entity_effects=True, time_effects=True).fit(**clustered)

out = pd.DataFrame({"pooled OLS": pooled.params[X], "pooled t": pooled.tstats[X], "two-way FE": twoway.params[X], "two-way t": twoway.tstats[X]})
print(out.round(2).to_string())
print(f"\nR-squared after removing state and year effects: {twoway.rsquared:.2f}")
print(f"a 10 percent higher rent goes with {twoway.params['ln_rent'] * np.log(1.10):+.1f} points; a 10 percent higher income with {twoway.params['ln_income'] * np.log(1.10):+.1f}")
