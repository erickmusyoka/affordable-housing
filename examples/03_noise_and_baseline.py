"""
Example 3: how noisy is the rent-burden series, and what does "next year equals this year" score?

Run from the project root:   uv run examples/03_noise_and_baseline.py
Reads data/processed/panel.csv (one row per state and year).
"""
import numpy as np
import pandas as pd

panel = pd.read_csv("data/processed/panel.csv", dtype={"state_fips": str})
d = panel.dropna(subset=["severe_rent_burden_pct"]).sort_values(["state_fips", "year"]).copy()
print(f"state-years with the target: {len(d)} ({d.state_fips.nunique()} states, {d.year.nunique()} years)")

# 1. Where does the variation sit? Peel off state averages, then year averages; what is left is the remainder.
y = d["severe_rent_burden_pct"]
total_ss = ((y - y.mean()) ** 2).sum()
state_part = d.groupby("state_fips")["severe_rent_burden_pct"].transform("mean") - y.mean()
resid_1 = y - y.mean() - state_part
year_part = resid_1.groupby(d["year"]).transform("mean")
rest = resid_1 - year_part
print("\nshare of the variance in severe burden")
print(f"  between states      {100 * (state_part ** 2).sum() / total_ss:5.1f} %")
print(f"  common year shocks  {100 * (year_part ** 2).sum() / total_ss:5.1f} %")
print(f"  left over           {100 * (rest ** 2).sum() / total_ss:5.1f} %   (real state movement plus sampling noise)")

# 2. How big is the sampling error? The ACS publishes a 90 percent margin of error with every estimate.
moe = d["severe_rent_burden_pct_moe"]
print(f"\n90 percent margin of error: median {moe.median():.2f} points, 90th percentile {moe.quantile(0.9):.2f}, maximum {moe.max():.2f}")
last = d[d.year == 2024].set_index("state_fips")
print("  2024:", ", ".join(f"{s} {last.loc[f, 'severe_rent_burden_pct_moe']:.1f}" for s, f in (("WY", "56"), ("VT", "50"), ("CA", "06"), ("TX", "48"))))

# 3. The benchmark: predict next year's value with this year's value (persistence).
d["next"] = d.groupby("state_fips")["severe_rent_burden_pct"].shift(-1)
d["next_year"] = d.groupby("state_fips")["year"].shift(-1)
pairs = d[(d.next_year == d.year + 1)].dropna(subset=["next"])          # only true consecutive years (2020 is missing)
err = (pairs["next"] - pairs["severe_rent_burden_pct"]).abs()
print(f"\npersistence error over {len(pairs)} year-to-year changes: mean absolute error {err.mean():.2f} points")

# 4. What share of those changes is smaller than the survey's own uncertainty? Combine the margins of the two years.
d["next_moe"] = d.groupby("state_fips")["severe_rent_burden_pct_moe"].shift(-1)
pairs = d[(d.next_year == d.year + 1)].dropna(subset=["next", "next_moe"])
combined_moe = np.sqrt(pairs["severe_rent_burden_pct_moe"] ** 2 + pairs["next_moe"] ** 2)
inside = (pairs["next"] - pairs["severe_rent_burden_pct"]).abs() <= combined_moe
print(f"changes smaller than the combined margin of error of the two years: {100 * inside.mean():.0f} %")
