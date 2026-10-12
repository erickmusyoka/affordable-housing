"""
Example 13: the rescue rate rose after 2021. Is it the mix of properties, can a model keep up, and what if it persists?

Run from the project root:   uv run examples/13_rescue_rate_drift.py
"""
import sys
sys.path.insert(0, "src")
import pandas as pd

import preservation as P
import rescue_drift as D
import uncertainty as U

events = P.build_events()
events["period"] = pd.cut(events.Y, [2010, 2015, 2020, 2023], labels=["2011-15", "2016-20", "2021-23"])

# 1. Is it the mix of properties? Compare the rate and the mix over time, and within the biggest group.
print("all lapses:")
print(events.groupby("period", observed=True).agg(lapses=("Y", "size"), rescue_rate_pct=("resub_3y", lambda s: 100 * s.mean()), median_units=("total_units", "median")).round(2).to_string())
print("\nHOME-only lapses (93 percent of all):")
print(events[events.group == "home_only"].groupby("period", observed=True).resub_3y.agg(lapses="size", rescue_rate_pct=lambda s: 100 * s.mean()).round(2).to_string())

# 2. Can a model updated in real time keep up? For lapses in year T, use only outcomes already labeled (lapses through T-3).
rt = D.real_time(events)
print("\nreal-time test, 2019-2023 (lower log loss is better):")
print(D.summary(rt).sort_values("log_loss").round(3).to_string())
print("\nmean predicted rescue rate (%) in 2022 and 2023, observed 3.9 and 4.5:")
print((100 * rt[rt.year >= 2022].pivot(index="method", columns="year", values="predicted")).round(1).to_string())

# 3. What if the recent rate persists? Move the model's intercept to reproduce the 2022-23 rate, then recompute the headline.
prep = U.prepare()
p_recent, shift, rate = D.recent_rate_probabilities(prep["ev"], prep["scored"])
lo0, hi0 = U._unrescued(prep["scored"], prep["scored"].p_rescue)
lo1, hi1 = U._unrescued(prep["scored"], p_recent)
print(f"\nif rescue stays at {100 * rate:.1f} percent: HOME and HUD units not rescued {lo0:,.0f}-{hi0:,.0f} -> {lo1:,.0f}-{hi1:,.0f}")
