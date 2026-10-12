# %% [markdown]
# # Part 5h: the rescue rate moved after 2021. Can a model keep up, and what does it do to the headline?
#
# The state hindcast (Part 5g) found that the rescue rate rose after 2021: 3.9 and 4.5 percent of 2022 and 2023 lapses were re-subsidized within three years, against about 1 percent before. A reviewer will call that **non-stationarity** and ask three things. Is it just a change in the mix of properties? Can a model that is updated as time passes anticipate it? And what would the headline be if the recent rate persisted? (`src/rescue_drift.py`)

# %%
import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

import preservation as P
import rescue_drift as D
from viz import apply_style, SERIES, save_figure, BASELINE, INK_2, dot

apply_style()
pd.set_option("display.width", 220)
ev = P.build_events()
ev["period"] = pd.cut(ev.Y, [2010, 2015, 2020, 2023], labels=["2011-15", "2016-20", "2021-23"])

# %% [markdown]
# ## 1. Is it the mix of properties?
#
# If the rise were a change in what lapses (more HUD properties, bigger properties) it would show up as a change in composition and the model would absorb it. It does not: the rise is inside the HOME-only group, which is 93 percent of lapses.

# %%
comp = ev.groupby("period", observed=True).agg(lapses=("Y", "size"), rescue_rate_pct=("resub_3y", lambda s: 100 * s.mean()), median_units=("total_units", "median"), hud_share_pct=("g_hud", lambda s: 100 * s.mean()))
display(comp.round(2))
grp = ev[ev.group.isin(["home_only", "hud"])].groupby(["group", "period"], observed=True).agg(lapses=("Y", "size"), rescue_rate_pct=("resub_3y", lambda s: 100 * s.mean()))
display(grp.round(2))
yearly = ev.groupby("Y").resub_3y.agg(rate=lambda s: 100 * s.mean(), lapses="size", rescued="sum")
display(yearly.round(2).T)

# %% [markdown]
# ## 2. Can a model that is updated in real time anticipate it?
#
# For lapses in year T a forecaster knows only outcomes of lapses through T-3, because a lapse needs three years to be labeled. Five methods are refit for every T from 2019 to 2023.

# %%
rt = D.real_time(ev)
summ = D.summary(rt).sort_values("log_loss")
display(summ.round(4))
pred = rt.pivot(index="method", columns="year", values="predicted") * 100
obs = rt.groupby("year")["observed"].first() * 100
display(pd.concat([pred.round(2), pd.DataFrame(obs.round(2)).T.rename(index={"observed": "OBSERVED"})]))

# %%
fig, axs = plt.subplots(1, 2, figsize=(10.6, 4.3))
ax = axs[0]
for g, c, lab in (("home_only", SERIES[0], "HOME only"), ("hud", SERIES[1], "HUD programs")):
    s = ev[ev.group == g].groupby("Y").resub_3y.mean() * 100
    ax.plot(s.index, s.values, color=c, lw=2.0, marker="o", ms=4)
    ax.annotate(lab, (s.index[-1], s.values[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK_2)
ax.set_xlabel("year the last subsidy ended"); ax.set_ylabel("re-subsidized within three years, %"); ax.set_xlim(2010.5, 2025.5)
ax.set_title("The rise is inside the HOME-only group", fontsize=10, loc="left")
ax = axs[1]
colors = {"static": SERIES[0], "recency weighted": SERIES[2], "intercept shift": SERIES[3], "year trend": "#8a5fd0", "constant rate": "#898781"}
for k, c in colors.items():
    ax.plot(pred.columns, pred.loc[k], color=c, lw=1.6, label=k, marker="o", ms=3.5)
ax.plot(obs.index, obs.values, color=SERIES[1], lw=2.6, marker="o", ms=5, label="observed")
ax.set_xticks(list(pred.columns)); ax.set_ylabel("mean rescue rate of the year's lapses, %"); ax.legend(fontsize=8, loc="upper left")
ax.set_title("No real-time method anticipates 2022-23", fontsize=10, loc="left")
save_figure(fig, "59_rescue_drift", part="05_preservation")
plt.show()

# %% [markdown]
# ## 3. What the headline would be if the recent rate persisted
#
# The scored properties get the fitted logistic regression with its intercept moved so that it reproduces the 2022-23 rescue rate on those lapses. This is a scenario, not a prediction: nothing in the data says the 2022-23 rate will last.

# %%
import uncertainty as U
prep = U.prepare(); pt = U.point_estimates()
p_recent, delta, rate = D.recent_rate_probabilities(prep["ev"], prep["scored"])
base_p = prep["scored"].p_rescue
print(f"2022-23 observed rescue rate {100 * rate:.2f} %; intercept shift {delta:+.2f} log-odds; mean rescue chance of the scored properties {100 * base_p.mean():.2f} % -> {100 * p_recent.mean():.2f} %")
lb0, ub0 = U._unrescued(prep["scored"], base_p); lb1, ub1 = U._unrescued(prep["scored"], p_recent)
print(f"HOME and HUD units not rescued: {lb0:,.0f} to {ub0:,.0f} -> {lb1:,.0f} to {ub1:,.0f}")
sens = U.sensitivity(prep, pt)                      # includes this case as a standard row
row = sens[sens.case.str.startswith("Rescue chance stays")].iloc[0]
print(f"total: {row.total_low:,.0f} to {row.total_high:,.0f} ({row.change_pct:+.1f} % on the midpoint); headline {sens.total_low.iloc[0]:,.0f} to {sens.total_high.iloc[0]:,.0f}")
display(sens.round(1))
rt.to_csv("outputs/tables/rescue_drift_real_time.csv", index=False)

# %% [markdown]
# ## Findings
#
# **The rise is not a change in the mix of properties.** The rescue rate was 1.11 percent for 2011-15 lapses, 0.85 percent for 2016-20, and 3.23 percent for 2021-23 (3.94 percent in 2022 and 4.49 in 2023). The share of HUD-program lapses (5 to 7 percent) and the median size (2 to 4 units) barely moved. The rise is inside the HOME-only group, 93 percent of lapses, whose rate went from 0.43 and 0.60 percent to 2.63 percent; the HUD-program rate (only about 60 lapses a year, so noisy) was 11.04 percent in 2011-15, dipped to 5.67 percent in 2016-20, and was 10.82 percent in 2021-23.
#
# **No real-time method anticipates it.** Refit every year on the outcomes already labeled (lapses through T-3), the static model, a recency-weighted version, an intercept shift to the latest two years, a year trend, and a constant recent rate all predict 0.7 to 1.7 percent for 2022 and 2023 against 3.9 and 4.5 observed. The simplest method (static) has the best log loss (0.098) and the best average calibration error (1.3 points); the extra machinery does not help, because a jump in the data cannot be learned from the years before it. This is a regime change, probably funding or program rules that the data do not contain, not something to model away.
#
# **So it is treated as a scenario.** If the rescue rate stays at its 2022-23 level (4.21 percent), the fitted model's intercept moves up 0.93 log-odds and the average rescue chance of the scored properties rises from 4.6 to 10.0 percent. HOME and HUD units not rescued fall from 295,000 to 353,000 to **252,000 to 301,000**, and the total from 900,000 to 1,033,000 to **857,000 to 981,000 (-4.9 percent on the midpoint)**. That is the largest modeling-side sensitivity in Part 6b, though still far smaller than the growth scenario. The headline keeps the all-years model, which is the conservative choice (a higher loss), and states the alternative.
#
# **Caveats.** Two high years (76 rescues among 1,807 lapses); the cause is unknown; HUD-program rates are noisy and back near their 2011-15 level, so the rise is mainly a HOME-program story; the intercept shift assumes the size gradient is unchanged, which the hindcast (Part 5g) showed is shaky at the top sizes.
