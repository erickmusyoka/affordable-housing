"""
Example 6: when a HOME or HUD subsidy ends, which properties get a new subsidy? A logistic regression on property size.

Run from the project root:   uv run examples/06_rescue_model.py
Events are properties whose last subsidy ended 2011-2023; the label is "re-subsidized within three years". Train on early lapses, test on later ones.
"""
import sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import preservation as P

events = P.build_events()                              # one row per property whose last subsidy ended
print(f"{len(events):,} events; rescued within 3 years: {int(events.resub_3y.sum())} ({100 * events.resub_3y.mean():.1f} %)")
print("rescue rate by program group:", (100 * events.groupby("group").resub_3y.mean()).round(1).to_dict())

train, test = P.split(events, "main")                  # train on lapses up to 2015, test on lapses from 2019
cols = P.FEATURE_SETS["size + program"]                # log total units plus program-group indicators
p_logit = P.fit_predict(train, test, cols, "logit")
p_boost = P.fit_predict(train, test, P.FEATURE_SETS["property"], "boosting")
y = test.resub_3y.to_numpy()
print(f"\ntest set: {len(test):,} events, {int(y.sum())} rescued")
print(f"AUC, logistic regression on size and program: {roc_auc_score(y, p_logit):.2f}")
print(f"AUC, gradient boosting on all property features: {roc_auc_score(y, p_boost):.2f}")
p_size = P.fit_predict(train, test, P.FEATURE_SETS["size only"], "logit")
print(f"AUC, logistic regression on size alone: {roc_auc_score(y, p_size):.2f}")
print(f"share of all rescues found in the top-scored fifth of events: {100 * P.capture(y, p_size):.0f} % (size alone), {100 * P.capture(y, p_logit):.0f} % (size and program)")

# what the model says about size: rescue chance by size band, among HUD-program events
hud = events[events.group == "hud"].assign(band=lambda d: pd.cut(d.total_units, [0, 10, 25, 50, 100, 200, 10**6]))
print("\nHUD-program rescue rate by size (units):")
print((100 * hud.groupby("band", observed=True).resub_3y.mean()).round(1).to_string())

# score the properties whose last subsidy ends 2026-2035
scored, model = P.rescue_projection(events)
ok = scored[scored.p_rescue.notna()]
print(f"\nscored {len(ok):,} active properties; expected units not rescued if every subsidy lapsed: "
      f"{(ok.units_lb * (1 - ok.p_rescue)).sum():,.0f} to {(ok.units_ub * (1 - ok.p_rescue)).sum():,.0f}")
