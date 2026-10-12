"""
Example 11: how uncertain is the headline total, and which assumption moves it most?

Run from the project root:   uv run examples/11_uncertainty_and_sensitivity.py   (takes about a minute; 200 draws here, 1,000 in the notebook)
One draw resamples the training data, refits each model, scores the properties again, and draws every outcome as a random trial.
"""
import sys
sys.path.insert(0, "src")
import numpy as np
import uncertainty as U

prep = U.prepare()                               # loads the events, the scored properties, the LIHTC sample and the insured loans once
pt = U.point_estimates()                         # the headline pieces, from outputs/tables/units_at_stake_state.csv
draws = U.simulate(200, prep=prep)               # 200 passes through the three preservation models
tot = U.totals(draws, pt)                        # the total in the low and high definitions, baseline growth

low5, high95 = np.percentile(tot.total_low_case, 5), np.percentile(tot.total_high_case, 95)
print(f"headline total: {pt['growth'] + pt['nhpd_lb'] + pt['lihtc_adj'] + pt['mort_low']:,.0f} to {pt['growth'] + pt['nhpd_ub'] + pt['lihtc_raw'] + pt['mort_high']:,.0f} units")
print(f"with sampling error (90 percent range of the draws): {low5:,.0f} to {high95:,.0f} units")

sens = U.sensitivity(prep, pt)                   # one assumption changed at a time
print("\nchange in the midpoint of the total, percent of the headline:")
for _, r in sens.iloc[1:10].iterrows():
    print(f"  {r.change_pct:+5.1f}  {r.case}")
