"""
Example 4: build the need index step by step, and see how sure we can be of the ranking.

Run from the project root:   uv run examples/04_need_index.py
Seven measures per state -> z-scores (distance from the average state, in standard deviations, signed so that higher always means
more need) -> three groups (burden, gap, pressure) averaged -> one score. Then redraw every measure within its survey error 1,000 times.
"""
import sys
sys.path.insert(0, "src")
import pandas as pd

import ahni

panel = pd.read_csv("data/processed/panel.csv", dtype={"state_fips": str})
est = ahni.build_components(panel)            # one value and one standard error per measure and state (2022-24 averages, CHAS 2018-2022)
z = ahni.zscores(est)                         # z-scores, capped at plus or minus 3
dims = ahni.dimension_scores(z)               # burden, gap, pressure
score = ahni.index_from_z(z, ahni.SCHEMES["equal dimensions (headline)"])
rank = ahni.rank_of(score)                    # 1 = highest need

print("highest need:", ", ".join(rank.sort_values().index[:10]))
print("lowest need: ", ", ".join(rank.sort_values().index[-6:][::-1]))

ranks = ahni.monte_carlo(est, n=1000)         # 1,000 redraws of every measure within its standard error
width = ranks.quantile(0.95) - ranks.quantile(0.05)
top = ranks.loc[:, rank[rank <= 8].index]
print(f"\ntop eight stay within two places of their headline rank in {100 * ((top - rank[top.columns]).abs() <= 2).mean().mean():.1f} percent of draws")
print(f"states whose 90 percent rank interval is 10 or more places wide: {int((width >= 10).sum())}")
schemes, loadings, pc_share = ahni.all_schemes(z)
print("rank correlation of each weighting with the headline:", schemes.apply(ahni.rank_of).corrwith(rank, method="spearman").round(2).to_dict())
