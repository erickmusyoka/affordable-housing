"""
Clean the NLIHC Gap state table (saved by hand from https://nlihc.org/gap on 2026-10-11, see data/raw/nlihc_gap/SOURCE.txt).

Reads:  data/raw/nlihc_gap/gap_state_table.csv
Writes: data/interim/nlihc_gap_state.csv    one row per state: state_fips, nlihc_eli_renters, nlihc_affordable_available_per100, nlihc_eli_severe_burden_pct
Used only to benchmark our own state measures (notebook 06_combined/03_external_benchmarks); it is never an input to a model.
"""

import pandas as pd

from common import FIPS_STATES, INTERIM, RAW, STATE_FIPS

ABBR = {'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'District of Columbia': 'DC', 'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'}


def main():
    g = pd.read_csv(RAW / "nlihc_gap" / "gap_state_table.csv")
    g["state_fips"] = g["state"].map(ABBR).map(STATE_FIPS)
    assert g.state_fips.notna().all(), g[g.state_fips.isna()].state.tolist()
    assert len(g) == 51 and set(g.state_fips) == set(FIPS_STATES)
    out = g.rename(columns={"eli_renter_households": "nlihc_eli_renters", "affordable_available_per_100_eli": "nlihc_affordable_available_per100",
                            "eli_severe_burden_pct": "nlihc_eli_severe_burden_pct"})[["state_fips", "state", "nlihc_eli_renters", "nlihc_affordable_available_per100", "nlihc_eli_severe_burden_pct"]]
    out.to_csv(INTERIM / "nlihc_gap_state.csv", index=False)
    print(f"nlihc_gap_state.csv: {len(out)} states; ELI renter households sum {out.nlihc_eli_renters.sum():,.0f}; per 100: median {out.nlihc_affordable_available_per100.median():.0f}")


if __name__ == "__main__":
    main()
