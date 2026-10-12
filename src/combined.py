"""
Combine the three models into one state table of units at stake by 2035 (Part 6).

    demand    stock-flow scenarios (src/scenarios.py): extra affordable units needed to hold the 2023 affordable-unit ratio as ELI renters change
              (baseline; the low scenario needs none, the high scenario more)
    supply 1  property-level rescue model for HOME and HUD-program subsidies expiring 2026-2035 (src/preservation.py): units not rescued if the subsidy
              lapses (lower and upper bound from the assisted-unit range)
    supply 2  state-effects exit model for LIHTC projects placed 1996-2005 (src/lihtc_exit.py): units expected to leave the monitoring list when each
              cohort reaches about 31 years (calibration-adjusted to unadjusted)
    supply 3  HUD-insured mortgage pathway (src/mortgage_exit.py): units of properties whose only active subsidy is an insured loan, not already on the
              date-based 2026-2035 list, expected to lose the mortgage by 2035 and not be rescued (low to high case)

The pieces are NOT strictly additive: a property that holds both a LIHTC and a HOME or HUD subsidy can appear in both supply columns, supply 1 is
conditional on a lapse (an upper bound on loss), and supply 2 counts leaving the monitoring list, not loss of affordability. The sum is used only to
rank states and to size the order of magnitude; each column is shown separately.
"""

import numpy as np
import pandas as pd

from common import INTERIM

TABLES = "outputs/tables"


def units_at_stake():
    sc = pd.read_csv(f"{TABLES}/scenarios_state.csv", dtype={"state_fips": str})
    base = sc[(sc.year == 2035)]
    need = base.pivot(index="state_fips", columns="scenario", values="extra_units_to_hold_2023_ratio").clip(lower=0)
    need.columns = [f"growth_need_{c}" for c in need.columns]
    eli = sc[(sc.year == 2023) & (sc.scenario == "baseline")].set_index("state_fips")[["eli_renters", "afford_units", "state"]]
    pres = pd.read_csv(f"{TABLES}/preservation_state.csv", dtype={"state_fips": str}).set_index("state_fips")
    lih = pd.read_csv(f"{TABLES}/lihtc_exit_state.csv", dtype={"state_fips": str}).set_index("state_fips")
    mor = pd.read_csv(f"{TABLES}/mortgage_exit_state.csv", dtype={"state_fips": str}).set_index("state_fips")
    t = eli.join(need).join(pres[["unrescued_lb", "unrescued_ub", "unscored_units_ub"]].rename(columns={"unrescued_lb": "nhpd_unrescued_lb", "unrescued_ub": "nhpd_unrescued_ub", "unscored_units_ub": "nhpd_unscored_units"}))
    t = t.join(lih[["expected_units_leaving_adj", "expected_units_leaving", "monitored_units_placed_1996_2005", "share_leaving_pct"]].rename(
        columns={"expected_units_leaving_adj": "lihtc_leaving_adj", "expected_units_leaving": "lihtc_leaving", "monitored_units_placed_1996_2005": "lihtc_monitored_1996_2005", "share_leaving_pct": "lihtc_leaving_pct"}))
    t = t.join(mor[["mortgage_unrescued_low", "mortgage_unrescued_high"]].rename(columns={"mortgage_unrescued_low": "mortgage_lb", "mortgage_unrescued_high": "mortgage_ub"}))
    t = t.fillna({"nhpd_unrescued_lb": 0, "nhpd_unrescued_ub": 0, "nhpd_unscored_units": 0, "lihtc_leaving_adj": 0, "lihtc_leaving": 0, "mortgage_lb": 0, "mortgage_ub": 0})
    t["stake_low"] = t.growth_need_baseline + t.nhpd_unrescued_lb + t.lihtc_leaving_adj + t.mortgage_lb
    t["stake_high"] = t.growth_need_baseline + t.nhpd_unrescued_ub + t.lihtc_leaving + t.mortgage_ub
    t["stake_mid"] = (t.stake_low + t.stake_high) / 2
    t["stake_per100_eli"] = 100 * t.stake_mid / t.eli_renters
    t["stake_pct_of_afford_stock"] = 100 * t.stake_mid / t.afford_units
    parts = t[["growth_need_baseline", "nhpd_unrescued_ub", "lihtc_leaving", "mortgage_ub"]].rename(columns={"growth_need_baseline": "growth", "nhpd_unrescued_ub": "nhpd", "lihtc_leaving": "lihtc", "mortgage_ub": "mortgage"})
    t["lead"] = parts.idxmax(axis=1)
    return t.reset_index()
