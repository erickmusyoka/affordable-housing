"""
Part 5d: the HUD-insured mortgage pathway in NHPD.

NHPD lists FHA-insured mortgages that carry affordability restrictions (Section 236 and its refinances, 221(d)(3) BMIR, and similar; data dictionary note 7) as
"HUD Insured" subsidies. Their status is "Inactive, Terminated Mortgage" when the loan appears in HUD's terminated-mortgage database, and the recorded end date
is the SCHEDULED maturity, not the termination date (48 percent of terminated records have a scheduled end after 2025, so the loan ended early; the actual
termination year is not in the data). A time-split model like the rescue model is therefore not possible; this module uses cohorts instead.

What it measures
    pathway_table      inactive subsidies by program and reason (Terminated Mortgage is 95 percent of inactive HUD Insured records and appears for no other program)
    outcome_table      for properties whose insured mortgage terminated: share still holding an active subsidy, by whether other subsidies were present
    cohort_curve       share of insured subsidies terminated, by start year (3-year smoothed): how quickly insured loans end
    exposure           properties whose ONLY active subsidy is an insured loan and whose scheduled end is not already in the 2026-2035 list: the units the date-based
                       expiry list misses. Cohort rule: a loan started in year s is observed today at age 2026 - s; by 2035 it will be nine years older, so its cumulative
                       termination share by 2035 is read from the cohort that started nine years earlier. Net loss = terminations x (1 - rescue rate), with the rescue
                       rate taken from history (properties that lost an insured mortgage and have no other active subsidy)
"""

import numpy as np
import pandas as pd

from common import INTERIM

AS_OF = 2026
HORIZON_END = 2035


def load():
    d = pd.read_csv(INTERIM / "nhpd_subsidies.csv.gz", dtype={"state_fips": str})
    p = pd.read_csv(INTERIM / "nhpd_properties.csv.gz", dtype={"state_fips": str, "county_fips": str}).set_index("property_id")
    return d, p


def pathway_table(d):
    i = d[d.status == "Inactive"]
    t = pd.crosstab(i.program, i.inactive_reason)
    t["total_inactive"] = t.sum(axis=1)
    t["terminated_mortgage_share"] = t.get("Terminated Mortgage", 0) / t["total_inactive"]
    return t.sort_values("total_inactive", ascending=False)


def outcome_table(d):
    """Properties whose insured mortgage terminated: whether they hold an active subsidy now, split by presence of other subsidies."""
    a = d[d.status == "Active"]
    act_any, act_other, act_ins = a.groupby("property_id").size(), set(a[a.program != "HUD Insured"].property_id), set(a[a.program == "HUD Insured"].property_id)
    others_ever = set(d[d.program != "HUD Insured"].property_id)
    tm = d[(d.program == "HUD Insured") & (d.inactive_reason == "Terminated Mortgage")]
    t = tm.groupby("property_id").agg(scheduled_end=("end_year", "max"), units=("assisted_units", "max"))
    t["other_subsidy_ever"] = t.index.isin(others_ever)
    t["other_active_now"] = t.index.isin(act_other)
    t["any_active_now"] = t.index.isin(act_any.index)
    t["insured_active_now"] = t.index.isin(act_ins)
    return t


def rescue_rate(outcomes):
    """Share of properties that lost an insured mortgage, have no other active subsidy, and hold an active insured record now (refinanced)."""
    g = outcomes[~outcomes.other_active_now]
    return float(g.insured_active_now.mean()), len(g)


def cohort_curve(d):
    ins = d[(d.program == "HUD Insured") & d.start_year.between(1965, 2024) & (d.status != "Inconclusive")].copy()
    ins["term"] = (ins.inactive_reason == "Terminated Mortgage").astype(int)
    g = ins.groupby("start_year").term.agg(["sum", "size"]).reindex(range(1965, 2025), fill_value=0)
    share = g["sum"].rolling(3, center=True, min_periods=1).sum() / g["size"].rolling(3, center=True, min_periods=1).sum()
    return pd.DataFrame({"share_terminated": share, "subsidies": g["size"]})


def exposed_properties(d, p):
    """Properties whose only active subsidy is a HUD Insured loan and whose last scheduled end is not already in the 2026 to 2035 list."""
    a = d[d.status == "Active"]
    only = a.groupby("property_id").program.apply(lambda s: set(s) == {"HUD Insured"})
    only = only[only].index
    e = a[a.property_id.isin(only) & (a.program == "HUD Insured")].groupby("property_id").agg(start_year=("start_year", "max"), units=("assisted_units", "max"))
    e["last_end_year"] = p.reindex(e.index).last_end_year.values
    e["state_fips"] = p.reindex(e.index).state_fips.values
    e["total_units"] = p.reindex(e.index).total_units.values
    return e[~e.last_end_year.between(AS_OF, HORIZON_END)].copy()


def projection(e, curve, shift, rescue):
    s = e.start_year.clip(1965 + shift, 2024).astype(int)
    now, fut = curve.share_terminated.reindex(s).to_numpy(), curve.share_terminated.reindex(s - shift).to_numpy()
    add = np.clip((fut - now) / (1 - now), 0, 1)
    term = e.units.to_numpy() * add
    return term, term * (1 - rescue)


def exposure_by_state(d, p, shifts=(7, 9, 11)):
    outcomes = outcome_table(d)
    resc, n_resc = rescue_rate(outcomes)
    alone = outcomes[~outcomes.other_subsidy_ever]
    resc_alone = float(alone[~alone.other_active_now].insured_active_now.mean())
    e = exposed_properties(d, p)
    curve = cohort_curve(d)
    cases = {"low": (shifts[0], resc), "central": (shifts[1], resc), "high": (shifts[2], resc_alone)}
    out = pd.DataFrame(index=sorted(e.state_fips.unique()))
    out["exposed_properties"] = e.groupby("state_fips").size()
    out["exposed_units"] = e.groupby("state_fips").units.sum()
    for k, (sh, rr) in cases.items():
        term, net = projection(e, curve, sh, rr)
        out[f"terminate_{k}"] = pd.Series(term, index=e.index).groupby(e.state_fips).sum()
        out[f"mortgage_unrescued_{k}"] = pd.Series(net, index=e.index).groupby(e.state_fips).sum()
    return out.fillna(0), {"rescue_default": resc, "rescue_alone_ever": resc_alone, "n_rescue_group": n_resc, "cases": cases}
