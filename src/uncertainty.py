"""
How uncertain is the headline total, and what does it depend on?

The combined result (src/combined.py) adds four pieces and reports a low and a high case. Those cases are bounds from different
definitions, not statistical intervals. This module adds the missing statistical part: it carries the sampling uncertainty of each piece
through to the total by bootstrap, and it computes how much the total moves when one assumption changes.

Sampling uncertainty (one draw = one pass through all three preservation pieces; growth is a scenario, not a draw):
    rescue     resample the 12,822 lapse events with replacement, refit the logistic regression, score the active expiring properties,
               then draw each property's outcome (rescued or not) as a Bernoulli trial with its fitted probability. The sum of units in
               properties that are not rescued is one draw (lower and upper unit counts).
    LIHTC      resample the 1990-95 projects, refit the state-effects model, project the 1996-2005 projects still monitored, and draw each
               project's exit as a Bernoulli trial with its remaining share (unadjusted, and scaled by the calibration factor).
    mortgage   resample the insured-loan records to redraw the cohort termination curve, draw the refinance rate from its observed
               binomial, and draw each exposed property's termination and rescue as Bernoulli trials (low, central, high shifts).
Each draw therefore includes parameter uncertainty (the refit) and outcome randomness (the Bernoulli trials).
"""

import numpy as np
import pandas as pd

import lihtc_exit as L
import mortgage_exit as M
import preservation as P
import rescue_drift as D
from common import INTERIM

TABLES = "outputs/tables"
CHOSEN_C = 0.3


# ------------------------------------------------------------------ inputs prepared once
def prepare():
    ev = P.build_events()
    q, base_model = P.rescue_projection(ev)
    scored = q[q.p_rescue.notna()].copy()

    proj = L.load_projects()
    sample = proj[proj.pis_year.between(1990, 1995)]
    final = L.StateLogit(CHOSEN_C).fit(sample)
    fut = L.project_future(proj, final)
    cell = fut.groupby(["state_fips", "pis_year"]).left_monitoring.mean().rename("obs_cell")
    fut = fut.join(cell, on=["state_fips", "pis_year"])
    mon = fut[fut.left_monitoring == 0].copy()
    cal = L.cohort_split_eval(proj, C=CHOSEN_C)
    scale = cal["mean_obs"] / cal["mean_pred"]

    subs, props = M.load()
    outcomes = M.outcome_table(subs)
    resc, n_resc = M.rescue_rate(outcomes)
    alone = outcomes[~outcomes.other_subsidy_ever]
    alone = alone[~alone.other_active_now]
    resc_alone, n_alone = float(alone.insured_active_now.mean()), len(alone)
    exposed = M.exposed_properties(subs, props)
    insured = subs[subs.program == "HUD Insured"]
    return dict(ev=ev, scored=scored, sample=sample, final=final, mon=mon, scale=scale, insured=insured, exposed=exposed,
                resc=resc, n_resc=n_resc, resc_alone=resc_alone, n_alone=n_alone)


def point_estimates():
    t = pd.read_csv(f"{TABLES}/units_at_stake_state.csv")
    return {"growth": t.growth_need_baseline.sum(), "nhpd_lb": t.nhpd_unrescued_lb.sum(), "nhpd_ub": t.nhpd_unrescued_ub.sum(),
            "lihtc_adj": t.lihtc_leaving_adj.sum(), "lihtc_raw": t.lihtc_leaving.sum(), "mort_low": t.mortgage_lb.sum(), "mort_high": t.mortgage_ub.sum(),
            "growth_low": pd.read_csv(f"{TABLES}/units_at_stake_state.csv").growth_need_low.sum(), "growth_high": t.growth_need_high.sum()}


# ------------------------------------------------------------------ one draw of each piece
def _rescue_draw(rng, prep, X, y, Xq, ulb, uub):
    idx = rng.integers(0, len(y), len(y))
    m = P.make_model("logit", 0).fit(X[idx], y[idx])
    p = m.predict_proba(Xq)[:, 1]
    lost = rng.random(len(p)) >= p                  # True where the property is not rescued
    return float((ulb * lost).sum()), float((uub * lost).sum())


def _lihtc_draw(rng, prep):
    s = prep["sample"]
    bs = s.sample(len(s), replace=True, random_state=int(rng.integers(1e9)))
    m = L.StateLogit(CHOSEN_C, states=prep["final"].states).fit(bs)
    mon = prep["mon"]
    pb = pd.Series(m.predict(mon, pis_year=L.TREND_CENTER), index=mon.index)
    out = []
    for p31 in (pb * prep["scale"], pb):             # calibration-adjusted, then unadjusted
        rem = ((p31 - mon.obs_cell) / (1 - mon.obs_cell)).clip(lower=0, upper=1).to_numpy()
        out.append(float((mon.n_units.to_numpy() * (rng.random(len(rem)) < rem)).sum()))
    return tuple(out)


def _mortgage_draw(rng, prep):
    ins = prep["insured"]
    boot = ins.sample(len(ins), replace=True, random_state=int(rng.integers(1e9)))
    curve = M.cohort_curve(boot)
    e = prep["exposed"]
    units = e.units.to_numpy()
    out = []
    for shift, rate, n in ((7, prep["resc"], prep["n_resc"]), (9, prep["resc"], prep["n_resc"]), (11, prep["resc_alone"], prep["n_alone"])):
        r = rng.binomial(n, rate) / n
        s = e.start_year.clip(1965 + shift, 2024).astype(int)
        now, fut = curve.share_terminated.reindex(s).to_numpy(), curve.share_terminated.reindex(s - shift).to_numpy()
        add = np.clip((fut - now) / (1 - now), 0, 1)
        terminated = rng.random(len(add)) < add
        rescued = rng.random(len(add)) < r
        out.append(float((units * terminated * ~rescued).sum()))
    return tuple(out)


def simulate(n=500, seed=20261011, prep=None):
    prep = prep or prepare()
    sc = prep["scored"]
    X, y = prep["ev"][P.FINAL_COLS].astype(float).to_numpy(), prep["ev"].resub_3y.to_numpy()
    Xq, ulb, uub = sc[P.FINAL_COLS].astype(float).to_numpy(), sc.units_lb.to_numpy(), sc.units_ub.to_numpy()
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        r_lb, r_ub = _rescue_draw(rng, prep, X, y, Xq, ulb, uub)
        l_adj, l_raw = _lihtc_draw(rng, prep)
        m_low, m_mid, m_high = _mortgage_draw(rng, prep)
        rows.append({"nhpd_lb": r_lb, "nhpd_ub": r_ub, "lihtc_adj": l_adj, "lihtc_raw": l_raw, "mort_low": m_low, "mort_mid": m_mid, "mort_high": m_high})
    return pd.DataFrame(rows)


def totals(draws, pt):
    """Total in the low and high definitions for every draw, under the baseline growth need."""
    out = pd.DataFrame({"total_low_case": pt["growth"] + draws.nhpd_lb + draws.lihtc_adj + draws.mort_low,
                        "total_high_case": pt["growth"] + draws.nhpd_ub + draws.lihtc_raw + draws.mort_high})
    out["preservation_low_case"] = out.total_low_case - pt["growth"]
    out["preservation_high_case"] = out.total_high_case - pt["growth"]
    return out


# ------------------------------------------------------------------ sensitivity of the total to one assumption at a time
def _unrescued(scored, p):
    lost = 1 - p
    return float((scored.units_lb * lost).sum()), float((scored.units_ub * lost).sum())


def overlap_units(scored):
    """HOME and HUD-program units (expected not rescued) in scored properties that also hold an active LIHTC subsidy started 1996-2005.
    Those properties can appear in both the HOME and HUD piece and the LIHTC piece, so this is the largest the double count could be."""
    d = pd.read_csv(INTERIM / "nhpd_subsidies.csv.gz", usecols=["property_id", "program", "status", "start_year"])
    ids = set(d[(d.program == "LIHTC") & (d.status == "Active") & d.start_year.between(1996, 2005)].property_id)
    m = scored.property_id.isin(ids)
    lost = 1 - scored.p_rescue
    return float((scored.units_lb * lost)[m].sum()), float((scored.units_ub * lost)[m].sum()), int(m.sum())


def sensitivity(prep, pt, extra_rescue_cases=None):
    """Return a frame: one row per alternative, with the total in the low and high definitions and the change in the midpoint."""
    ev, sc = prep["ev"], prep["scored"]
    base_low = pt["growth"] + pt["nhpd_lb"] + pt["lihtc_adj"] + pt["mort_low"]
    base_high = pt["growth"] + pt["nhpd_ub"] + pt["lihtc_raw"] + pt["mort_high"]
    rows = [("Headline (as reported)", base_low, base_high)]

    def add(name, low, high):
        rows.append((name, low, high))

    add("Growth need: low scenario (renting keeps falling)", base_low - pt["growth"] + pt["growth_low"], base_high - pt["growth"] + pt["growth_low"])
    add("Growth need: high scenario (renting rises)", base_low - pt["growth"] + pt["growth_high"], base_high - pt["growth"] + pt["growth_high"])
    # unit definition
    add("Units: lower bound (largest assisted subsidy) in both cases", base_low, base_low - pt["nhpd_lb"] + pt["nhpd_lb"] + 0)
    rows.pop()
    low_ub = pt["growth"] + pt["nhpd_ub"] + pt["lihtc_adj"] + pt["mort_low"]
    add("HOME and HUD units: property total (upper bound) in both cases", low_ub, base_high)
    # rescue model choice
    y = ev.resub_3y.to_numpy()
    p_size = P.make_model("logit", 0).fit(ev[["log_total_units"]].astype(float), y).predict_proba(sc[["log_total_units"]].astype(float))[:, 1]
    lb, ub = _unrescued(sc, p_size)
    add("Rescue model: size only", pt["growth"] + lb + pt["lihtc_adj"] + pt["mort_low"], pt["growth"] + ub + pt["lihtc_raw"] + pt["mort_high"])
    cols = P.FEATURE_SETS["size + program"]
    p_boost = P.make_model("boosting", 0).fit(ev[cols].astype(float), y).predict_proba(sc[cols].astype(float))[:, 1]
    lb, ub = _unrescued(sc, p_boost)
    add("Rescue model: gradient boosting instead of logistic regression", pt["growth"] + lb + pt["lihtc_adj"] + pt["mort_low"], pt["growth"] + ub + pt["lihtc_raw"] + pt["mort_high"])
    p_recent, _, _ = D.recent_rate_probabilities(ev, sc)            # the rescue rate stays at its 2022-23 level (it rose after 2021)
    lb, ub = _unrescued(sc, p_recent)
    add("Rescue chance stays at its 2022-23 level", pt["growth"] + lb + pt["lihtc_adj"] + pt["mort_low"], pt["growth"] + ub + pt["lihtc_raw"] + pt["mort_high"])
    for name, p in (extra_rescue_cases or {}).items():
        lb, ub = _unrescued(sc, p)
        add(name, pt["growth"] + lb + pt["lihtc_adj"] + pt["mort_low"], pt["growth"] + ub + pt["lihtc_raw"] + pt["mort_high"])
    # LIHTC and mortgage choices
    add("LIHTC: calibration-adjusted exit share in both cases", base_low, base_low - pt["nhpd_lb"] + pt["nhpd_ub"] - pt["mort_low"] + pt["mort_high"])
    rows.pop()
    add("LIHTC: unadjusted exit share in both cases", base_low - pt["lihtc_adj"] + pt["lihtc_raw"], base_high)
    add("Insured mortgages: low case in both", base_low, base_high - pt["mort_high"] + pt["mort_low"])
    # double counting
    o_lb, o_ub, n_over = overlap_units(sc)
    add(f"Remove possible double count with LIHTC ({n_over} properties)", base_low - o_lb, base_high - o_ub)
    # contribution of each piece
    add("Drop the insured-mortgage piece", base_low - pt["mort_low"], base_high - pt["mort_high"])
    add("Drop the LIHTC piece", base_low - pt["lihtc_adj"], base_high - pt["lihtc_raw"])
    add("Drop the HOME and HUD piece", base_low - pt["nhpd_lb"], base_high - pt["nhpd_ub"])
    t = pd.DataFrame(rows, columns=["case", "total_low", "total_high"])
    t["midpoint"] = (t.total_low + t.total_high) / 2
    t["change_vs_headline"] = t.midpoint - t.midpoint.iloc[0]
    t["change_pct"] = 100 * t.change_vs_headline / t.midpoint.iloc[0]
    return t
