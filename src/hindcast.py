"""
Hindcast of the rescue model at state level (reviewer question: does the model get the aggregate right, not just the ranking of properties?).

Train on lapses through 2015 (as in Part 5), predict every lapse from 2019 to 2023, and compare predicted with actual outcomes in units, by state.
Two benchmarks guard against flattering the model:
    units only    the number of lapsing units, with no rescue model (what "units x any constant rate" gives for ranking states)
    constant rate the training base rate applied to every property (right ranking of properties by size, wrong level)
Quantities:
    unrescued units  what the headline reports; dominated by units that are not rescued under any model, so easy to match
    rescued units    the part the model actually predicts
    rescued share    rescued units over lapsing units, by state: does the model explain differences between states?
"""

import numpy as np
import pandas as pd

import preservation as P

SIZE_BINS = [0, 10, 25, 50, 100, 200, 1e6]


def run(split="main"):
    ev = P.build_events()
    tr, te = P.split(ev, split)
    cols = P.FINAL_COLS
    m = P.make_model("logit", 0).fit(tr[cols].astype(float), tr.resub_3y)
    te = te.copy()
    te["p"] = m.predict_proba(te[cols].astype(float))[:, 1]
    return tr, te


def state_table(te, base_rate, units="total_units"):
    d = te.assign(u=te[units])
    d["unrescued_pred"], d["unrescued_act"] = d.u * (1 - d.p), d.u * (1 - d.resub_3y)
    d["rescued_pred"], d["rescued_act"], d["rescued_const"] = d.u * d.p, d.u * d.resub_3y, d.u * base_rate
    g = d.groupby("state_fips").agg(events=("p", "size"), rescued_events=("resub_3y", "sum"), units=("u", "sum"), unrescued_pred=("unrescued_pred", "sum"),
                                   unrescued_act=("unrescued_act", "sum"), rescued_pred=("rescued_pred", "sum"), rescued_act=("rescued_act", "sum"), rescued_const=("rescued_const", "sum"))
    g["share_pred"], g["share_act"] = g.rescued_pred / g.units, g.rescued_act / g.units
    return g


def by_year(te):
    return te.groupby("Y").agg(predicted=("p", "mean"), observed=("resub_3y", "mean"), events=("p", "size"), rescued=("resub_3y", "sum"))


def by_size(te):
    return te.groupby(pd.cut(te.total_units, SIZE_BINS), observed=True).agg(predicted=("p", "mean"), observed=("resub_3y", "mean"), events=("p", "size"), rescued=("resub_3y", "sum"))


def summary(tr, te):
    base = tr.resub_3y.mean()
    out = {"train_events": len(tr), "test_events": len(te), "train_rate": base, "test_rate": te.resub_3y.mean(), "mean_predicted": te.p.mean()}
    for label, units in (("upper", "total_units"), ("lower", "assisted_units")):
        g = state_table(te, base, units)
        out[f"unrescued_pred_{label}"], out[f"unrescued_act_{label}"] = g.unrescued_pred.sum(), g.unrescued_act.sum()
    g = state_table(te, base)
    out.update({"states": len(g), "unrescued_pearson_model": g.unrescued_pred.corr(g.unrescued_act), "unrescued_pearson_units_only": g.units.corr(g.unrescued_act),
                "rescued_total_act": g.rescued_act.sum(), "rescued_total_model": g.rescued_pred.sum(), "rescued_total_const": g.rescued_const.sum(),
                "rescued_pearson_model": g.rescued_act.corr(g.rescued_pred), "rescued_pearson_units_only": g.rescued_act.corr(g.units),
                "rescued_spearman_model": g.rescued_act.corr(g.rescued_pred, method="spearman"), "rescued_spearman_units_only": g.rescued_act.corr(g.units, method="spearman")})
    big = g[g.events >= 40]
    out.update({"states_40plus": len(big), "share_pearson": big.share_act.corr(big.share_pred), "share_spearman": big.share_act.corr(big.share_pred, method="spearman")})
    return out
