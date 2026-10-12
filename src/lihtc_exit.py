"""
Part 5b: when the LIHTC clock runs out, does a project leave the agency's monitoring list?

Outcome: `left_monitoring` = the allocating agency no longer monitors the project for LIHTC compliance (NONPROG = 1 in the 2024 HUD database). The
dictionary says this does NOT mean the project stopped being affordable, and the field was reported informally before 2012, so this is "left the
monitoring list", the observable signal that affordability restrictions are no longer being checked.

Sample: projects allocated in 1990 or later (the 30-year regime) with a placed-in-service year from 1990 to 1995, observed in 2024 at ages 29 to 34.
Projects placed later are scored as if they behave like these cohorts when they reach the same age.

Model: logit(P) = state effect + cohort trend, with an L2 penalty that shrinks small states toward the national level. A first look found that project
characteristics (size, sponsor type, credit type, location type, financing) add nothing across states (leave-states-out AUC 0.52); the signal is the
state, which could be state policy or state reporting practice and cannot be separated here.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import INTERIM

TREND_CENTER = 1993          # cohort at the middle of the observed ages
TREND_SCALE = 3.0            # the trend is scaled up so the penalty barely shrinks it
TRAIN_COHORTS = (1990, 1995)


def load_projects():
    p = pd.read_csv(INTERIM / "lihtc_projects.csv.gz", dtype={"state_fips": str})
    p = p[(p.alloc_pre1990 == 0) & p.pis_year.notna() & p.n_units.notna()].copy()
    p["pis_year"] = p.pis_year.astype(int)
    return p


class StateLogit:
    """Penalized logit with one effect per state and a linear cohort trend."""

    def __init__(self, C=1.0, states=None):
        self.C = C
        self.states = list(states) if states is not None else None

    def _x(self, df):
        d = pd.get_dummies(pd.Categorical(df.state_fips, categories=self.states), dtype=float).to_numpy()
        trend = (df.pis_year.to_numpy(float) - TREND_CENTER)[:, None] * TREND_SCALE / 3.0
        return np.hstack([trend, d])

    def fit(self, df):
        if self.states is None:
            self.states = sorted(df.state_fips.unique())
        self.m = LogisticRegression(C=self.C, max_iter=5000).fit(self._x(df), df.left_monitoring)
        return self

    def predict(self, df, pis_year=None):
        d = df if pis_year is None else df.assign(pis_year=pis_year)
        return self.m.predict_proba(self._x(d))[:, 1]


def cohort_split_eval(p, C=1.0):
    """Train on 1990-92, test on 1993-95: project-level AUC, calibration, and the weighted state-level error against the national-mean baseline."""
    tr, te = p[p.pis_year.between(1990, 1992)], p[p.pis_year.between(1993, 1995)]
    m = StateLogit(C).fit(tr)
    pr = m.predict(te)
    base = np.full(len(te), tr.left_monitoring.mean())
    t = te.assign(p=pr)
    g = t.groupby("state_fips").agg(obs=("left_monitoring", "mean"), pred=("p", "mean"), n=("p", "size"))
    g = g[g.n >= 15]
    # national baseline with the same trend: logit with only a trend
    tr_nat = tr.assign(state_fips="all"); te_nat = te.assign(state_fips="all")
    nat = StateLogit(C, states=["all"]).fit(tr_nat).predict(te_nat)
    gn = te.assign(p=nat).groupby("state_fips").p.mean().reindex(g.index)
    return {"auc": roc_auc_score(te.left_monitoring, pr), "mean_pred": pr.mean(), "mean_obs": te.left_monitoring.mean(), "states": len(g),
            "state_mae": np.average((g.obs - g.pred).abs(), weights=g.n), "state_mae_national_trend": np.average((g.obs - gn).abs(), weights=g.n),
            "state_corr": np.corrcoef(g.obs, g.pred)[0, 1], "table": g}


def project_future(p_all, model, first=1996, last=2005):
    """Projects placed first..last that are still monitored, with the share expected to have left monitoring by age ~31 (cohort 1993's age)."""
    f = p_all[p_all.pis_year.between(first, last)].copy()
    f["p31"] = model.predict(f, pis_year=TREND_CENTER)
    return f
