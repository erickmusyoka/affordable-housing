"""
The rescue rate is not constant: 1.1 percent of 2011-15 lapses were re-subsidized within three years, 0.9 percent of 2016-20 lapses, and 3.2 percent of
2021-23 lapses (3.9 percent in 2022 and 4.5 percent in 2023). Is that a change in the mix of properties, can a model that is updated as time passes
anticipate it, and what would the headline be if the recent rate persisted?

Real-time test: to predict lapses in year T, a forecaster knows only the outcomes of lapses through T-3 (a lapse in year Y needs until Y+3 to be labeled).
Methods compared, each refit for every T from 2019 to 2023:
    static             logistic regression on size and program, trained on every labeled earlier lapse
    recency weighted   the same, with weights that halve every two years of age
    intercept shift    the static model with its intercept moved so that it matches the rescue rate of the two latest labeled years
    year trend         the static model plus a linear event-year term, extrapolated
    constant rate      the rescue rate of the two latest labeled years, applied to every property
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

import preservation as P

COLS = P.FINAL_COLS
EPS = 1e-9


def _fit(train, weights=None, extra=()):
    m = P.make_model("logit", 0)
    X = train[list(COLS) + list(extra)].astype(float)
    if weights is None:
        return m.fit(X, train.resub_3y)
    return m.fit(X, train.resub_3y, logisticregression__sample_weight=weights)


def shift_to_rate(p, target):
    """Log-odds shift d such that the mean of expit(logit(p) + d) equals `target`."""
    z = logit(np.clip(np.asarray(p, float), EPS, 1 - EPS))
    return brentq(lambda d: expit(z + d).mean() - target, -8, 8)


def real_time(ev, first=2019, last=2023):
    rows = []
    for T in range(first, last + 1):
        te, tr = ev[ev.Y == T], ev[ev.Y <= T - 3]
        y = te.resub_3y.to_numpy()
        m = _fit(tr)
        p0 = m.predict_proba(te[COLS].astype(float))[:, 1]
        recent = tr[tr.Y >= T - 4]
        d = shift_to_rate(m.predict_proba(recent[COLS].astype(float))[:, 1], recent.resub_3y.mean())
        preds = {"static": p0,
                 "recency weighted": _fit(tr, (0.5 ** ((T - 3 - tr.Y) / 2.0)).to_numpy()).predict_proba(te[COLS].astype(float))[:, 1],
                 "intercept shift": expit(logit(np.clip(p0, EPS, 1 - EPS)) + d),
                 "constant rate": np.full(len(te), recent.resub_3y.mean())}
        tr2, te2 = tr.assign(trend=tr.Y - 2011.0), te.assign(trend=T - 2011.0)
        preds["year trend"] = P.make_model("logit", 0).fit(tr2[list(COLS) + ["trend"]].astype(float), tr2.resub_3y).predict_proba(te2[list(COLS) + ["trend"]].astype(float))[:, 1]
        for k, p in preds.items():
            p = np.clip(p, 1e-6, 1 - 1e-6)
            rows.append({"year": T, "method": k, "events": len(te), "observed": y.mean(), "predicted": p.mean(), "log_loss": log_loss(y, p, labels=[0, 1]),
                         "brier": brier_score_loss(y, p), "auc": roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan})
    return pd.DataFrame(rows)


def summary(r):
    return r.groupby("method").apply(lambda d: pd.Series({"log_loss": np.average(d.log_loss, weights=d.events), "brier": np.average(d.brier, weights=d.events),
                                                           "mean_abs_calibration_error": np.average((d.predicted - d.observed).abs(), weights=d.events)}), include_groups=False)


def recent_rate_probabilities(ev, scored, years=(2022, 2023)):
    """Rescue probabilities for the scored properties if the rescue rate stays at its 2022-23 level: the fitted logistic regression with its
    intercept moved so that it reproduces the observed 2022-23 rate on those lapses."""
    m = P.make_model("logit", 0).fit(ev[COLS].astype(float), ev.resub_3y)
    recent = ev[ev.Y.isin(years)]
    d = shift_to_rate(m.predict_proba(recent[COLS].astype(float))[:, 1], recent.resub_3y.mean())
    p = m.predict_proba(scored[COLS].astype(float))[:, 1]
    return expit(logit(np.clip(p, EPS, 1 - EPS)) + d), d, recent.resub_3y.mean()
