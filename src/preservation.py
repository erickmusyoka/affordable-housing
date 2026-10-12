"""
Part 5: subsidy preservation risk. When a property's last subsidy ends, is it re-subsidized within three years?

Event (one row per property): the property's last subsidy that ended with reason "End Date Passed" ended in year Y (2011 to 2023), and no other
subsidy covered the property at Y (another subsidy with start year <= Y that was still running or had no end date). Everything about the event
is built ONLY from subsidies that had ended by Y, from the property's size, and from state market conditions in year Y-1 or earlier.

Label (fixed horizon, so that every event has the same chance to be observed): `resub_3y` = 1 if the property has any subsidy whose start year is
in [Y, Y+3] and that runs past Y, as of the October 2026 snapshot. Events are restricted to Y <= 2023 so the whole window is observed.

Caveats that are part of the result, not footnotes:
    * "Re-subsidized" depends on how completely NHPD records new subsidies; small properties may be tracked less well.
    * HOME affordability periods end by design, and HOME makes up 94 percent of the history. LIHTC has few events (a 30-year clock only
      starts expiring from 2017) and RHS almost none, so a model trained here cannot speak for those programs.
    * Total units comes from the 2026 property file (static); the event's own subsidy units are used as given.
"""

import numpy as np
import pandas as pd

from common import INTERIM, PROCESSED

HORIZON = 3
FIRST_Y, LAST_Y = 2011, 2023
HUD_PROGRAMS = ["Section 8", "Section 202", "Section 236", "HUD Insured", "Mod Rehab", "Public Housing", "Project Based Vouchers"]
MARKET = ["rent_to_income_pct", "rental_vacancy_rate_pct", "rent_growth_pct", "severe_rent_burden_pct", "renter_share_pct", "unemployment_rate_pct"]
GROUPS = ["home_only", "hud", "lihtc", "rhs", "other"]


def program_group(programs):
    s = pd.Series(list(programs))
    if (s == "HOME").all():
        return "home_only"
    if s.str.startswith("RHS").any():
        return "rhs"
    if s.isin(HUD_PROGRAMS).any():
        return "hud"
    if (s == "LIHTC").any():
        return "lihtc"
    return "other"


def market_lagged():
    """State market variables carried forward from the last available year, then shifted one year (what was known entering Y)."""
    p = pd.read_csv(PROCESSED / "panel.csv", dtype={"state_fips": str})[["state_fips", "year"] + MARKET].sort_values(["state_fips", "year"])
    p[MARKET] = p.groupby("state_fips")[MARKET].ffill()
    p["year"] = p["year"] + 1
    return p.rename(columns={"year": "Y"})


def build_events():
    d = pd.read_csv(INTERIM / "nhpd_subsidies.csv.gz", dtype={"state_fips": str})
    props = pd.read_csv(INTERIM / "nhpd_properties.csv.gz", dtype={"state_fips": str, "county_fips": str, "tract_fips": str}).set_index("property_id")
    d["state_fips"] = d["state_fips"].str.zfill(2)
    ended = d[(d.status == "Inactive") & (d.inactive_reason == "End Date Passed") & d.end_year.between(FIRST_Y, LAST_Y)]
    Y = ended.groupby("property_id").end_year.max().rename("Y")
    sub = d.join(Y, on="property_id", how="inner")
    covered = sub[(sub.start_year <= sub.Y) & ((sub.end_year > sub.Y) | sub.end_year.isna())].property_id.unique()
    # label: any subsidy that starts in [Y, Y+3] and runs past Y (checked on all subsidies before restricting to the ended ones)
    new = sub[(sub.start_year >= sub.Y) & (sub.start_year <= sub.Y + HORIZON) & ((sub.end_year > sub.Y) | sub.end_year.isna())].property_id.unique()
    past = sub[(sub.end_year <= sub.Y) & ~sub.property_id.isin(covered)]
    g = past.groupby("property_id")
    ev = pd.DataFrame({
        "Y": g.Y.first(), "state_fips": g.state_fips.first(), "n_ended": g.size(),
        "assisted_units": g.assisted_units.max(), "start_year": g.start_year.min(), "end_year": g.end_year.max(),
        "group": g.program.apply(program_group),
    })
    ev["duration"] = ev.end_year - ev.start_year
    ev["total_units"] = props.reindex(ev.index).total_units
    ev["county_fips"] = props.reindex(ev.index).county_fips
    ev["log_total_units"] = np.log(ev.total_units.clip(lower=1))
    ev["log_assisted_units"] = np.log(ev.assisted_units.clip(lower=1))
    ev["resub_3y"] = ev.index.isin(new).astype(int)
    ev = ev.dropna(subset=["total_units", "assisted_units"]).reset_index()
    ev = ev.merge(market_lagged(), on=["state_fips", "Y"], how="left")
    for gname in GROUPS:
        ev[f"g_{gname}"] = (ev["group"] == gname).astype(int)
    return ev


FEATURES_PROPERTY = ["log_total_units", "log_assisted_units", "n_ended", "duration", "start_year"] + [f"g_{g}" for g in GROUPS if g != "home_only"]
FEATURES_MARKET = MARKET


def add_rent_gap(ev):
    """Attach the county ratio of the fair market rent to the rent affordable at 60 percent of median (2 bedrooms) from the HOME rent limits.
    Only FY2019 and FY2024 are held, so each event gets the closer vintage (events up to 2021 use FY2019, later ones FY2024). A county with no match
    stays missing. rent_gap_within is the deviation from the state mean (so state differences are removed), in national standard deviations."""
    h = pd.read_csv(INTERIM / "home_rent_limits_county.csv", dtype={"county_fips": str})
    h = h[["county_fips", "year", "fmr_to_rent60_2br"]]
    ev = ev.copy()
    ev["gap_year"] = np.where(ev["Y"] <= 2021, 2019, 2024)
    ev = ev.merge(h.rename(columns={"year": "gap_year", "fmr_to_rent60_2br": "rent_gap"}), on=["county_fips", "gap_year"], how="left")
    sd = ev["rent_gap"].std()
    # demeaned within state AND vintage, so differences between the FY2019 and FY2024 levels (and with them the time trend in the rescue rate) do not leak in
    ev["rent_gap_within"] = (ev["rent_gap"] - ev.groupby(["state_fips", "gap_year"])["rent_gap"].transform("mean")) / sd
    return ev


def add_price_growth(ev, years=5):
    """Attach the county house price growth over the `years` years ending in Y-1 (known entering the event year Y, FHFA all-transactions index).
    price_growth_rel is the growth relative to the mean of all matched events in the same event year (so the national price cycle drops out, as in
    Begley and Reina), then demeaned within state, in national standard deviations of the relative growth. A county with no index stays missing."""
    h = pd.read_csv(INTERIM / "fhfa_hpi_county.csv", dtype={"county_fips": str})[["county_fips", "year", f"growth_{years}y_pct"]]
    ev = ev.copy()
    ev["hpi_year"] = ev["Y"] - 1
    ev = ev.merge(h.rename(columns={"year": "hpi_year", f"growth_{years}y_pct": "price_growth"}), on=["county_fips", "hpi_year"], how="left")
    rel = ev["price_growth"] - ev.groupby("Y")["price_growth"].transform("mean")
    ev["price_growth_rel"] = rel / rel.std()
    ev["price_growth_within"] = (rel - rel.groupby(ev["state_fips"]).transform("mean")) / rel.std()
    return ev


SPLITS = {"main": (2015, 2019), "robust": (2017, 2020)}      # (last training Y, first test Y); the gap keeps label windows from overlapping


def split(ev, name="main"):
    last_train, first_test = SPLITS[name]
    return ev[ev.Y <= last_train].copy(), ev[ev.Y >= first_test].copy()


# ---------------------------------------------------------------- models and evaluation
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_model(kind, seed=0):
    if kind == "logit":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000))
    if kind == "forest":
        return RandomForestClassifier(n_estimators=400, min_samples_leaf=10, max_features=0.5, class_weight="balanced_subsample", n_jobs=4, random_state=seed)
    if kind == "boosting":
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.04, max_iter=150, min_samples_leaf=20, l2_regularization=2.0, random_state=seed)
    raise ValueError(kind)


FEATURE_SETS = {
    "size only": ["log_total_units"],
    "program only": [f"g_{g}" for g in GROUPS if g != "home_only"],
    "size + program": ["log_total_units"] + [f"g_{g}" for g in GROUPS if g != "home_only"],
    "property": FEATURES_PROPERTY,
    "market only": FEATURES_MARKET,
    "property + market": FEATURES_PROPERTY + FEATURES_MARKET,
}


def _prep(train, test, cols, kind):
    xtr, xte = train[cols].astype(float), test[cols].astype(float)
    if kind != "boosting":
        med = xtr.median()
        xtr, xte = xtr.fillna(med), xte.fillna(med)
    return xtr, xte


def fit_predict(train, test, cols, kind, seed=0):
    xtr, xte = _prep(train, test, cols, kind)
    m = make_model(kind, seed).fit(xtr, train.resub_3y)
    return m.predict_proba(xte)[:, 1]


def bootstrap_metric(y, p, fn, n=500, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(y):
            out.append(fn(y[i], p[i]))
    return np.percentile(out, [5, 95])


def capture(y, p, top=0.2):
    """Share of all re-subsidized properties found in the top `top` share of properties ranked by predicted probability. Ties at the cut-off are shared
    proportionally (the expected result under random tie-breaking), so models with few distinct scores do not depend on row order."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    k = max(int(round(len(y) * top)), 1)
    thr = np.sort(p)[::-1][k - 1]
    above, tied = p > thr, p == thr
    need = k - above.sum()
    return (y[above].sum() + y[tied].sum() * need / tied.sum()) / y.sum()


# ---------------------------------------------------------------- forward application to active properties
FINAL_COLS = FEATURE_SETS["size + program"]


def active_expiring(first=2026, last=2035):
    """Active properties whose LAST active subsidy ends in [first, last] (the policy metric of Section 6), with the same features as the events."""
    d = pd.read_csv(INTERIM / "nhpd_subsidies.csv.gz", dtype={"state_fips": str})
    p = pd.read_csv(INTERIM / "nhpd_properties.csv.gz", dtype={"state_fips": str})
    p["state_fips"] = p["state_fips"].str.zfill(2)
    p["has_indefinite"] = p.has_indefinite.fillna(False).astype(bool)
    q = p[(p.property_status == "Active") & p.last_end_year.between(first, last) & ~p.has_indefinite].copy()
    act = d[(d.status == "Active")]
    q["group"] = q.property_id.map(act.groupby("property_id").program.apply(program_group))
    q["log_total_units"] = np.log(q.total_units.clip(lower=1))
    for g in GROUPS:
        q[f"g_{g}"] = (q["group"] == g).astype(int)
    return q


def rescue_projection(ev, modeled=("home_only", "hud"), seed=0):
    """Fit the size + program logit on every event and score the active expiring properties. Returns the scored frame; only the groups with
    enough history (`modeled`) get a probability, the rest are left missing on purpose."""
    cols = FINAL_COLS
    xtr = ev[cols].astype(float)
    m = make_model("logit", seed).fit(xtr, ev.resub_3y)
    q = active_expiring()
    q["p_rescue"] = np.nan
    ok = q["group"].isin(modeled)
    q.loc[ok, "p_rescue"] = m.predict_proba(q.loc[ok, cols].astype(float))[:, 1]
    return q, m
