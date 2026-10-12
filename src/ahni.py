"""
Affordable Housing Need Index (AHNI): a measure of CURRENT need for affordable rental housing, by state.

The AHNI is a summary measurement, not a forecast, and is never predicted from its own components (CLAUDE.md Section 2).

Three dimensions, seven components. Every component is oriented so that HIGHER MEANS MORE NEED.
    burden    ACS severe rent burden (2022-24 mean), ACS rent burden 30 percent or more (2022-24 mean),
              CHAS severe burden among renters at or below 50 percent of area median income (2018-2022)
    gap       CHAS affordable units per 100 extremely low income renters (2018-2022, inverted),
              ACS rental vacancy rate (2022-24 mean, inverted)
    pressure  Rent as a share of income (ACS 2022-24 mean), rent growth minus income growth 2021 to 2024

Homelessness (PIT) is kept out of the index and used to validate it. Assistance coverage, expiring affordability, and
permit pace are the other policy metrics and are reported beside the index, not inside it.

Each component carries a standard error (90 percent margin of error divided by 1.645), propagated by the Census
formulas (proportion and ratio) or the delta method, and pooled across years assuming independent annual samples. A
Monte Carlo redraw of every component within its error gives rank intervals.

Scores: each component is converted to a z-score across the 51 states, capped at plus or minus 3, so one extreme state
(DC, Florida) cannot dominate. A dimension score is the mean of its components; the index is the weighted mean of the
dimension scores. Four weighting schemes are computed; the headline is equal weights across the three dimensions.
"""

import numpy as np
import pandas as pd

from common import RAW, STATE_FIPS

Z90 = 1.645
WINDOW = [2022, 2023, 2024]
BASE_WINDOW = [2010, 2011, 2012]
ABBR = {v: k for k, v in STATE_FIPS.items()}

COMPONENTS = {
    "severe_burden_acs": ("burden", +1, "ACS severe rent burden, %, mean 2022-24"),
    "burden30_acs": ("burden", +1, "ACS rent burden 30%+, %, mean 2022-24"),
    "severe_burden_chas_le50": ("burden", +1, "CHAS severe burden, renters at or below 50% HAMFI, %"),
    "affordable_units_per100_eli": ("gap", -1, "CHAS affordable units per 100 extremely low income renters (inverted)"),
    "vacancy_rate": ("gap", -1, "ACS rental vacancy rate, %, mean 2022-24 (inverted)"),
    "rent_to_income": ("pressure", +1, "ACS rent as % of income, mean 2022-24"),
    "rent_minus_income_growth": ("pressure", +1, "Rent growth minus income growth, 2021 to 2024, points"),
}
DIMENSIONS = ["burden", "gap", "pressure"]
SCHEMES = {
    "equal dimensions (headline)": {"burden": 1 / 3, "gap": 1 / 3, "pressure": 1 / 3},
    "expert (burden 50, gap 30, pressure 20)": {"burden": 0.5, "gap": 0.3, "pressure": 0.2},
}


def _pooled(est, se):
    """Mean over years and its standard error, assuming independent annual samples. Both are (state x year) frames."""
    n = est.notna().sum(axis=1)
    return est.mean(axis=1), np.sqrt((se ** 2).sum(axis=1, min_count=1)) / n


def _ratio_moe(a, a_moe, b, b_moe):
    r = a / b
    return (1 / b) * np.sqrt(a_moe ** 2 + r ** 2 * b_moe ** 2)


def _prop_moe(num, num_moe, den, den_moe):
    p = num / den
    rad = (num_moe ** 2 - p ** 2 * den_moe ** 2).where(lambda x: x >= 0, num_moe ** 2 + p ** 2 * den_moe ** 2)
    return np.sqrt(rad) / den


def _acs_wide(panel, col):
    return panel[panel.acs_available == 1].pivot(index="state_abbr", columns="year", values=col)


def chas_components():
    """CHAS 2018-2022 severe burden (renters at or below 50 percent HAMFI) and affordable units per 100 ELI renters,
    each with a standard error from the published margins of error."""
    t8 = pd.read_csv(RAW / "chas" / "2018-2022" / "Table8.csv")
    t15 = pd.read_csv(RAW / "chas" / "2018-2022" / "Table15C.csv")
    for t in (t8, t15):
        t.columns = [c.lower() for c in t.columns]
    t8, t15 = t8[t8.st != 72].set_index("st"), t15[t15.st != 72].set_index("st")
    e = lambda n: t8[f"t8_est{n}"].astype(float)
    m = lambda n: t8[f"t8_moe{n}"].astype(float)
    num = e(76) + e(89)
    den = e(69) + e(82) - e(79) - e(92)
    num_moe = np.sqrt(m(76) ** 2 + m(89) ** 2)
    den_moe = np.sqrt(m(69) ** 2 + m(82) ** 2 + m(79) ** 2 + m(92) ** 2)
    severe = 100 * num / den
    severe_moe = 100 * _prop_moe(num, num_moe, den, den_moe)

    units, units_moe = t15["t15c_est4"].astype(float), t15["t15c_moe4"].astype(float)
    eli, eli_moe = e(69), m(69)
    aff = 100 * units / eli
    aff_moe = 100 * _ratio_moe(units, units_moe, eli, eli_moe)
    out = pd.DataFrame({"severe_burden_chas_le50": severe, "severe_burden_chas_le50_se": severe_moe / Z90,
                        "affordable_units_per100_eli": aff, "affordable_units_per100_eli_se": aff_moe / Z90})
    out.index = [ABBR[f"{s:02d}"] for s in out.index]
    return out


def build_components(panel):
    """Return a frame indexed by state abbreviation with one est and one se column per component."""
    out = pd.DataFrame(index=sorted(STATE_FIPS))
    w = lambda c: _acs_wide(panel, c)[WINDOW]
    for key, col in (("severe_burden_acs", "severe_rent_burden_pct"), ("burden30_acs", "rent_burden_30_pct"),
                     ("vacancy_rate", "rental_vacancy_rate_pct")):
        est, se = _pooled(w(col), w(col + "_moe") / Z90)
        out[key], out[key + "_se"] = est, se

    rent, rent_se = w("median_gross_rent"), w("median_gross_rent_moe") / Z90
    inc, inc_se = w("median_hh_income"), w("median_hh_income_moe") / Z90
    ri = 1200 * rent / inc
    ri_se = ri * np.sqrt((rent_se / rent) ** 2 + (inc_se / inc) ** 2)      # delta method, covariance ignored
    out["rent_to_income"], out["rent_to_income_se"] = _pooled(ri, ri_se)

    full = lambda c: _acs_wide(panel, c)
    r, i = full("median_gross_rent"), full("median_hh_income")
    rs, is_ = full("median_gross_rent_moe") / Z90, full("median_hh_income_moe") / Z90
    rg = 100 * (r[2024] / r[2021] - 1)
    ig = 100 * (i[2024] / i[2021] - 1)
    rg_se = 100 * (r[2024] / r[2021]) * np.sqrt((rs[2024] / r[2024]) ** 2 + (rs[2021] / r[2021]) ** 2)
    ig_se = 100 * (i[2024] / i[2021]) * np.sqrt((is_[2024] / i[2024]) ** 2 + (is_[2021] / i[2021]) ** 2)
    out["rent_minus_income_growth"] = rg - ig
    out["rent_minus_income_growth_se"] = np.sqrt(rg_se ** 2 + ig_se ** 2)

    ch = chas_components()
    for c in ("severe_burden_chas_le50", "affordable_units_per100_eli"):
        out[c], out[c + "_se"] = ch[c], ch[c + "_se"]
    return out[[x for k in COMPONENTS for x in (k, k + "_se")]]


def zscores(est, cap=3.0):
    """Oriented z-scores across states: positive always means more need."""
    z = pd.DataFrame(index=est.index)
    for k, (_, sign, _) in COMPONENTS.items():
        v = est[k]
        z[k] = (sign * (v - v.mean()) / v.std(ddof=1)).clip(-cap, cap)
    return z


def dimension_scores(z):
    return pd.DataFrame({d: z[[k for k, (dim, _, _) in COMPONENTS.items() if dim == d]].mean(axis=1) for d in DIMENSIONS})


def index_from_z(z, weights):
    return (dimension_scores(z) * pd.Series(weights)).sum(axis=1)


def component_weights_equal(z):
    """Each of the seven components weighted equally (differs from equal dimensions because gap and pressure have two)."""
    return z.mean(axis=1)


def pca_scores(z):
    """First principal component of the seven oriented z-scores, signed so that more need is a higher score."""
    x = z.to_numpy() - z.to_numpy().mean(axis=0)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    load = vt[0] if vt[0].sum() > 0 else -vt[0]
    share = (s ** 2 / (s ** 2).sum())[0]
    return pd.Series(x @ load, index=z.index), pd.Series(load, index=z.columns), share


def all_schemes(z):
    out = {name: index_from_z(z, w) for name, w in SCHEMES.items()}
    out["equal components"] = component_weights_equal(z)
    pc, load, share = pca_scores(z)
    out["first principal component"] = pc
    return pd.DataFrame(out), load, share


def rank_of(score):
    """1 = highest need."""
    return score.rank(ascending=False, method="min").astype(int)


def monte_carlo(est_frame, n=1000, seed=20261006, weights=None):
    """Redraw every component within its standard error, rebuild the index, and return ranks (n x states)."""
    weights = weights or SCHEMES["equal dimensions (headline)"]
    rng = np.random.default_rng(seed)
    base = est_frame[list(COMPONENTS)]
    se = est_frame[[k + "_se" for k in COMPONENTS]].to_numpy()
    ranks = np.empty((n, len(base)), dtype=int)
    for t in range(n):
        draw = pd.DataFrame(base.to_numpy() + rng.standard_normal(base.shape) * se, index=base.index, columns=base.columns)
        ranks[t] = rank_of(index_from_z(zscores(draw), weights)).to_numpy()
    return pd.DataFrame(ranks, columns=base.index)


def leave_one_out(est_frame, headline_rank):
    rows = []
    for drop in COMPONENTS:
        keep = [k for k in COMPONENTS if k != drop]
        z = zscores(est_frame)[keep]
        dims = pd.DataFrame({d: z[[k for k in keep if COMPONENTS[k][0] == d]].mean(axis=1) for d in DIMENSIONS if any(COMPONENTS[k][0] == d for k in keep)})
        score = dims.mean(axis=1)
        r = rank_of(score)
        rows.append({"dropped": drop, "rank correlation with headline": r.corr(headline_rank, method="spearman"),
                     "largest rank change": int((r - headline_rank).abs().max()), "states in top 10 unchanged": int(len(set(r[r <= 10].index) & set(headline_rank[headline_rank <= 10].index)))})
    return pd.DataFrame(rows)


def momentum(panel):
    """Change in need between 2010-12 and 2022-24 from three ACS components (severe burden, burden 30+, rent to income).
    Positive means need rose. Kept apart from the level index."""
    def avg(col, years):
        return _acs_wide(panel, col)[years].mean(axis=1)
    chg = pd.DataFrame({
        "severe_burden": avg("severe_rent_burden_pct", WINDOW) - avg("severe_rent_burden_pct", BASE_WINDOW),
        "burden30": avg("rent_burden_30_pct", WINDOW) - avg("rent_burden_30_pct", BASE_WINDOW),
        "rent_to_income": avg("rent_to_income_pct", WINDOW) - avg("rent_to_income_pct", BASE_WINDOW),
    })
    z = (chg - chg.mean()) / chg.std(ddof=1)
    return chg, z.mean(axis=1).rename("momentum")
