"""
Small panel helpers shared by the exploratory notebooks and the Part 2 models.

two_way_demean       remove state and time means (alternating projections), so correlations are WITHIN state and
                     year: they ask whether a state moves differently from its own norm and from the national
                     pattern, not whether states differ from each other
cluster_bootstrap_corr   correlation with a confidence interval from resampling whole states
vif                  variance inflation factors for a set of regressors
"""

import numpy as np
import pandas as pd


def two_way_demean(df, cols, state="state_fips", time="origin_year", iters=100, tol=1e-10):
    """Return the columns of df with state means and time means removed. Rows must be complete in cols."""
    x = df[cols].astype(float).copy()
    for _ in range(iters):
        before = x.to_numpy().copy()
        x = x - x.groupby(df[state]).transform("mean")
        x = x - x.groupby(df[time]).transform("mean")
        if np.abs(x.to_numpy() - before).max() < tol:
            break
    return x


def cluster_bootstrap_corr(x, y, groups, n=500, seed=20261006, level=0.95):
    """Pearson correlation of x and y with a percentile interval from resampling groups (states) with replacement."""
    x, y, groups = np.asarray(x, float), np.asarray(y, float), np.asarray(groups)
    point = np.corrcoef(x, y)[0, 1]
    ids = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in ids}
    rng = np.random.default_rng(seed)
    draws = np.empty(n)
    for b in range(n):
        pick = rng.choice(ids, size=len(ids), replace=True)
        rows = np.concatenate([index[g] for g in pick])
        draws[b] = np.corrcoef(x[rows], y[rows])[0, 1]
    lo, hi = np.percentile(draws, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return point, lo, hi


def vif(frame):
    """Variance inflation factor of each column against the others (with an intercept)."""
    x = frame.to_numpy(float)
    out = {}
    for j, c in enumerate(frame.columns):
        others = np.delete(x, j, axis=1)
        design = np.column_stack([np.ones(len(x)), others])
        beta, *_ = np.linalg.lstsq(design, x[:, j], rcond=None)
        resid = x[:, j] - design @ beta
        r2 = 1 - resid.var() / x[:, j].var()
        out[c] = 1 / (1 - r2)
    return pd.Series(out, name="VIF")


def pairwise_corr(wide, min_periods=5):
    """Pairwise correlations of the columns... of a (units x time) frame, over each pair's common periods.
    Returns (correlation matrix, matrix of common-period counts), both indexed by unit."""
    x = wide.T                                  # time x units
    corr = x.corr(min_periods=min_periods)
    notna = x.notna().astype(float)
    counts = notna.T @ notna
    return corr, counts


def pesaran_cd(wide, min_periods=5):
    """Pesaran (2004) CD test of cross-sectional dependence on a (units x time) frame of residuals, allowing an unbalanced panel:
        CD = sqrt(2 / (N (N - 1))) * sum over pairs i<j of sqrt(T_ij) * rho_ij
    Under independence CD is approximately standard normal. Returns a dict with CD, the two-sided p-value, the mean pairwise correlation,
    the mean absolute pairwise correlation, the number of units, and the mean number of common periods per pair."""
    from scipy import stats
    corr, counts = pairwise_corr(wide, min_periods)
    n = corr.shape[0]
    iu = np.triu_indices(n, k=1)
    rho, t = corr.to_numpy()[iu], counts.to_numpy()[iu]
    ok = ~np.isnan(rho)
    cd = np.sqrt(2.0 / (n * (n - 1))) * np.sum(np.sqrt(t[ok]) * rho[ok])
    return {"CD": cd, "p_value": 2 * (1 - stats.norm.cdf(abs(cd))), "mean_corr": rho[ok].mean(), "mean_abs_corr": np.abs(rho[ok]).mean(),
            "units": n, "mean_common_periods": t[ok].mean()}


def group_corr_test(wide, groups, min_periods=5, n_perm=2000, seed=20261007):
    """Is the mean pairwise correlation higher within groups (for example census divisions) than between them? Permutation test:
    group labels are shuffled across units. Returns within mean, between mean, difference, and the one-sided permutation p-value."""
    corr, _ = pairwise_corr(wide, min_periods)
    units = list(corr.index)
    g = pd.Series(groups).reindex(units).to_numpy()
    c = corr.to_numpy()
    iu = np.triu_indices(len(units), k=1)
    rho = c[iu]
    def stat(labels):
        same = labels[iu[0]] == labels[iu[1]]
        ok = ~np.isnan(rho)
        return rho[same & ok].mean(), rho[~same & ok].mean()
    w, b = stat(g)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_perm)
    for k in range(n_perm):
        pw, pb = stat(rng.permutation(g))
        diffs[k] = pw - pb
    return {"within": w, "between": b, "difference": w - b, "p_value": float((np.sum(diffs >= (w - b)) + 1) / (n_perm + 1))}
