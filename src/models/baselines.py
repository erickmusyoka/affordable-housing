"""Baseline forecasters. Persistence is the benchmark every other model must beat (CLAUDE.md Section 3)."""

import numpy as np

from evaluation import Forecaster


class Persistence(Forecaster):
    """Next value equals this value: the forecast is the target at the origin."""
    name = "persistence"

    def predict(self, test):
        return test["y_origin"].to_numpy()


class Drift(Forecaster):
    """Origin value plus the average h-year change observed in the training rows (pooled over states)."""
    name = "drift"

    def fit(self, train):
        self.mean_change = float((train["y_target"] - train["y_origin"]).mean())
        return self

    def predict(self, test):
        return test["y_origin"].to_numpy() + self.mean_change


class StateMeanChange(Forecaster):
    """Origin value plus each state's own average h-year change in training (a state-specific drift)."""
    name = "state_drift"

    def fit(self, train):
        d = (train["y_target"] - train["y_origin"]).groupby(train["state_fips"]).mean()
        self.by_state = d
        self.fallback = float(d.mean())
        return self

    def predict(self, test):
        return test["y_origin"].to_numpy() + test["state_fips"].map(self.by_state).fillna(self.fallback).to_numpy()


class HistoryMean(Forecaster):
    """Each state's mean target over the training rows (what persistence must beat for a persistent series)."""
    name = "history_mean"

    def fit(self, train):
        self.by_state = train.groupby("state_fips")["y_target"].mean()
        self.fallback = float(train["y_target"].mean())
        return self

    def predict(self, test):
        return test["state_fips"].map(self.by_state).fillna(self.fallback).to_numpy()


class PooledAR1(Forecaster):
    """Pooled least squares of the target on its origin value: y_target = a + b * y_origin. A sanity check of the
    harness, not a Part 2 model."""
    name = "pooled_ar1"

    def fit(self, train):
        X = np.column_stack([np.ones(len(train)), train["y_origin"].to_numpy()])
        self.coef, *_ = np.linalg.lstsq(X, train["y_target"].to_numpy(), rcond=None)
        return self

    def predict(self, test):
        return self.coef[0] + self.coef[1] * test["y_origin"].to_numpy()


class ShrinkToStateMean(Forecaster):
    """(1 - w) * origin value + w * the state's own mean target over the training rows. w = 0 is persistence and w = 1
    is the state history mean. Tests whether the mean reversion seen within states is usable out of sample."""

    def __init__(self, weight):
        self.weight = weight
        self.name = f"shrink_{weight:.2f}"

    def fit(self, train):
        self.by_state = train.groupby("state_fips")["y_target"].mean()
        self.fallback = float(train["y_target"].mean())
        return self

    def predict(self, test):
        m = test["state_fips"].map(self.by_state).fillna(self.fallback).to_numpy()
        return (1 - self.weight) * test["y_origin"].to_numpy() + self.weight * m
