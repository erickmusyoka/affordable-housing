"""
Example 1: a fair forecasting test. Fit only on the past, predict the future, and compare with "next year equals this year".

Run from the project root:   uv run examples/01_rolling_origin_backtest.py   (needs src/ on the path, so the script adds it)
Uses the project's test harness (src/evaluation.py): for each origin year it trains on rows whose target year is already known,
predicts 1 to 3 years ahead, and scores every forecast against what was later published. First origin 2016.
"""
import sys
sys.path.insert(0, "src")
import pandas as pd

from evaluation import run_backtest, summarize
from features import FEATURE_SETS, make_supervised
from models.baselines import Persistence, PooledAR1, ShrinkToStateMean

panel = pd.read_csv("data/processed/panel.csv", dtype={"state_fips": str})
TARGET = "severe_rent_burden_pct"
models = [Persistence(), PooledAR1(), ShrinkToStateMean(0.5)]       # the benchmark, one line through all states, and a half-way pull to each state's own average

frames = []
for horizon in (1, 2, 3):
    data = make_supervised(panel, TARGET, horizon, FEATURE_SETS["core"])     # one row per state and origin year, with only what was known then
    for m in models:
        frames.append(run_backtest(m, data, TARGET, horizon, first_origin=2016))
results = pd.concat(frames, ignore_index=True)

table = summarize(results)                                    # mean absolute error, the ratio to persistence (below 1 is better), and the share of origin years won
print(table[["horizon", "model", "origins", "mae", "mae_vs_persistence", "origins_won_vs_persistence"]].round(3).to_string(index=False))
