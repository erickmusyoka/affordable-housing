"""
Example 12: does the rescue model get the aggregate right? A hindcast on 2019-2023 lapses.

Run from the project root:   uv run examples/12_state_hindcast.py
Train on lapses through 2015, predict every lapse from 2019 to 2023, and compare predicted with actual units, by state.
"""
import sys
sys.path.insert(0, "src")
import hindcast as H

train, test = H.run("main")                       # test has a column p: the predicted chance of rescue
S = H.summary(train, test)

print(f"test lapses: {S['test_events']:,}; observed rescue rate {100 * S['test_rate']:.1f} %, mean predicted {100 * S['mean_predicted']:.1f} %")
print(f"\nunrescued units: predicted {S['unrescued_pred_upper']:,.0f}, actual {S['unrescued_act_upper']:,.0f}")
print(f"  across {S['states']} states, correlation with actual: model {S['unrescued_pearson_model']:.3f}, lapsing units alone {S['unrescued_pearson_units_only']:.3f}")
print(f"\nrescued units: predicted {S['rescued_total_model']:,.0f}, actual {S['rescued_total_act']:,.0f}")
print(f"  across states, correlation with actual: model {S['rescued_pearson_model']:.2f}, lapsing units alone {S['rescued_pearson_units_only']:.2f}")
print(f"  rescued SHARE by state ({S['states_40plus']} states with 40+ lapses), predicted against actual: {S['share_pearson']:.2f}")

print("\npredicted and observed rescue rate (%) by lapse year:")
print((100 * H.by_year(test)[["predicted", "observed"]]).round(1).to_string())
