import numpy as np
import pandas as pd
import pytest

from fabguard import spc
from fabguard.fleet import reliability, simulate
from fabguard.maintenance import FEATURES, labels
from fabguard.model import Recipe, response, wafer
from fabguard.optimize import expected_improvement, search


def test_matching_confirms_on_new_wafers():
    from fabguard.matching import experiment

    calibration, confirmation, results = experiment(seed=7, n=60)
    assert len(calibration) == 240 and len(confirmation) == 480
    assert results.after_mean_nm.max() - results.after_mean_nm.min() < 3
    assert results.before_mean_nm.max() - results.before_mean_nm.min() > 20
    assert results.after_equivalent_3nm.all()


def test_constant_subgroup_means_have_undefined_correlation():
    baseline = spc.fit(np.tile([498, 499, 500, 501, 502], (30, 1)))
    assert baseline.lag1 is None
    assert baseline.sigma > 0


def test_nominal_units_and_fault_direction():
    nominal = response()
    assert nominal["rate_nm_min"] == pytest.approx(250)
    assert nominal["depth_nm"] == pytest.approx(500)
    assert response(fault="rf_loss")["rate_nm_min"] < 250
    assert response(age=1)["defect_probability"] > nominal["defect_probability"]
    assert response(fault="sensor_bias")["depth_nm"] == nominal["depth_nm"]


@pytest.mark.parametrize("kwargs", [{"power_w": 2000}, {"time_s": 0}, {"pressure_mtorr": np.nan}])
def test_recipe_bounds(kwargs):
    with pytest.raises(ValueError):
        Recipe(**kwargs)


def test_sensor_bias_does_not_change_wafer_truth():
    a = wafer(rng=np.random.default_rng(5))
    b = wafer(fault="sensor_bias", rng=np.random.default_rng(5))
    assert a["depth_nm"] == b["depth_nm"]
    assert b["pressure_mtorr"] - a["pressure_mtorr"] == pytest.approx(6)


def test_chart_arithmetic_and_future_cannot_change_limits():
    base = np.tile([498, 499, 500, 501, 502], (30, 1)).astype(float)
    base[:, :] += np.linspace(-0.1, 0.1, 30)[:, None]
    b = spc.fit(base)
    assert b.sigma == pytest.approx(4 / 2.326)
    c = spc.monitor(b, [[501] * 5, [502] * 5, [500] * 5])
    assert c.ewma.tolist() == pytest.approx([500.2, 500.56, 500.448])
    assert c.xbar_ucl.iloc[0] == pytest.approx(500 + 0.577 * 4)
    changed = spc.monitor(b, [[501] * 5, [502] * 5, [900] * 5])
    assert c.ewma_ucl.tolist() == changed.ewma_ucl.tolist()
    assert c.cusum_plus.iloc[0] == pytest.approx(1 / (b.sigma / np.sqrt(5)) - 0.5)
    cap = spc.capability(base)
    assert cap["Cp"] == pytest.approx(40 / (6 * 4 / 2.326))
    assert cap["Cp"] == pytest.approx(cap["Cpk"])


@pytest.mark.parametrize("value", [np.ones((30, 4)), np.full((30, 5), np.nan), np.ones((30, 5))])
def test_bad_baseline(value):
    with pytest.raises(ValueError):
        spc.fit(value)


def test_reliability_hand_calculated_and_zero_failures():
    events = pd.DataFrame(
        [
            dict(tool="A", kind="production", duration_s=1000, units=10, good_units=8),
            dict(tool="A", kind="repair", duration_s=100, units=0, good_units=0),
            dict(tool="A", kind="clean", duration_s=100, units=0, good_units=0),
        ]
    )
    m = reliability(events, ideal_cycle_s=80).iloc[0]
    assert m.MTBF_h == pytest.approx(1000 / 3600)
    assert m.MTTR_h == pytest.approx(100 / 3600)
    assert m.availability == pytest.approx(1000 / 1200)
    assert m.OEE == pytest.approx((1000 / 1200) * 0.8 * 0.8)
    no_fail = reliability(events.loc[events.kind != "repair"], 80).iloc[0]
    assert no_fail.MTBF_h is None
    with pytest.raises(ValueError):
        reliability(events, ideal_cycle_s=200)


def test_fleet_reproducible_and_event_accounting():
    first = simulate(7, lots=20)
    second = simulate(7, lots=20)
    for a, b in zip(first, second):
        pd.testing.assert_frame_equal(a, b)
    wafers, _, events = first
    assert len(wafers) == 400
    assert events.units.sum() == len(wafers)
    assert events.good_units.sum() == wafers.good.sum()
    for _, block in events.groupby("tool"):
        np.testing.assert_allclose(block.start_s.iloc[1:], block.end_s.iloc[:-1])


def test_labels_exclude_current_failure_cross_service_and_censoring():
    rows = [
        dict(campaign=1, tool="A", episode=0, lot=i, failure_after_lot=i == 5) for i in range(6)
    ]
    rows += [
        dict(campaign=1, tool="A", episode=1, lot=i, failure_after_lot=False) for i in range(6, 14)
    ]
    result = labels(pd.DataFrame(rows))
    assert result.lot.tolist() == [0, 1, 2, 3, 4, 6, 7, 8]
    assert result.failure_next_5.tolist() == [1, 1, 1, 1, 1, 0, 0, 0]
    assert {"truth_age", "failure_after_lot", "episode", "campaign", "lot"}.isdisjoint(FEATURES)


def test_expected_improvement_and_equal_search_budget():
    assert expected_improvement(np.array([1.0]), np.array([0.0]), 2)[0] == 0
    history, recipes = search(seed=3, budget=7, initial=4)
    assert history.groupby("method").size().tolist() == [7, 7]
    for _, block in history.groupby("method"):
        assert (block.best_loss.diff().dropna() <= 0).all()
    assert set(recipes) == {"bayesian", "random"}
