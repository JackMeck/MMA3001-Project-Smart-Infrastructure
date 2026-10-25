"""Tests for occupancy.models."""
import numpy as np
import pandas as pd
import pytest

from occupancy import config
from occupancy.models import ScheduleBaseline, constant_model, final_models, linear_on, svr_on


def test_baseline_returns_slot_mean():
    X = pd.DataFrame({"weekend": [0, 0, 1], "hour": [9.0, 9.0, 9.0]})
    m = ScheduleBaseline().fit(X, [2.0, 4.0, 10.0])
    out = m.predict(pd.DataFrame({"weekend": [0, 1], "hour": [9.0, 9.0]}))
    assert out.tolist() == [3.0, 10.0]


def test_baseline_falls_back_to_overall_mean_for_unseen_slot():
    X = pd.DataFrame({"weekend": [0, 0], "hour": [9.0, 10.0]})
    m = ScheduleBaseline().fit(X, [2.0, 4.0])
    assert m.predict(pd.DataFrame({"weekend": [1], "hour": [3.0]}))[0] == pytest.approx(3.0)


def test_baseline_ignores_sensor_columns(modelling_df):
    y = modelling_df["occupancy"]
    a = ScheduleBaseline().fit(modelling_df, y).predict(modelling_df)
    b = ScheduleBaseline().fit(modelling_df.assign(co2=0.0), y).predict(modelling_df.assign(co2=999.0))
    np.testing.assert_allclose(a, b)


@pytest.mark.parametrize("name", ["Baseline (schedule)", "Linear Regression", "SVR"])
def test_final_models_fit_and_predict(modelling_df, name):
    X, y = modelling_df[config.FEATURES], modelling_df["occupancy"]
    pred = final_models()[name]().fit(X, y).predict(X)
    assert pred.shape == (len(X),) and np.isfinite(pred).all()


def test_sensor_models_beat_constant_on_synthetic_signal(modelling_df):
    X, y = modelling_df[config.FEATURES], modelling_df["occupancy"]
    lin = np.abs(linear_on(config.LEAN)().fit(X, y).predict(X) - y).mean()
    const = np.abs(constant_model("median")().fit(X, y).predict(X) - y).mean()
    assert lin < 0.5 * const


def test_models_handle_missing_input_by_median_imputation(modelling_df):
    X, y = modelling_df[config.FEATURES].copy(), modelling_df["occupancy"]
    X.iloc[:10, X.columns.get_loc("temperature")] = np.nan
    pred = svr_on(config.LEAN, C=1, epsilon=0.01)().fit(X, y).predict(X)
    assert np.isfinite(pred).all()


def test_scaling_statistics_come_from_training_data_only(modelling_df):
    """The pipeline must not leak test statistics: predicting must not change the fit."""
    X, y = modelling_df[config.FEATURES], modelling_df["occupancy"]
    m = linear_on(config.LEAN)().fit(X.iloc[:500], y.iloc[:500])
    mean_before = m.steps[-2][1].mean_.copy()
    m.predict(X.iloc[500:] * 100)
    np.testing.assert_array_equal(mean_before, m.steps[-2][1].mean_)


def test_factories_return_fresh_unfitted_models():
    make = linear_on(config.LEAN)
    assert make() is not make()
