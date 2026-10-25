import numpy as np
import pandas as pd
import pytest

from smart_infrastructure import Config
from smart_infrastructure.models import (ScheduleBaseline, candidate_models, final_models,
                                         linear_on, svr_on)


def slots():
    return pd.DataFrame({"hour": [9.0, 9.0, 9.0, 9.0, 14.0, 14.0],
                         "weekend": [0, 0, 1, 1, 0, 0]})


def test_schedule_baseline_predicts_slot_means_and_ignores_unseen():
    X, y = slots(), np.array([10.0, 20.0, 1.0, 3.0, 8.0, 8.0])
    m = ScheduleBaseline().fit(X, y)
    assert m.predict(X).tolist() == [15.0, 15.0, 2.0, 2.0, 8.0, 8.0]
    unseen = pd.DataFrame({"hour": [3.0], "weekend": [0]})
    assert m.predict(unseen)[0] == pytest.approx(y.mean())


def test_schedule_baseline_ignores_sensor_columns():
    X, y = slots(), np.arange(6.0)
    a = ScheduleBaseline().fit(X, y).predict(X)
    b = ScheduleBaseline().fit(X.assign(co2=999.0), y).predict(X.assign(co2=-5.0))
    assert a.tolist() == b.tolist()


def test_linear_on_only_uses_listed_columns():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(50, 3)), columns=["a", "b", "c"])
    y = 3 * X["a"] + rng.normal(0, 0.01, 50)
    m = linear_on(["a"])().fit(X, y)
    X2 = X.copy(); X2["b"] = 1e6                              # must not matter
    np.testing.assert_allclose(m.predict(X), m.predict(X2))
    assert np.corrcoef(m.predict(X), y)[0, 1] > 0.99


def test_models_handle_missing_values_via_imputer():
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0], "b": range(6)})
    y = pd.Series([1.0, 2, 3, 4, 5, 6])
    for make in (linear_on(["a", "b"]), svr_on(["a", "b"], 1.0, 0.1)):
        assert np.isfinite(make().fit(X, y).predict(X)).all()


def test_factories_return_fresh_unfitted_models():
    make = linear_on(["a"])
    assert make() is not make()
    assert not hasattr(make().steps[-1][1], "coef_")


def test_svr_settings_are_passed_through():
    svr = svr_on(["a"], C=0.3, epsilon=0.01)().steps[-1][1]
    assert (svr.C, svr.epsilon) == (0.3, 0.01)


def test_final_models_follow_config():
    names = list(final_models(Config()))
    assert names == ["Baseline (schedule)", "Linear Regression", "SVR"]
    svr = final_models(Config(svr_C=2.0, svr_epsilon=0.5))["SVR"]().steps[-1][1]
    assert (svr.C, svr.epsilon) == (2.0, 0.5)


def test_candidate_models_use_all_sensor_columns():
    m = candidate_models(["co2", "temperature"])
    assert list(m) == ["Baseline (schedule)", "Linear Regression", "SVR"]
    cols = m["Linear Regression"]().steps[0][1].transformers[0][2]
    assert cols == ["co2", "temperature"]
