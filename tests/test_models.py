import numpy as np
import pytest

from smart_infrastructure import build_features, make_models, param_grids
from smart_infrastructure.models import TIME_COLS, ScheduleBaseline


def test_model_names_and_schedule_option():
    assert {"baseline_mean", "baseline_co2_linear", "linear", "tree", "svr", "nn"} \
        == set(make_models())
    assert "baseline_schedule" in make_models(include_schedule=True)


def test_every_model_ends_with_named_model_step():
    for p in make_models(include_schedule=True).values():
        assert p.steps[-1][0] == "model"


def test_param_grids_target_existing_models():
    assert set(param_grids()) <= set(make_models())


def test_schedule_baseline_recovers_time_of_week_mean():
    # two slots: hour 6 (angle pi/2) weekday -> 10, weekend -> 2
    sin, cos = np.sin(2 * np.pi * 6 / 24), np.cos(2 * np.pi * 6 / 24)
    X = np.array([[sin, cos, 0.0]] * 3 + [[sin, cos, 1.0]] * 3)
    y = np.array([9, 10, 11, 1, 2, 3.0])
    pred = ScheduleBaseline().fit(X, y).predict(X)
    assert pred[:3].tolist() == [10.0] * 3 and pred[3:].tolist() == [2.0] * 3


def test_schedule_baseline_unseen_slot_falls_back_to_mean():
    X = np.array([[0.0, 1.0, 0.0]])
    model = ScheduleBaseline().fit(X, np.array([4.0]))
    other = np.array([[1.0, 0.0, 1.0]])
    assert model.predict(other)[0] == 4.0


def test_sensor_model_beats_mean_baseline(cfg, clean_df):
    fs = build_features(clean_df, cfg)
    X = fs.X.fillna(fs.X.median()); y = fs.y
    cut = int(len(X) * 0.8)
    models = make_models()
    errs = {}
    for name in ("baseline_mean", "linear"):
        models[name].fit(X.iloc[:cut], y.iloc[:cut])
        errs[name] = np.mean(np.abs(models[name].predict(X.iloc[cut:]) - y.iloc[cut:]))
    assert errs["linear"] < 0.5 * errs["baseline_mean"]


def test_time_cols_exposed():
    assert TIME_COLS == ["hour_sin", "hour_cos", "is_weekend"]
