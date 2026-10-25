import pytest

from smart_infrastructure import Config
from smart_infrastructure.evaluation import chronological_split
from smart_infrastructure.models import linear_on
from smart_infrastructure.sensitivity import (ablation_variants, data_requirement,
                                              feature_ablation, leave_one_out, svr_grid,
                                              training_size_curve)

SENSORS = ["co2", "temperature", "humidity", "voc", "pm2_5", "co2_chg1h"]
FEATURES = SENSORS + ["hour", "weekend"]


@pytest.fixture(scope="module")
def split(prepared, cfg):
    _, _, df, _ = prepared
    return chronological_split(df[FEATURES], df["occupancy"], cfg)


@pytest.fixture(scope="module")
def small():
    return Config(svr_C_grid=[0.1, 1.0], svr_epsilon_grid=[0.01, 0.1], curve_fractions=[0.5, 1.0])


def test_ablation_variants_names_and_columns(cfg):
    v = ablation_variants(FEATURES, SENSORS, cfg)
    assert list(v) == ["Full (all sensors + time)", "Lean (CO2 + temp + time)", "CO2 + time",
                       "Time only (no sensors)", "Sensors only (no time)"]
    assert v["CO2 + time"]().steps[0][1].transformers[0][2] == ["co2", "hour", "weekend"]


def test_feature_ablation_sensors_matter(split, cfg):
    X_tr, y_tr, X_te, y_te = split
    ab = feature_ablation(X_tr, y_tr, X_te, y_te, FEATURES, SENSORS, cfg)
    assert {"CV MAE", "CV RMSE", "Test MAE", "Test RMSE", "Test R2"} == set(ab.columns)
    # on data where occupancy drives CO2, removing the sensors must hurt a lot
    assert ab.loc["Time only (no sensors)", "CV MAE"] > 2 * ab.loc["Full (all sensors + time)", "CV MAE"]


def test_leave_one_out_dropping_co2_hurts_most(split, cfg):
    X_tr, y_tr, *_ = split
    loo = leave_one_out(X_tr, y_tr, FEATURES, cfg)
    assert list(loo.index) == ["(none, full model)"] + FEATURES
    assert loo.loc["(none, full model)", "change in RMSE"] == 0.0
    assert loo["change in RMSE"].idxmax() == "co2"
    assert loo.loc["voc", "change in RMSE"] < 0.05            # an irrelevant input changes little


def test_svr_grid_best_is_the_minimum(split, small):
    X_tr, y_tr, *_ = split
    grid, best = svr_grid(X_tr, y_tr, small)
    assert grid.shape == (2, 2) and grid.index.name == "C" and grid.columns.name == "epsilon"
    C, eps = best
    assert grid.loc[C, eps] == grid.to_numpy().min()
    assert C in small.svr_C_grid and eps in small.svr_epsilon_grid


def test_training_size_curve_uses_only_early_rows_and_validation_block(split, small):
    X_tr, y_tr, *_ = split
    curve = training_size_curve(linear_on(["co2", "hour", "weekend"]), X_tr, y_tr, small)
    assert list(curve.columns) == ["MAE", "RMSE", "R2"] and len(curve) == 2
    pool = int(0.8 * len(X_tr)) - small.gap_steps
    assert curve.index[1] == f"100% ({pool} rows)"
    assert curve.index[0] == f"50% ({int(0.5 * pool)} rows)"


def test_data_requirement_reports_train_and_validation_mae(split, small):
    X_tr, y_tr, *_ = split
    out = data_requirement({"Lin": linear_on(["co2", "hour", "weekend"])}, X_tr, y_tr, small)
    assert list(out.columns) == ["Lin val MAE", "Lin train MAE"] and (out > 0).all().all()
