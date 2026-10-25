import pytest

from smart_infrastructure import Config
from smart_infrastructure.evaluation import chronological_split
from smart_infrastructure.floorspace import (floorspace_scan, floorspace_sensitivity,
                                             occupancy_all_floorspaces)
from smart_infrastructure.models import final_models
from synth import OTHER_ROOM, ROOM

FEATURES = ["co2", "temperature", "humidity", "voc", "pm2_5", "co2_chg1h", "hour", "weekend"]


@pytest.fixture(scope="module")
def setup(raw_files, prepared, cfg):
    env, _, df, _ = prepared
    occ_all = occupancy_all_floorspaces(raw_files[1], cfg)
    X_tr, _, X_te, _ = chronological_split(df[FEATURES], df["occupancy"], cfg)
    return env, occ_all, X_tr.index.max(), X_te.index.min()


def test_all_floorspaces_found(setup):
    _, occ_all, *_ = setup
    assert set(occ_all) == {ROOM, OTHER_ROOM}
    assert all(s.name == "occupancy" for s in occ_all.values())


def test_scan_ranked_and_uses_training_period_only(setup, cfg):
    env, occ_all, train_end, _ = setup
    scan = floorspace_scan(env, occ_all, cfg, train_end)
    assert set(scan.index) == {ROOM[:8], OTHER_ROOM[:8]}
    rho = scan["Spearman(CO2, occupancy)"]
    assert rho.is_monotonic_decreasing and (rho > 0.5).all()  # CO2 tracks occupancy in the synthetic data
    full = floorspace_scan(env, occ_all, cfg, env.index.max())
    assert (full["train rows"] > scan["train rows"]).all()    # restricting to training drops rows


def test_scan_min_rows_filter(setup):
    env, occ_all, train_end, _ = setup
    assert floorspace_scan(env, occ_all, Config(scan_min_train=10**9), train_end).empty


def test_sensitivity_covers_every_room_and_model(setup, cfg):
    env, occ_all, train_end, test_start = setup
    s = floorspace_sensitivity(env, occ_all, final_models(cfg), FEATURES, cfg, train_end, test_start)
    assert set(s["floorspace"]) == {ROOM[:8], OTHER_ROOM[:8]}
    assert set(s["model"]) == {"Baseline (schedule)", "Linear Regression", "SVR", "Constant (median)"}
    assert {"MAE", "RMSE", "R2", "bias", "test mean occ"} <= set(s.columns)
    sub = s[s.floorspace == ROOM[:8]].set_index("model")["MAE"]
    assert sub["Linear Regression"] < sub["Constant (median)"]
