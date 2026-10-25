"""Smoke and logic tests for occupancy.experiments on synthetic data."""
import numpy as np
import pandas as pd
import pytest

from occupancy import config
from occupancy.evaluation import chronological_split
from occupancy.experiments import (data_requirement, feature_ablation, final_test, floorspace_scan,
                                   leave_one_out, naive_benchmarks, svr_grid, time_models)
from occupancy.models import final_models


@pytest.fixture
def split(modelling_df):
    return chronological_split(modelling_df)


def test_ablation_time_only_is_worse_than_lean_on_sensor_driven_data(split):
    out = feature_ablation(*split)
    assert out.loc["Lean (CO2 + temp + time)", "CV MAE"] <= out.loc["Time only (no sensors)", "CV MAE"] + 1e-9


def test_leave_one_out_first_row_is_zero_change(split):
    out = leave_one_out(split.X_tr, split.y_tr)
    assert out.iloc[0]["change in RMSE"] == 0 and len(out) == len(config.FEATURES) + 1


def test_svr_grid_returns_minimum_of_table(split):
    table, (c, e) = svr_grid(split.X_tr, split.y_tr, c_grid=[0.1, 1], eps_grid=[0.01, 0.1])
    assert table.loc[c, e] == table.values.min()


def test_data_requirement_rows_and_columns(split):
    out = data_requirement({"lin": final_models()["Linear Regression"]}, split.X_tr, split.y_tr, fracs=(0.5, 1.0))
    assert len(out) == 2 and {"lin val MAE", "lin train MAE"} <= set(out.columns)


def test_final_test_predictions_are_non_negative_and_aligned(split):
    table, preds = final_test(final_models(), *split)
    assert set(table.index) == set(preds) and all((p >= 0).all() and len(p) == len(split.y_te) for p in preds.values())


def test_naive_benchmark_constant_has_zero_r2_like_behaviour(split):
    out = naive_benchmarks(*split)
    assert list(out.index) == ["Constant (mean of training)", "Constant (median of training)"]


def test_time_models_reports_positive_costs(split):
    out = time_models(final_models(), split.X_tr, split.y_tr, split.X_te, repeats=1)
    assert (out["fit (ms)"] > 0).all() and (out["peak fit memory (MB)"] > 0).all()


def test_floorspace_scan_uses_training_period_and_sorts(modelling_df):
    rng = np.random.default_rng(1)
    noisy = modelling_df.assign(occupancy=rng.normal(size=len(modelling_df)))
    out = floorspace_scan({"aaaaaaaa-x": modelling_df, "bbbbbbbb-y": noisy}, train_end="2024-03-15", min_rows=100)
    assert out.index[0] == "aaaaaaaa"
    assert out["Spearman(CO2, occupancy)"].is_monotonic_decreasing
