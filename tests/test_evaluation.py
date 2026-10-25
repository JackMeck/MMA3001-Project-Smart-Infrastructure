import numpy as np
import pandas as pd
import pytest

from smart_infrastructure import Config
from smart_infrastructure.evaluation import (chronological_split, computational_cost,
                                             cross_validate_models, cv_fold_scores, cv_mae,
                                             final_test, mae_improvement, metrics,
                                             naive_benchmarks, splitter, time_it)
from smart_infrastructure.models import linear_on


@pytest.fixture
def xy():
    idx = pd.date_range("2024-03-04", periods=500, freq="15min")
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(size=500), "hour": idx.hour + idx.minute / 60,
                      "weekend": (idx.dayofweek >= 5).astype(int)}, index=idx)
    y = pd.Series(5 + 2 * X["a"] + rng.normal(0, 0.1, 500), index=idx)
    return X, y


def test_metrics_perfect_and_known_values():
    y = np.array([2.0, 4.0, 6.0])
    perfect = metrics(y, y)
    assert perfect["MAE"] == perfect["RMSE"] == perfect["bias"] == 0.0 and perfect["R2"] == 1.0
    m = metrics(y, y - 1)                                     # always 1 person too low
    assert m["MAE"] == pytest.approx(1.0) and m["RMSE"] == pytest.approx(1.0)
    assert m["bias"] == pytest.approx(-1.0)                   # negative = under-estimate


def test_metrics_clip_negative_predictions_to_zero():
    assert metrics(np.array([0.0, 0.0]), np.array([-3.0, -1.0]))["MAE"] == 0.0


def test_chronological_split_order_sizes_and_gap(xy):
    X, y = xy
    c = Config()
    X_tr, y_tr, X_te, y_te = chronological_split(X, y, c)
    assert len(X_te) == 100 and len(X_tr) == 400 - c.gap_steps
    assert X_tr.index.max() < X_te.index.min()
    assert (X_te.index.min() - X_tr.index.max()) == pd.Timedelta(minutes=15 * (c.gap_steps + 1))
    assert X_tr.index.equals(y_tr.index) and X_te.index.equals(y_te.index)


@pytest.mark.parametrize("frac", [0, 1, 1.2, -0.1])
def test_chronological_split_invalid_fraction(xy, frac):
    with pytest.raises(ValueError):
        chronological_split(*xy, Config(train_fraction=frac))


def test_chronological_split_too_short_raises():
    idx = pd.date_range("2024-03-04", periods=8, freq="15min")
    with pytest.raises(ValueError, match="Not enough"):
        chronological_split(pd.DataFrame({"a": range(8)}, index=idx), pd.Series(range(8), index=idx), Config())


def test_cv_folds_never_use_future_rows_and_respect_gap(xy):
    X, _ = xy
    c = Config()
    folds = list(splitter(c).split(X))
    assert len(folds) == c.cv_splits
    for tr, va in folds:
        assert tr.max() < va.min()
        assert va.min() - tr.max() > c.gap_steps              # gap rows are skipped
    assert [len(t) for t, _ in folds] == sorted(len(t) for t, _ in folds)   # expanding window


def test_cv_fold_scores_one_row_per_fold(xy):
    X, y = xy
    s = cv_fold_scores(linear_on(["a"]), X, y, Config())
    assert len(s) == 5 and {"MAE", "RMSE", "R2", "bias"} <= set(s.columns)
    assert cv_mae(linear_on(["a"]), X, y, Config()) == pytest.approx(s["MAE"].mean())
    assert s["MAE"].max() < 0.2                               # the signal is recoverable


def test_cross_validate_models_keeps_model_order(xy):
    X, y = xy
    models = {"z": linear_on(["a"]), "b": linear_on(["a", "hour"])}
    assert list(cross_validate_models(models, X, y, Config()).index) == ["z", "b"]


def test_final_test_is_non_negative_and_aligned(xy):
    X, y = xy
    X_tr, y_tr, X_te, y_te = chronological_split(X, y - 100, Config())   # forces negative predictions
    test, preds = final_test({"lin": linear_on(["a"])}, X_tr, y_tr, X_te, y_te)
    assert (preds["lin"] >= 0).all() and len(preds["lin"]) == len(y_te)
    assert list(test.index) == ["lin"]


def test_mae_improvement():
    t = pd.DataFrame({"MAE": [4.0, 2.0, 5.0]}, index=["Baseline (schedule)", "good", "bad"])
    imp = mae_improvement(t)
    assert imp["Baseline (schedule)"] == 0 and imp["good"] == 0.5 and imp["bad"] == -0.25


def test_naive_benchmarks_use_training_constants(xy):
    X, y = xy
    X_tr, y_tr, X_te, y_te = chronological_split(X, y, Config())
    nb = naive_benchmarks(X_tr, y_tr, X_te, y_te)
    assert list(nb.index) == ["Constant (mean of training)", "Constant (median of training)"]
    expected = np.abs(y_te - y_tr.mean()).mean()
    assert nb.loc["Constant (mean of training)", "MAE"] == pytest.approx(expected)


def test_time_it_returns_best_of_repeats():
    calls = []
    t = time_it(lambda: calls.append(1), repeats=4)
    assert len(calls) == 4 and 0 <= t < 1


def test_computational_cost_columns(xy):
    X, y = xy
    X_tr, y_tr, X_te, _ = chronological_split(X, y, Config())
    cost = computational_cost({"lin": linear_on(["a"])}, X_tr, y_tr, X_te, repeats=1)
    assert list(cost.columns) == ["fit (ms)", "predict (ms)", "predict per 1000 rows (ms)",
                                  "peak fit memory (MB)"]
    assert (cost.to_numpy() > 0).all()
