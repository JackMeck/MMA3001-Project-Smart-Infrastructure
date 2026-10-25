"""Tests for occupancy.evaluation: metrics, chronological split, cross-validation."""
import numpy as np
import pandas as pd
import pytest

from occupancy import config
from occupancy.evaluation import chronological_split, cv_folds, cv_scores, cv_table, metrics
from occupancy.models import linear_on


def test_metrics_known_values():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    p = np.array([1.0, 1.0, 1.0, 5.0])      # errors: +1, 0, -1, +2
    m = metrics(y, p)
    assert m["MAE"] == pytest.approx(1.0)
    assert m["RMSE"] == pytest.approx(np.sqrt(6 / 4))
    assert m["bias"] == pytest.approx(0.5)
    assert m["R2"] == pytest.approx(1 - 6 / 5)


def test_metrics_perfect_prediction():
    y = np.array([0.5, 1.5, 2.5])
    m = metrics(y, y)
    assert m["MAE"] == 0 and m["RMSE"] == 0 and m["R2"] == 1 and m["bias"] == 0


def test_metrics_clip_negative_predictions_at_zero():
    assert metrics(np.array([0.0, 0.0]), np.array([-5.0, -1.0]))["MAE"] == 0.0


def test_split_is_chronological_with_gap_and_no_overlap(modelling_df):
    sp = chronological_split(modelling_df)
    n = len(modelling_df)
    assert len(sp.X_te) == n - int(0.8 * n)
    assert sp.X_tr.index.max() < sp.X_te.index.min()
    gap = (sp.X_te.index.min() - sp.X_tr.index.max()) / pd.Timedelta("15min")
    assert gap == config.GAP + 1                      # 8 discarded rows between the blocks
    assert not set(sp.X_tr.index) & set(sp.X_te.index)
    assert list(sp.X_tr.columns) == config.FEATURES and "occupancy" not in sp.X_tr


def test_cv_folds_expand_forward_and_respect_gap(modelling_df):
    X = modelling_df[config.FEATURES]
    sizes = []
    for tr, va in cv_folds(X, n_splits=5, gap=8):
        assert tr.max() + 8 < va.min()                 # validation strictly after training + gap
        assert tr.min() == 0                           # expanding window
        sizes.append(len(tr))
    assert sizes == sorted(sizes) and len(sizes) == 5


def test_cv_scores_shape_and_table(modelling_df):
    sp = chronological_split(modelling_df)
    s = cv_scores(linear_on(config.LEAN), sp.X_tr, sp.y_tr)
    assert s.shape == (5, 4) and list(s.columns) == ["MAE", "RMSE", "R2", "bias"]
    t = cv_table({"lin": linear_on(config.LEAN)}, sp.X_tr, sp.y_tr)
    assert t.loc["lin", "MAE"] == pytest.approx(s["MAE"].mean())
