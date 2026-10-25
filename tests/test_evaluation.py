import numpy as np
import pandas as pd
import pytest

from smart_infrastructure.evaluation import (chronological_split, leave_one_room_out,
                                             regression_metrics, time_block_folds)


@pytest.fixture
def meta():
    return pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=1000, freq="15min"),
                         "room": "r"})


def test_chronological_split_no_overlap_and_ordered(meta):
    tr, te = chronological_split(meta, 0.2, gap_min=60)
    assert set(tr).isdisjoint(te)
    assert meta.timestamp.iloc[tr].max() < meta.timestamp.iloc[te].min()
    assert abs(len(te) / len(meta) - 0.2) < 0.02


def test_chronological_split_gap_respected(meta):
    tr, te = chronological_split(meta, 0.2, gap_min=120)
    gap = meta.timestamp.iloc[te].min() - meta.timestamp.iloc[tr].max()
    assert gap > pd.Timedelta(minutes=120)


@pytest.mark.parametrize("frac", [0, 1, -0.1, 1.5])
def test_chronological_split_invalid_fraction(meta, frac):
    with pytest.raises(ValueError):
        chronological_split(meta, frac, 0)


def test_time_block_folds_train_precedes_validation(meta):
    folds = time_block_folds(meta.timestamp, n_splits=4, gap_min=30)
    assert len(folds) == 4
    for tr, va in folds:
        assert set(tr).isdisjoint(va)
        assert meta.timestamp.iloc[tr].max() < meta.timestamp.iloc[va].min()


def test_time_block_folds_expanding_window(meta):
    folds = time_block_folds(meta.timestamp, n_splits=3, gap_min=0)
    sizes = [len(tr) for tr, _ in folds]
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)


def test_leave_one_room_out_has_no_shared_rooms():
    m = pd.DataFrame({"room": list("aabbcc")})
    splits = list(leave_one_room_out(m))
    assert len(splits) == 3
    for tr, va in splits:
        assert set(m.room.iloc[tr]).isdisjoint(m.room.iloc[va])


def test_metrics_perfect_prediction(cfg):
    y = np.array([0, 2, 5, 10.0])
    m = regression_metrics(y, y, cfg)
    assert m["mae"] == m["rmse"] == m["bias"] == m["asym_cost"] == 0.0
    assert m["r2"] == 1.0


def test_metrics_known_values_and_bias_sign(cfg):
    y = np.array([10.0, 10.0, 10.0, 10.0])
    m = regression_metrics(y, y - 2, cfg)        # always under-estimates by 2
    assert m["mae"] == pytest.approx(2.0)
    assert m["rmse"] == pytest.approx(2.0)
    assert m["bias"] == pytest.approx(-2.0)      # negative = under-estimation
    assert m["under_rate"] == 1.0
    assert m["asym_cost"] == pytest.approx(cfg.cost_under * 2.0)


def test_metrics_clip_negative_predictions(cfg):
    m = regression_metrics([0.0, 0.0], [-5.0, -3.0], cfg)
    assert m["mae"] == 0.0


def test_metrics_single_class_gives_nan(cfg):
    m = regression_metrics([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], cfg)
    assert np.isnan(m["occ_recall"])
