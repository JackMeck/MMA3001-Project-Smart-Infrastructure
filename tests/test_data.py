import numpy as np
import pandas as pd
import pytest

from smart_infrastructure.data import (detect_placeholders, occupancy_to_grid,
                                       stuck_mask, to_events)


def test_stuck_mask_flags_only_long_runs():
    df = pd.DataFrame({"a": [1, 1, 1, 1, 2, 3, 3]})
    mask = stuck_mask(df, ["a"], min_run=3)
    assert mask.tolist() == [True, True, True, True, False, False, False]


def test_stuck_mask_never_flags_missing_values():
    df = pd.DataFrame({"a": [np.nan] * 5})
    assert not stuck_mask(df, ["a"], min_run=3).any()


def test_stuck_mask_empty_inputs():
    assert not stuck_mask(pd.DataFrame({"a": []}), ["a"], 3).any()
    assert not stuck_mask(pd.DataFrame({"a": [1, 1, 1]}), [], 3).any()


def test_detect_placeholders_flags_repeated_fingerprint():
    df = pd.DataFrame({"co2": [500.0] * 6 + [450, 460, 470, 480],
                       "temp": [20.0] * 6 + [21, 22, 23, 24]})
    mask, found = detect_placeholders(df, ["co2", "temp"], frac=0.5)
    assert mask.sum() == 6
    assert found[0]["values"] == {"co2": 500.0, "temp": 20.0}
    assert found[0]["rows"] == 6


def test_detect_placeholders_none_found():
    df = pd.DataFrame({"co2": np.arange(10.0), "temp": np.arange(10.0)})
    mask, found = detect_placeholders(df, ["co2", "temp"], frac=0.5)
    assert not mask.any() and found == []


def test_to_events_converts_to_naive_local_time(cfg):
    raw = pd.DataFrame({"floorspaceid": ["f1", "f2"], "headcount": ["3", "9"],
                        "collecteddate": ["2023-11-24 00:36:52+11", "2023-11-24 01:00:00+11"]})
    ev = to_events(raw, cfg, ["f1"])
    assert len(ev) == 1
    assert ev["t"].iloc[0] == pd.Timestamp("2023-11-24 00:36:52")
    assert ev["headcount"].iloc[0] == 3.0


def test_to_events_drops_unparseable_headcount(cfg):
    raw = pd.DataFrame({"floorspaceid": ["f1", "f1"], "headcount": ["x", "4"],
                        "collecteddate": ["2023-11-24 00:00:00+11"] * 2})
    assert to_events(raw, cfg)["headcount"].tolist() == [4.0]


def test_to_events_missing_column_raises(cfg):
    with pytest.raises(ValueError, match="headcount"):
        to_events(pd.DataFrame({"floorspaceid": ["f"], "collecteddate": ["2024-01-01"]}), cfg)


def test_occupancy_to_grid_holds_constant_headcount(cfg):
    ev = pd.DataFrame({"t": pd.to_datetime(["2024-03-04 09:00", "2024-03-04 10:00"]),
                       "headcount": [5.0, 5.0]})
    grid = occupancy_to_grid(ev, cfg)
    inside = grid.loc["2024-03-04 09:15":"2024-03-04 10:00", "occupancy"]
    assert (inside == 5.0).all()
    assert grid.index.name == "timestamp"
