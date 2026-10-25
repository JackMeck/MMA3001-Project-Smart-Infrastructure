"""Tests for occupancy.data: parsing, placeholder removal, grid alignment, no-future-info."""
import numpy as np
import pandas as pd
import pytest

from occupancy import config
from occupancy.data import (add_co2_change, build_modelling_table, events_to_occupancy,
                            parse_env_records, remove_placeholders, to_grid)


def test_parse_skips_errors_missing_values_and_unknown_variables(env_json):
    cells, t0 = env_json
    out = parse_env_records(cells)
    assert list(out.index) == [t0, t0 + 900]          # error-coded and value-less entries dropped
    assert set(out.columns) == {"co2", "temperature"}  # unknown variable ignored
    assert out.loc[t0, "co2"] == 500 and out.loc[t0, "temperature"] == 21.5


def test_parse_empty_input_returns_empty_table():
    assert parse_env_records(["[]"]).empty


def test_remove_placeholders_drops_repeated_reading_only():
    n = 100
    env = pd.DataFrame({"co2": [520.0] * 30 + list(np.linspace(400, 600, 70)),
                        "temperature": [21.5] * 30 + [22.0] * 70,
                        "humidity": [42.0] * 30 + list(np.linspace(30, 60, 70))})
    out, dropped = remove_placeholders(env)
    assert len(out) == 70 and dropped == [(520.0, 21.5, 42.0)]
    assert (out["co2"] != 520.0).all()


def test_remove_placeholders_keeps_everything_when_none_found():
    env = pd.DataFrame({"co2": np.arange(100.0), "temperature": 20.0, "humidity": np.arange(100.0)})
    out, dropped = remove_placeholders(env)
    assert len(out) == 100 and dropped == []


def test_grid_assigns_reading_to_next_grid_point():
    idx = pd.to_datetime(["2024-03-01 10:01", "2024-03-01 10:15", "2024-03-01 10:16"])
    env = pd.DataFrame({"co2": [400.0, 500.0, 600.0]}, index=idx)
    out = to_grid(env, "2024-03-01", "2024-03-02")
    # 10:01 -> 10:15 (averaged with the 10:15 reading), 10:16 -> 10:30
    assert out.loc["2024-03-01 10:15", "co2"] == 450.0
    assert out.loc["2024-03-01 10:30", "co2"] == 600.0
    assert (out.index.to_series().diff().dropna() == pd.Timedelta("15min")).all()


def test_grid_fills_short_gaps_but_not_long_ones():
    idx = pd.to_datetime(["2024-03-01 10:00", "2024-03-01 10:30", "2024-03-01 13:00"])
    env = pd.DataFrame({"co2": [400.0, 410.0, 420.0]}, index=idx)
    out = to_grid(env, "2024-03-01", "2024-03-02")
    assert out.loc["2024-03-01 10:15", "co2"] == 400.0          # 1-step gap filled forward
    assert out.loc["2024-03-01 10:30":"2024-03-01 11:15", "co2"].eq(410.0).all()  # 45 min filled
    assert np.isnan(out.loc["2024-03-01 11:30", "co2"])           # beyond 3 steps stays missing


def test_co2_change_uses_past_values_only(modelling_df):
    """No-future-information property: changing later data cannot alter earlier features."""
    env = modelling_df[["co2"]].copy()
    a = add_co2_change(env)
    env2 = env.copy()
    env2.iloc[200:, 0] += 1000.0                                # corrupt the future
    b = add_co2_change(env2)
    pd.testing.assert_series_equal(a["co2_chg1h"].iloc[:200], b["co2_chg1h"].iloc[:200])
    assert a["co2_chg1h"].iloc[:4].isna().all()                  # first hour has no history
    assert a["co2_chg1h"].iloc[4] == pytest.approx(env["co2"].iloc[4] - env["co2"].iloc[0])


def test_events_to_occupancy_holds_last_state_and_averages():
    ev = pd.DataFrame({"collecteddate": ["2024-03-01T00:00:00Z", "2024-03-01T00:30:00Z",
                                         "2024-03-01T02:00:00Z"],
                       "headcount": [2, 6, 0]})
    occ = events_to_occupancy(ev, tz="UTC")
    assert occ.name == "occupancy"
    assert occ.loc["2024-03-01 00:15"] == pytest.approx(2.0)     # held at 2 until 00:30
    assert occ.loc["2024-03-01 01:00"] == pytest.approx(6.0)     # then held at 6


def test_events_to_occupancy_ignores_non_numeric_headcounts():
    ev = pd.DataFrame({"collecteddate": ["2024-03-01T00:00:00Z", "2024-03-01T00:10:00Z"],
                       "headcount": [3, "bad"]})
    assert events_to_occupancy(ev, tz="UTC").dropna().eq(3.0).all()


def test_modelling_table_adds_calendar_features_and_drops_missing_target(modelling_df):
    env = modelling_df[config.SENSOR_COLS[:-1]].copy()
    occ = modelling_df["occupancy"].copy()
    occ.iloc[0] = np.nan
    df = build_modelling_table(add_co2_change(env), occ)
    assert len(df) == len(modelling_df) - 1
    assert set(["hour", "weekend", "occupancy"]) <= set(df.columns)
    sat = df[df.index.dayofweek == 5]
    assert (sat["weekend"] == 1).all() and (df[df.index.dayofweek == 0]["weekend"] == 0).all()
    assert df["hour"].between(0, 23.75).all()
