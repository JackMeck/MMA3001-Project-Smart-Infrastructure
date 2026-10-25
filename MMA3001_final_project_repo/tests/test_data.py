import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from smart_infrastructure import Config
from smart_infrastructure.data import (add_sensor_features, build_table, drop_placeholders,
                                       events_to_occupancy, load_environment, load_occupancy,
                                       parse_env_json, read_events, sensor_columns, to_grid,
                                       unzip_data)
from synth import OTHER_ROOM, ROOM


def js(var, values):
    return json.dumps([{"variable": {"name": var}, "values": values}])


def test_parse_skips_error_codes_unmapped_and_missing_values():
    c = Config()
    raw = pd.DataFrame({"jsondata": [json.dumps([
        {"variable": {"name": "Carbon dioxide"},
         "values": [{"time": 1_700_000_000, "value": 500},
                    {"time": 1_700_000_300, "errorCode": 9},
                    {"time": 1_700_000_600, "value": 520}, {"time": 1_700_000_900},
                    {"time": 1_700_001_200, "value": 9999, "errorCode": 4}]},   # value + error: not a measurement
        {"variable": {"name": "Unknown thing"}, "values": [{"time": 1_700_000_000, "value": 1}]},
    ])]})
    env = parse_env_json(raw, c)
    assert env["co2"].tolist() == [500.0, 520.0]
    assert list(env.columns) == ["co2"]
    assert env.index.tz is None and env.index.is_monotonic_increasing


def test_parse_converts_epoch_to_melbourne_local_time():
    c = Config()
    # 2024-03-04 00:00:00 UTC is 11:00 in Melbourne (daylight saving, +11)
    env = parse_env_json(pd.DataFrame({"jsondata": [js("Carbon dioxide",
                         [{"time": 1709510400, "value": 450}])]}), c)
    assert env.index[0] == pd.Timestamp("2024-03-04 11:00:00")


def test_parse_no_co2_raises():
    with pytest.raises(ValueError, match="CO2"):
        parse_env_json(pd.DataFrame({"jsondata": [js("Temperature", [{"time": 1, "value": 20}])]}), Config())
    with pytest.raises(ValueError, match="CO2"):
        parse_env_json(pd.DataFrame({"jsondata": ["[]"]}), Config())


def test_drop_placeholders_removes_repeated_fixed_reading_and_duplicates():
    c = Config()
    n = 100
    idx = pd.date_range("2024-03-04", periods=n, freq="5min")
    env = pd.DataFrame({"co2": np.linspace(400, 600, n), "temperature": np.linspace(20, 22, n),
                        "humidity": np.linspace(40, 50, n)}, index=idx)
    env.iloc[:10] = [450.0, 20.0, 40.0]                      # 10% identical fingerprint
    dup = env.iloc[[50]].copy(); dup["co2"] = 999.0
    out, ph = drop_placeholders(pd.concat([env, dup]).sort_index(), c)
    assert ph == [(450.0, 20.0, 40.0)]
    assert len(out) == n - 10
    assert not out.index.duplicated().any()
    assert out.loc[idx[50], "co2"] == 999.0                  # keep="last"


def test_drop_placeholders_keeps_everything_when_none_repeat():
    c = Config()
    idx = pd.date_range("2024-03-04", periods=50, freq="5min")
    env = pd.DataFrame({"co2": np.arange(50.0) + 400, "temperature": np.arange(50.0) + 20,
                        "humidity": np.arange(50.0) + 40}, index=idx)
    out, ph = drop_placeholders(env, c)
    assert ph == [] and len(out) == 50


def test_to_grid_is_causal_averages_and_fills_short_gaps_only():
    c = Config(window_start="2024-03-04", window_end="2024-03-05")
    idx = pd.to_datetime(["2024-03-04 00:01", "2024-03-04 00:14", "2024-03-04 00:31",
                          "2024-03-04 03:00"])
    env = pd.DataFrame({"co2": [1.0, 3.0, 5.0, 9.0]}, index=idx)
    g = to_grid(env, c)
    assert g.index[0] == pd.Timestamp("2024-03-04 00:15")     # moved FORWARD, never back
    assert g.loc["2024-03-04 00:15", "co2"] == 2.0            # two readings averaged
    assert g.loc["2024-03-04 00:45", "co2"] == 5.0            # reading at 00:31 -> next grid point
    assert g.loc["2024-03-04 01:30", "co2"] == 5.0            # carried forward 3 steps (the limit)
    assert np.isnan(g.loc["2024-03-04 01:45", "co2"])         # beyond the limit stays missing
    assert pd.Series(g.index).diff().dropna().nunique() == 1


def test_to_grid_outside_window_raises():
    env = pd.DataFrame({"co2": [1.0]}, index=pd.to_datetime(["2023-01-01"]))
    with pytest.raises(ValueError, match="window"):
        to_grid(env, Config())


def test_add_sensor_features_uses_past_only():
    c = Config()
    env = pd.DataFrame({"co2": np.arange(10.0) * 10})
    out = add_sensor_features(env, c)
    assert np.isnan(out["co2_chg1h"].iloc[:4]).all()
    assert out["co2_chg1h"].iloc[4] == 40.0                   # x(t) - x(t-4)
    assert "co2_chg1h" not in env.columns                     # input not modified
    assert sensor_columns(out) == ["co2", "co2_chg1h"]


def test_load_environment_filters_sensor_and_reports_audit(raw_files, cfg):
    env, audit = load_environment(raw_files[0], cfg)
    assert audit["raw_rows"] == 70                            # other sensor excluded
    assert audit["placeholder_readings_dropped"] == [(450.0, 20.0, 40.0)]
    assert audit["grid_rows"] == len(env)
    assert env.index.min() >= pd.Timestamp(cfg.window_start)
    assert {"co2", "temperature", "humidity", "voc", "pm2_5", "co2_chg1h"} == set(env.columns)
    assert not ((env["co2"] == 450.0) & (env["temperature"] == 20.0)).any()


def test_load_environment_unknown_sensor_raises(raw_files):
    with pytest.raises(ValueError):
        load_environment(raw_files[0], Config(env_sensor_id="does-not-exist"))


def events(times, counts, tz="+11"):
    return pd.DataFrame({"headcount": counts,
                         "collecteddate": [f"{t}{tz}" for t in times]})


def test_events_to_occupancy_holds_state_and_converts_time():
    c = Config()
    ev = events(["2024-03-04 09:00:00", "2024-03-04 11:00:00"], [5, 7])
    occ = events_to_occupancy(ev, c)
    assert occ.name == "occupancy"
    assert occ.loc["2024-03-04 09:15":"2024-03-04 10:45"].eq(5.0).all()   # held
    assert occ.loc["2024-03-04 11:00"] == pytest.approx((14 * 5 + 7) / 15)
    assert occ.index.max() == pd.Timestamp("2024-03-04 11:00")          # series ends at last event


def test_events_to_occupancy_averages_within_interval_and_stops_after_limit():
    c = Config(occ_ffill_limit_min=60)
    ev = events(["2024-03-04 09:00:00", "2024-03-04 09:07:00", "2024-03-04 12:00:00"], [0, 15, 2])
    occ = events_to_occupancy(ev, c)
    assert occ.loc["2024-03-04 09:15"] == pytest.approx(15 * 9 / 15)   # 6 min at 0, 9 min at 15
    assert occ.loc["2024-03-04 10:00"] == 15.0
    assert occ.loc["2024-03-04 10:15"] == 15.0                          # only the held minutes count
    assert np.isnan(occ.loc["2024-03-04 10:30"])                        # beyond the carry limit
    assert occ.loc["2024-03-04 12:00"] == 2.0


def test_events_to_occupancy_ignores_bad_headcounts_and_duplicates():
    ev = events(["2024-03-04 09:00:00", "2024-03-04 09:00:00", "2024-03-04 09:30:00"],
                ["3", "4", "oops"])
    occ = events_to_occupancy(ev, Config())
    assert occ.dropna().iloc[0] == 4.0                        # duplicate time: last wins


def test_events_to_occupancy_empty_raises():
    with pytest.raises(ValueError, match="No valid"):
        events_to_occupancy(events(["2024-03-04 09:00:00"], ["bad"]), Config())


def test_read_events_filters_floorspace_and_dates(raw_files, cfg):
    only = read_events(raw_files[1], cfg, floorspace=ROOM)
    assert set(only["floorspaceid"]) == {ROOM}
    both = read_events(raw_files[1], cfg, date_range=("2024-03-10", "2024-03-12"))
    assert set(both["floorspaceid"]) == {ROOM, OTHER_ROOM}
    assert both["collecteddate"].str[:10].between("2024-03-10", "2024-03-12").all()


def test_read_events_chunked_equals_single_read(raw_files):
    a = read_events(raw_files[1], Config(chunksize=100), floorspace=ROOM)
    b = read_events(raw_files[1], Config(chunksize=10**7), floorspace=ROOM)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_load_occupancy_counts_events(raw_files, cfg):
    occ, n = load_occupancy(raw_files[1], cfg)
    assert n == len(read_events(raw_files[1], cfg, floorspace=ROOM)) and occ.notna().sum() > 1000


def test_build_table_columns_and_time_features(prepared):
    env, occ, df, _ = prepared
    assert {"occupancy", "hour", "weekend"} <= set(df.columns)
    assert df["co2"].notna().all() and df["occupancy"].notna().all()
    assert df["hour"].between(0, 23.75).all() and set(df["weekend"]) == {0, 1}
    sat = df[df.index.dayofweek == 5]
    assert (sat["weekend"] == 1).all()
    row = df.iloc[10]
    assert row["hour"] == row.name.hour + row.name.minute / 60


def test_build_table_no_overlap_raises():
    env = pd.DataFrame({"co2": [1.0]}, index=pd.to_datetime(["2024-03-04"]))
    occ = pd.Series([1.0], index=pd.to_datetime(["2030-01-01"]), name="occupancy")
    with pytest.raises(ValueError, match="No overlapping"):
        build_table(env, occ)


def test_unzip_data_extracts_once_and_skips_existing(tmp_path):
    z = tmp_path / "zips"; z.mkdir()
    with zipfile.ZipFile(z / "5EnvSensor.zip", "w") as f:
        f.writestr("a.csv", "x\n1\n")
    out = tmp_path / "data"
    assert unzip_data(str(z), str(out)) == ["a.csv"]
    (out / "a.csv").write_text("edited")
    unzip_data(str(z), str(out))
    assert (out / "a.csv").read_text() == "edited"            # not overwritten
