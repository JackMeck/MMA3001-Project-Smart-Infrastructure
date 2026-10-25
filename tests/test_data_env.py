"""Raw environmental JSON -> cleaned grid, using small hand-built records."""
import json

import numpy as np
import pandas as pd
import pytest

from smart_infrastructure.data import (build_dataset, env_to_grid, occupancy_to_grid,
                                       parse_env_json, prepare_env)

NAMES = {"co2": "Carbon dioxide", "temperature": "Temperature", "humidity": "Humidity",
         "voc": "Light Volatile Organic Compounds", "pm2_5": "Particulate matter 2.5"}
T0 = pd.Timestamp("2024-03-04 00:00", tz="UTC").timestamp()


def record(values_by_var):
    """values_by_var: {canonical: [(epoch, value | 'ERR'), ...]} -> jsondata string."""
    out = []
    for col, items in values_by_var.items():
        vals = [{"time": t, "errorCode": 7} if v == "ERR" else {"time": t, "value": v}
                for t, v in items]
        out.append({"variable": {"name": NAMES[col]}, "values": vals})
    return json.dumps(out)


def make_raw(n=60, step_s=300, sensor="S1"):
    rng = np.random.default_rng(1)
    times = [T0 + i * step_s for i in range(n)]
    data = {c: [(t, float(v)) for t, v in zip(times, rng.normal(m, s, n))]
            for c, (m, s) in {"co2": (600, 40), "temperature": (22, 1), "humidity": (50, 3),
                              "voc": (100, 10), "pm2_5": (5, 1)}.items()}
    return data, pd.DataFrame({"sensorid": [sensor], "jsondata": [record(data)]})


def test_parse_counts_error_codes_and_bad_json(cfg):
    data, raw = make_raw(n=10)
    data["co2"][0] = (data["co2"][0][0], "ERR")
    raw = pd.concat([pd.DataFrame({"sensorid": ["S1"], "jsondata": [record(data)]}),
                     pd.DataFrame({"sensorid": ["S1"], "jsondata": ["{not json"]}),
                     pd.DataFrame({"sensorid": ["OTHER"], "jsondata": [record(data)]})])
    recs, stats = parse_env_json(raw, cfg, ["S1"])
    assert stats["n_bad_json"] == 1
    assert stats["n_error_code_entries"] == 1
    assert stats["n_input_rows"] == 2            # OTHER sensor filtered out
    assert len(recs) == 10                       # error-coded co2 entry keeps its timestamp
    assert recs["co2"].isna().sum() == 1
    assert recs["timestamp"].dtype.kind == "M" and recs["timestamp"].dt.tz is None


def test_parse_missing_columns_raises(cfg):
    with pytest.raises(ValueError, match="jsondata"):
        parse_env_json(pd.DataFrame({"sensorid": ["a"]}), cfg)


def test_parse_empty_returns_empty_frame(cfg):
    recs, _ = parse_env_json(pd.DataFrame({"sensorid": ["S1"], "jsondata": ["[]"]}), cfg)
    assert recs.empty and "co2" in recs.columns


def test_prepare_env_removes_out_of_range(cfg):
    data, _ = make_raw(n=60)
    data["co2"][5] = (data["co2"][5][0], 99999.0)       # above physical range
    raw = pd.DataFrame({"sensorid": ["S1"], "jsondata": [record(data)]})
    grid, audit, recs = prepare_env(raw, cfg, "S1")
    assert audit["out_of_range"]["co2"] == 1
    assert grid["co2"].max() < 5000
    assert audit["n_usable_records"] == len(recs)


def test_prepare_env_drops_placeholder_readings(cfg):
    data, _ = make_raw(n=60)
    for i in range(20):                                  # 1/3 of rows share one fingerprint
        t = data["co2"][i][0]
        data["co2"][i], data["temperature"][i], data["humidity"][i] = (t, 450.0), (t, 20.0), (t, 40.0)
    raw = pd.DataFrame({"sensorid": ["S1"], "jsondata": [record(data)]})
    _, audit, recs = prepare_env(raw, cfg, "S1")
    assert audit["n_placeholder_rows"] == 20
    assert len(recs) == 40


def test_prepare_env_unknown_sensor_raises(cfg):
    _, raw = make_raw()
    with pytest.raises(ValueError, match="No parsable records"):
        prepare_env(raw, cfg, "does-not-exist")


def test_prepare_env_requires_sensor_id():
    from smart_infrastructure import monash_config
    _, raw = make_raw()
    with pytest.raises(ValueError, match="env_sensor_id"):
        prepare_env(raw, monash_config())


def test_prepare_env_variable_without_data_raises(cfg):
    data, _ = make_raw()
    del data["pm2_5"]
    raw = pd.DataFrame({"sensorid": ["S1"], "jsondata": [record(data)]})
    with pytest.raises(ValueError, match="no valid data"):
        prepare_env(raw, cfg, "S1")


def test_env_to_grid_is_causal_and_regular(cfg):
    """Each reading is moved to the NEXT grid point (never an earlier one)."""
    ts = pd.to_datetime(["2024-03-04 00:01", "2024-03-04 00:14", "2024-03-04 00:31"])
    df = pd.DataFrame({"timestamp": ts, **{c: [1.0, 2.0, 3.0] for c in cfg.env_cols}})
    grid = env_to_grid(df, cfg)
    assert list(grid.index[:2]) == [pd.Timestamp("2024-03-04 00:15"),
                                    pd.Timestamp("2024-03-04 00:30")]
    assert grid.loc["2024-03-04 00:15", "co2"] == pytest.approx(1.5)   # two readings averaged
    assert pd.Series(grid.index).diff().dropna().nunique() == 1        # regular spacing


def test_env_to_grid_long_gaps_stay_missing(cfg):
    ts = pd.to_datetime(["2024-03-04 00:00", "2024-03-04 06:00"])
    df = pd.DataFrame({"timestamp": ts, **{c: [1.0, 2.0] for c in cfg.env_cols}})
    grid = env_to_grid(df, cfg)
    assert grid["co2"].isna().sum() > 0


def test_build_dataset_joins_filters_and_names_target(cfg):
    _, raw = make_raw(n=120)
    grid, _, _ = prepare_env(raw, cfg, "S1")
    ev = pd.DataFrame({"t": pd.to_datetime(["2024-03-04 00:00", "2024-03-04 12:00"]),
                       "headcount": [3.0, 3.0]})
    ds = build_dataset(grid, occupancy_to_grid(ev, cfg), cfg, "room1")
    assert list(ds.columns[:3]) == ["timestamp", "room", cfg.target_col]
    assert set(ds["room"]) == {"room1"}
    assert (ds[cfg.target_col] == 3.0).all()


def test_build_dataset_no_overlap_raises(cfg):
    _, raw = make_raw(n=60)
    grid, _, _ = prepare_env(raw, cfg, "S1")
    ev = pd.DataFrame({"t": pd.to_datetime(["2030-01-01 00:00", "2030-01-01 01:00"]),
                       "headcount": [1.0, 1.0]})
    with pytest.raises(ValueError, match="No overlapping rows"):
        build_dataset(grid, occupancy_to_grid(ev, cfg), cfg, "r")
