"""Shared fixtures: small synthetic datasets so tests run in seconds without real data."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def modelling_df():
    """Two weeks of 15-minute data with a known CO2 -> headcount relationship."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-03-04", periods=14 * 96, freq="15min")
    hour = idx.hour + idx.minute / 60
    weekend = (idx.dayofweek >= 5).astype(int)
    occ = np.where((hour >= 9) & (hour < 17) & (weekend == 0), 3.0, 0.2) + rng.normal(0, 0.1, len(idx))
    co2 = 420 + 40 * occ + rng.normal(0, 3, len(idx))
    df = pd.DataFrame({"co2": co2, "temperature": 21 + 0.3 * occ + rng.normal(0, .1, len(idx)),
                       "humidity": 40 + rng.normal(0, 1, len(idx)), "voc": rng.normal(100, 5, len(idx)),
                       "pm2_5": rng.normal(5, 1, len(idx))}, index=idx)
    df["co2_chg1h"] = df["co2"].diff(4)
    df["hour"], df["weekend"], df["occupancy"] = hour, weekend, np.clip(occ, 0, None)
    return df


def make_record(name, values):
    """One JSON record in the sensor file format."""
    return {"variable": {"name": name}, "values": values}


@pytest.fixture
def env_json():
    """Two CSV cells: valid readings, one error-coded reading, one unknown variable."""
    t0 = 1_709_000_000
    cell1 = json.dumps([make_record("Carbon dioxide", [{"time": t0, "value": 500},
                                                       {"time": t0 + 60, "value": 999, "errorCode": 3},
                                                       {"time": t0 + 120}]),
                        make_record("Temperature", [{"time": t0, "value": 21.5}]),
                        make_record("Mystery", [{"time": t0, "value": 1}])])
    cell2 = json.dumps([make_record("Carbon dioxide", [{"time": t0 + 900, "value": 520}])])
    return [cell1, cell2], t0
