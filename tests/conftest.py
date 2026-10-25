"""Shared fixtures: a small synthetic clean grid frame (no real data needed)."""
import numpy as np
import pandas as pd
import pytest

from smart_infrastructure import monash_config


@pytest.fixture
def cfg():
    return monash_config(env_sensor_id="sensor", floorspace_id="abcdefgh-0000")


@pytest.fixture
def clean_df(cfg):
    """~40 days of 15-minute data: occupancy follows a weekday daytime pattern and
    CO2 tracks occupancy, so a sensor model should clearly beat the mean baseline."""
    rng = np.random.default_rng(0)
    ts = pd.date_range("2024-03-01", periods=24 * 4 * 40, freq=cfg.resample_rule)
    day = np.clip(np.sin(np.pi * (ts.hour.values - 7) / 12), 0, None)
    occ = np.clip(20 * day * (ts.dayofweek.values < 5) + rng.normal(0, 2, len(ts)), 0, None)
    df = pd.DataFrame({"timestamp": ts, "room": "abcdefgh", cfg.target_col: occ})
    for c in cfg.env_cols:
        df[c] = 400 + 10 * occ + rng.normal(0, 20, len(ts)) if c == "co2" \
            else rng.normal(22, 1, len(ts))
    return df
