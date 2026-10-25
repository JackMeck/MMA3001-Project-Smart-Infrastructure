from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Config:
    """Settings for the Smart Infrastructure occupancy workflow."""

    data_path: str = "data/raw/occupancy.csv"
    fmt: str = "wide"
    timestamp_col: str = "timestamp"
    room_col: str = "room"
    target_col: str = "occupancy"
    env_cols: List[str] = field(default_factory=lambda: ["co2", "temperature", "humidity", "light", "sound"])
    exclude_cols: List[str] = field(default_factory=list)
    long_name_col: str = "measurement"
    long_value_col: str = "value"
    long_name_map: Dict[str, str] = field(default_factory=dict)
    timezone: str = "Australia/Melbourne"
    resample_rule: str = "5min"
    max_ffill_steps: int = 3
    valid_ranges: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "co2": (300.0, 5000.0),
            "temperature": (5.0, 45.0),
            "humidity": (0.0, 100.0),
            "light": (0.0, 20000.0),
            "sound": (20.0, 120.0),
            "occupancy": (0.0, 500.0),
            "voc": (0.0, 60000.0),
            "pm2_5": (0.0, 1000.0),
            "formaldehyde": (0.0, 2000.0),
            "pressure": (800.0, 1100.0),
        }
    )
    windows_min: List[int] = field(default_factory=lambda: [15, 30, 60])
    use_time_features: bool = True
    feature_groups: List[str] = field(default_factory=lambda: ["raw", "rolling", "trend", "time"])
    test_fraction: float = 0.2
    cv_splits: int = 5
    cv_gap_min: int = 60
    selection_metric: str = "rmse"
    cost_under: float = 2.0
    cost_over: float = 1.0
    occupied_threshold: float = 1.0
    seed: int = 42
    resolution_rules: List[str] = field(default_factory=lambda: ["1min", "5min", "15min", "30min"])
    source: str = "generic"

    env_path: str = "data/raw/5EnvSensor_MayToDec2024_180kRows.csv"
    occupancy_path: str = "data/raw/5occupancySensor_MayToDec2024_9MRows.csv"
    env_sensor_id: str = ""
    floorspace_id: str = ""
    floorspace_ids: List[str] = field(default_factory=list)
    window_start: str = ""
    window_end: str = ""
    env_variable_map: Dict[str, str] = field(
        default_factory=lambda: {
            "Carbon dioxide": "co2",
            "Temperature": "temperature",
            "Humidity": "humidity",
            "Light Volatile Organic Compounds": "voc",
            "Particulate matter 2.5": "pm2_5",
            "Formaldehyde": "formaldehyde",
            "Pressure": "pressure",
        }
    )
    required_env: List[str] = field(default_factory=lambda: ["co2"])
    placeholder_cols: List[str] = field(default_factory=lambda: ["co2", "temperature", "humidity"])
    placeholder_frac: float = 0.05
    stuck_min_run: int = 12
    occ_ffill_min: int = 4320

    @classmethod
    def from_json(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if "valid_ranges" in raw:
            raw["valid_ranges"] = {k: tuple(v) for k, v in raw["valid_ranges"].items()}
        return cls(**raw)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)

    @property
    def step_minutes(self) -> float:
        import pandas as pd

        return pd.Timedelta(self.resample_rule).total_seconds() / 60.0


def monash_config(**overrides) -> Config:
    base = dict(
        source="monash",
        env_cols=["co2", "temperature", "humidity", "voc", "pm2_5"],
        resample_rule="15min",
        max_ffill_steps=3,
        windows_min=[30, 60, 120],
        cv_gap_min=120,
        resolution_rules=["5min", "15min", "30min", "60min"],
    )
    base.update(overrides)
    return Config(**base)


__all__ = ["Config", "monash_config"]
