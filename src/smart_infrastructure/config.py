"""Configuration for the occupancy-estimation workflow.

All dataset-specific assumptions (column names, units, valid ranges) live here so
the code can be pointed at the real Monash Smart Infrastructure extract by
editing one JSON file rather than the source.
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Config:
    """Settings for the full workflow.

    Attributes:
        data_path: Path to the raw CSV extract.
        fmt: ``"wide"`` (one column per measurement) or ``"long"``
            (one row per timestamp/room/measurement/value).
        timestamp_col: Name of the timestamp column in the raw file.
        room_col: Room/zone identifier column. If absent, a single room is assumed.
        target_col: Canonical name of the occupancy column (people count).
        env_cols: Canonical names of the environmental sensor columns used as inputs.
        exclude_cols: Columns that are occupancy-derived (e.g. PIR, people counter,
            camera count) and must NEVER be used as features (target leakage).
        long_name_col: (long format) column holding the measurement name.
        long_value_col: (long format) column holding the measurement value.
        long_name_map: (long format) raw measurement name -> canonical column name.
        timezone: Local timezone used for time-of-day features.
        resample_rule: Regular grid spacing, e.g. ``"5min"``.
        max_ffill_steps: Maximum number of grid steps a sensor value may be carried
            forward across a gap (never back-filled; the target is never filled).
        valid_ranges: Physical validity range ``(low, high)`` per canonical column
            (units documented in the README). Values outside become missing.
        windows_min: Trailing window lengths in minutes for rolling features.
        use_time_features: Include time-of-day / weekend features.
        feature_groups: Feature groups to keep: any of ``raw``, ``rolling``, ``trend``, ``time``
            (``raw`` is needed for the CO2-only baseline, ``time`` for the schedule baseline).
        test_fraction: Final chronological test fraction (used once).
        cv_splits: Number of expanding-window time-block CV folds.
        cv_gap_min: Minutes dropped between training and validation blocks to
            limit leakage through autocorrelation / overlapping rolling windows.
        selection_metric: Metric for choosing the final model (``rmse`` or ``mae``).
        cost_under: Cost weight per person of UNDER-estimating occupancy.
        cost_over: Cost weight per person of OVER-estimating occupancy.
        occupied_threshold: People (or mean people per interval) at or above which a
            space counts as occupied; applied to BOTH observed and predicted values.
        seed: Random seed for reproducibility.
        resolution_rules: Grid spacings tried in the input-resolution study.
        source: ``"generic"`` (wide/long CSV) or ``"monash"`` (nested JSON + events).
        env_path, occupancy_path: Raw Monash CSV files (monash adapter only).
        env_sensor_id: Environmental sensor ID used as the model input.
        floorspace_id: Occupancy floorspace ID used as the target.
        floorspace_ids: Floorspaces compared by ``pairing_scan``.
        window_start, window_end: Optional analysis window (local time, ISO strings).
        env_variable_map: Raw JSON variable name -> canonical column name.
        required_env: Canonical columns that must be present for a row to be kept.
        placeholder_cols: Columns used to detect repeated placeholder readings.
        placeholder_frac: A reading combination making up at least this share of a
            sensor's records is treated as a placeholder, not a measurement.
        stuck_min_run: Consecutive identical readings treated as a stuck sensor.
        occ_ffill_min: Longest time (minutes) the last occupancy state is carried
            forward across a silence in the event stream.
    """

    data_path: str = "data/raw/occupancy.csv"
    fmt: str = "wide"
    timestamp_col: str = "timestamp"
    room_col: str = "room"
    target_col: str = "occupancy"
    env_cols: List[str] = field(
        default_factory=lambda: ["co2", "temperature", "humidity", "light", "sound"]
    )
    exclude_cols: List[str] = field(default_factory=list)
    long_name_col: str = "measurement"
    long_value_col: str = "value"
    long_name_map: Dict[str, str] = field(default_factory=dict)
    timezone: str = "Australia/Melbourne"
    resample_rule: str = "5min"
    max_ffill_steps: int = 3
    valid_ranges: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "co2": (300.0, 5000.0),         # ppm
            "temperature": (5.0, 45.0),     # degC
            "humidity": (0.0, 100.0),       # %RH
            "light": (0.0, 20000.0),        # lux
            "sound": (20.0, 120.0),         # dB
            "occupancy": (0.0, 500.0),      # people
            "voc": (0.0, 60000.0),          # ppb
            "pm2_5": (0.0, 1000.0),         # ug/m3
            "formaldehyde": (0.0, 2000.0),  # ug/m3
            "pressure": (800.0, 1100.0),    # mb
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

    # --- Monash Smart Infrastructure extract (see occupancy/monash.py)
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
    placeholder_cols: List[str] = field(
        default_factory=lambda: ["co2", "temperature", "humidity"]
    )
    placeholder_frac: float = 0.05
    stuck_min_run: int = 12
    occ_ffill_min: int = 4320

    # ------------------------------------------------------------------ I/O
    @classmethod
    def from_json(cls, path: str) -> "Config":
        """Load a configuration from a JSON file (unknown keys raise ``TypeError``)."""
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if "valid_ranges" in raw:
            raw["valid_ranges"] = {k: tuple(v) for k, v in raw["valid_ranges"].items()}
        return cls(**raw)

    def to_json(self, path: str) -> None:
        """Write the configuration to a JSON file (recorded with every run)."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)

    @property
    def step_minutes(self) -> float:
        """Grid spacing in minutes derived from ``resample_rule``."""
        import pandas as pd

        return pd.Timedelta(self.resample_rule).total_seconds() / 60.0
