"""Settings for the occupancy-nowcasting workflow.

Everything that was a hard-coded constant in the notebook (sensor id, room id, study
window, grid spacing, split sizes, model settings) lives in one :class:`Config`
dataclass, so a run can be repeated or changed without editing the analysis code.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Config:
    """All settings for one run.

    Defaults reproduce the final notebook for sensor ``6012002000869`` and room
    ``941f0352-...`` in the Monash Smart Infrastructure extract.

    Attributes:
        env_sensor_id: Environmental sensor whose readings are the model inputs.
        floorspace: Floorspace id (room) whose headcount is predicted.
        window_start: First day of the period with real sensor data (inclusive).
        window_end: Last day of that period (inclusive).
        tz: Time zone used to convert epoch/ISO timestamps to local clock time.
        env_variables: Raw JSON variable name -> canonical column name.
        placeholder_cols: Columns forming the reading "fingerprint" used to detect a
            sensor repeating one fixed value.
        placeholder_frac: A fingerprint making up at least this share of records is a
            placeholder, not a measurement.
        grid: Pandas frequency string of the regular modelling grid.
        env_ffill_limit: Maximum number of grid steps a sensor reading is carried
            forward over a gap.
        occ_ffill_limit_min: Maximum minutes the last headcount is carried forward.
        change_lag_steps: Lag (grid steps) of the CO2 change feature; 4 x 15 min = 1 h.
        train_fraction: Fraction of rows used for training (rest is the final test).
        gap_steps: Grid steps removed between training and test (and between CV folds)
            so that lagged features cannot straddle the boundary.
        cv_splits: Number of time-ordered cross-validation folds.
        lean_cols: Inputs of the final "lean" models.
        svr_C: Final SVR regularisation strength (chosen on cross-validation).
        svr_epsilon: Final SVR epsilon-tube width (chosen on cross-validation).
        svr_C_grid: ``C`` values searched in the SVR settings sweep.
        svr_epsilon_grid: ``epsilon`` values searched in the SVR settings sweep.
        curve_fractions: Fractions of the training pool used for the data-requirement
            curve.
        scan_start: First date (``YYYY-MM-DD``) read for the all-floorspace appendix.
        scan_end: Last date read for the appendix.
        scan_min_train: Minimum training rows for a floorspace to be scored.
        scan_min_test: Minimum test rows for a floorspace to be scored.
        chunksize: Rows per chunk when reading the large occupancy CSV.
        env_csv_name: File name of the environmental CSV inside the data folder.
        occ_csv_name: File name of the occupancy CSV inside the data folder.
    """

    env_sensor_id: str = "6012002000869"
    floorspace: str = "941f0352-bb69-47d0-a9c5-ac29eb4f7319"
    window_start: str = "2024-02-29"
    window_end: str = "2024-06-14"
    tz: str = "Australia/Melbourne"
    env_variables: Dict[str, str] = field(default_factory=lambda: {
        "Carbon dioxide": "co2",
        "Temperature": "temperature",
        "Humidity": "humidity",
        "Light Volatile Organic Compounds": "voc",
        "Particulate matter 2.5": "pm2_5",
    })
    placeholder_cols: List[str] = field(
        default_factory=lambda: ["co2", "temperature", "humidity"])
    placeholder_frac: float = 0.05
    grid: str = "15min"
    env_ffill_limit: int = 3
    occ_ffill_limit_min: int = 4320
    change_lag_steps: int = 4
    train_fraction: float = 0.8
    gap_steps: int = 8
    cv_splits: int = 5
    lean_cols: List[str] = field(
        default_factory=lambda: ["co2", "temperature", "hour", "weekend"])
    svr_C: float = 0.3
    svr_epsilon: float = 0.01
    svr_C_grid: List[float] = field(default_factory=lambda: [0.01, 0.03, 0.1, 0.3, 1])
    svr_epsilon_grid: List[float] = field(
        default_factory=lambda: [0, 0.001, 0.01, 0.05, 0.1, 0.2])
    curve_fractions: List[float] = field(
        default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 1.0])
    scan_start: str = "2024-02-20"
    scan_end: str = "2024-06-16"
    scan_min_train: int = 500
    scan_min_test: int = 200
    chunksize: int = 1_000_000
    env_csv_name: str = "5EnvSensor_MayToDec2024_180kRows.csv"
    occ_csv_name: str = "5occupancySensor_MayToDec2024_9MRows.csv"

    @property
    def step_minutes(self) -> float:
        """Grid spacing in minutes, derived from :attr:`grid`."""
        import pandas as pd

        return pd.Timedelta(self.grid).total_seconds() / 60.0
