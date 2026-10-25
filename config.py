"""Project-wide constants.

Everything that was a hard-coded "setting" in the original notebook lives here so
that the workflow can be reproduced or re-pointed at another sensor or room by
editing one file.
"""

#: Identifier of the one environmental sensor with usable data in the study window.
ENV_SENSOR_ID = "6012002000869"

#: UUID of the floorspace whose headcount is predicted (no location metadata exists).
FLOORSPACE = "941f0352-bb69-47d0-a9c5-ac29eb4f7319"

#: Inclusive date window (YYYY-MM-DD) containing real sensor data.
WINDOW_START, WINDOW_END = "2024-02-29", "2024-06-14"

#: Local time zone used for all timestamps (the files store UTC).
TZ = "Australia/Melbourne"

#: Names of the environmental variables in the raw JSON mapped to short column names.
ENV_VARS = {
    "Carbon dioxide": "co2",
    "Temperature": "temperature",
    "Humidity": "humidity",
    "Light Volatile Organic Compounds": "voc",
    "Particulate matter 2.5": "pm2_5",
}

#: Raw sensor columns plus the derived one-hour CO2 change (4 x 15 min), in a fixed order.
SENSOR_COLS = ["co2", "temperature", "humidity", "voc", "pm2_5", "co2_chg1h"]

#: Calendar features added to every modelling table.
TIME_COLS = ["hour", "weekend"]

#: All model inputs for the "full" feature set.
FEATURES = SENSOR_COLS + TIME_COLS

#: The lean input set chosen by cross-validation (CO2, temperature and time).
LEAN = ["co2", "temperature", "hour", "weekend"]

#: Name of the target column (mean headcount in people per interval).
TARGET = "occupancy"

#: Width of the regular time grid.
GRID = "15min"

#: Intervals per hour on the grid (used for the one-hour CO2 change).
STEPS_PER_HOUR = 4

#: Short sensor gaps (<= this many grid steps, i.e. 45 min) are forward-filled.
MAX_FFILL_STEPS = 3

#: Headcount is a held state; carry the last value forward at most 3 days (minutes).
MAX_HOLD_MINUTES = 4320

#: Identical CO2/temperature/humidity combinations making up >= this share are placeholders.
PLACEHOLDER_SHARE = 0.05

#: Fraction of the series used for training; the rest is the held-out test set.
TRAIN_FRAC = 0.8

#: Gap (grid steps; 8 = 2 h) between train and validation/test, longer than the ~1 h CO2 lag.
GAP = 8

#: Number of expanding-window cross-validation folds.
N_SPLITS = 5

#: Final SVR hyper-parameters, chosen on cross-validation only.
SVR_C, SVR_EPSILON = 0.3, 0.01

#: Cut-off dates of the main split, reused by the floorspace-sensitivity study.
TRAIN_END, TEST_START = "2024-05-23 23:45", "2024-05-24 02:00"

#: Date range read from the 9M-row occupancy file for the all-floorspace study.
SCAN_START, SCAN_END = "2024-02-20", "2024-06-16"
