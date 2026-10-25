"""Loading, cleaning and gridding of the environmental and occupancy data.

Inputs
------
* **Environmental CSV** - columns ``sensorid`` and ``jsondata``. Each ``jsondata``
  cell is a JSON list of records ``{"variable": {"name": ...}, "values": [{"time":
  <unix seconds>, "value": <number>, optionally "errorCode": ...}]}``.
* **Occupancy CSV** (~9 million rows) - columns ``floorspaceid``, ``headcount`` and
  ``collecteddate`` (ISO-8601, UTC). One row per headcount *event*.

Outputs
-------
* An environment table indexed by a regular 15-minute grid (local time, tz-naive).
* An occupancy series (mean headcount per 15-minute interval).
* A modelling table joining both and adding ``hour`` and ``weekend``.

Invalid input handling
----------------------
Readings carrying an ``errorCode``, with a missing value, or with an unknown
variable name are discarded. Placeholder readings (one fixed CO2/temperature/
humidity combination repeated in >= 5 % of records) are removed. Duplicate
timestamps keep the last record. Short gaps (<= 45 min) are forward-filled;
longer gaps stay missing and are dropped when the modelling table is built.
"""

from __future__ import annotations

import glob
import json
import os
import zipfile
from typing import Iterable

import pandas as pd

from . import config


def extract_zips(zip_dir: str, data_dir: str, pattern: str = "*Sensor*.zip") -> list[str]:
    """Extract the data zip files into ``data_dir`` (existing files are kept).

    Args:
        zip_dir: Folder containing the zip files.
        data_dir: Destination folder (created if missing).
        pattern: Glob pattern for the zip files inside ``zip_dir``.

    Returns:
        Sorted list of file names present in ``data_dir`` afterwards.
    """
    os.makedirs(data_dir, exist_ok=True)
    for z in glob.glob(os.path.join(zip_dir, pattern)):
        with zipfile.ZipFile(z) as f:
            for name in f.namelist():
                if not os.path.exists(os.path.join(data_dir, name)):
                    f.extract(name, data_dir)
    return sorted(os.listdir(data_dir))


def parse_env_records(jsondata: Iterable[str]) -> pd.DataFrame:
    """Flatten nested sensor JSON into a wide table of readings.

    Args:
        jsondata: Iterable of JSON strings (one per CSV row), see module docstring.

    Returns:
        DataFrame indexed by Unix-second timestamps (as read) with one column per
        variable in :data:`occupancy.config.ENV_VARS`. Entries with an
        ``errorCode`` or no ``value`` are skipped. Empty if nothing is valid.
    """
    rows: dict = {}
    for js in jsondata:
        for rec in json.loads(js):
            name = config.ENV_VARS.get(rec.get("variable", {}).get("name"))
            if name is None:
                continue
            for v in rec.get("values", []):
                if "value" in v and "errorCode" not in v:
                    rows.setdefault(v["time"], {})[name] = v["value"]
    if not rows:
        return pd.DataFrame(columns=list(config.ENV_VARS.values()))
    return pd.DataFrame.from_dict(rows, orient="index").apply(pd.to_numeric, errors="coerce").sort_index()


def remove_placeholders(env: pd.DataFrame, share: float = config.PLACEHOLDER_SHARE,
                        key: tuple = ("co2", "temperature", "humidity")) -> tuple[pd.DataFrame, list]:
    """Drop stuck sensor readings.

    A (CO2, temperature, humidity) combination making up ``share`` or more of the
    records is a placeholder rather than a measurement.

    Args:
        env: Environment table with the ``key`` columns.
        share: Minimum fraction of records for a combination to count as a placeholder.
        key: Columns that define a reading.

    Returns:
        ``(cleaned, placeholders)`` - the table without placeholder rows and the list
        of dropped combinations.
    """
    key = list(key)
    counts = env[key].value_counts()
    placeholder = counts[counts / len(env) >= share].index
    if len(placeholder) == 0:
        return env, []
    keep = ~env.set_index(key).index.isin(placeholder)
    return env[keep], list(placeholder)


def to_grid(env: pd.DataFrame, start: str, end: str, freq: str = config.GRID,
            max_ffill: int = config.MAX_FFILL_STEPS) -> pd.DataFrame:
    """Place readings on a regular grid without using future information.

    Each reading is assigned to the *next* grid point (``ceil``), readings sharing a
    grid point are averaged, and gaps of up to ``max_ffill`` steps are forward-filled.

    Args:
        env: Table with a sorted, tz-naive local-time DatetimeIndex.
        start, end: Inclusive window (anything ``DataFrame.loc`` accepts).
        freq: Grid spacing.
        max_ffill: Longest gap (grid steps) to fill.

    Returns:
        Table on a continuous grid; longer gaps remain NaN.
    """
    env = env.loc[start:end].copy()
    env.index = env.index.ceil(freq)
    env = env.groupby(level=0).mean()
    grid = pd.date_range(env.index.min(), env.index.max(), freq=freq)
    return env.reindex(grid).ffill(limit=max_ffill)


def load_environment(env_csv: str, sensor_id: str = config.ENV_SENSOR_ID,
                     start: str = config.WINDOW_START, end: str = config.WINDOW_END,
                     tz: str = config.TZ, verbose: bool = True) -> pd.DataFrame:
    """Load, clean and grid one sensor's environmental readings.

    Args:
        env_csv: Path to the environmental CSV.
        sensor_id: Sensor to keep.
        start, end: Study window (local dates).
        tz: Local time zone for the index.
        verbose: Print row counts and dropped placeholder readings.

    Returns:
        DataFrame on a 15-minute grid with columns ``co2, temperature, humidity,
        voc, pm2_5`` (units: ppm, deg C, %, sensor units, sensor units).
    """
    raw = pd.read_csv(env_csv, usecols=["sensorid", "jsondata"])
    raw = raw[raw["sensorid"].astype(str) == sensor_id]
    env = parse_env_records(raw["jsondata"])
    env.index = pd.to_datetime(pd.to_numeric(env.index), unit="s", utc=True).tz_convert(tz).tz_localize(None)
    env = env.dropna(subset=["co2"])
    n_valid = len(env)
    env, placeholders = remove_placeholders(env)
    env = env[~env.index.duplicated(keep="last")].sort_index()
    if verbose:
        print(len(raw), "raw rows |", n_valid, "rows after error-code filter")
        print("placeholder readings dropped:", placeholders)
    env = to_grid(env, start, end)
    env = env.reindex(columns=list(config.ENV_VARS.values()))
    if verbose:
        print(env.shape, "| first", env.index.min(), "| last", env.index.max())
    return env


def events_to_occupancy(ev: pd.DataFrame, tz: str = config.TZ) -> pd.Series:
    """Convert a headcount event stream into mean headcount per 15-minute interval.

    The headcount is a *held state*: the last event persists (up to 3 days) until the
    next one, then the state is averaged over each 15-minute interval.

    Args:
        ev: Events with columns ``collecteddate`` (ISO-8601 UTC) and ``headcount``.
        tz: Local time zone.

    Returns:
        Series named ``occupancy`` (people) on a 15-minute grid in local tz-naive time.
    """
    t = pd.to_datetime(ev["collecteddate"], format="ISO8601", utc=True).dt.tz_convert(tz).dt.tz_localize(None)
    s = pd.Series(pd.to_numeric(ev["headcount"], errors="coerce").to_numpy(), index=t).dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s = s.resample("1min").last().ffill(limit=config.MAX_HOLD_MINUTES)
    return s.resample(config.GRID, closed="right", label="right").mean().rename(config.TARGET)


def _read_occupancy_chunks(occ_csv: str, chunksize: int = 1_000_000):
    """Yield the occupancy CSV in chunks (it is ~9M rows)."""
    yield from pd.read_csv(occ_csv, usecols=["floorspaceid", "headcount", "collecteddate"], chunksize=chunksize)


def load_occupancy(occ_csv: str, floorspace: str = config.FLOORSPACE, tz: str = config.TZ,
                   verbose: bool = True) -> pd.Series:
    """Load the occupancy series of one floorspace.

    Args:
        occ_csv: Path to the occupancy CSV.
        floorspace: Floorspace UUID to keep.
        tz: Local time zone.
        verbose: Print event and row counts.

    Returns:
        Mean headcount per 15-minute interval (see :func:`events_to_occupancy`).
    """
    ev = pd.concat([c[c["floorspaceid"] == floorspace] for c in _read_occupancy_chunks(occ_csv)])
    occ = events_to_occupancy(ev, tz)
    if verbose:
        print(len(ev), "events ->", occ.notna().sum(), "15-min rows")
    return occ


def load_all_occupancy(occ_csv: str, start: str = config.SCAN_START, end: str = config.SCAN_END,
                       tz: str = config.TZ) -> dict[str, pd.Series]:
    """Load occupancy for *every* floorspace in a date range (one pass over the file).

    Args:
        occ_csv: Path to the occupancy CSV.
        start, end: Inclusive UTC date strings (``YYYY-MM-DD``) used to prefilter rows.
        tz: Local time zone.

    Returns:
        Mapping floorspace UUID -> occupancy series.
    """
    parts = []
    for c in _read_occupancy_chunks(occ_csv):
        d = c["collecteddate"].astype(str).str[:10]
        parts.append(c[(d >= start) & (d <= end)])
    ev_all = pd.concat(parts)
    return {fs: events_to_occupancy(g, tz) for fs, g in ev_all.groupby("floorspaceid")}


def add_co2_change(env: pd.DataFrame) -> pd.DataFrame:
    """Add ``co2_chg1h``: CO2 now minus CO2 one hour ago (past readings only).

    Args:
        env: Environment table on the 15-minute grid with a ``co2`` column.

    Returns:
        Copy of ``env`` with the extra column (NaN for the first hour).
    """
    env = env.copy()
    env["co2_chg1h"] = env["co2"].diff(config.STEPS_PER_HOUR)
    return env


def build_modelling_table(env: pd.DataFrame, occ: pd.Series) -> pd.DataFrame:
    """Join sensor inputs with the occupancy target and add calendar features.

    Args:
        env: Environment table including ``co2_chg1h`` (see :func:`add_co2_change`).
        occ: Occupancy series.

    Returns:
        Table with the sensor columns, ``hour`` (0-23.75, interval end), ``weekend``
        (0/1) and ``occupancy``. Rows missing ``occupancy`` or ``co2`` are dropped;
        other missing inputs are left for the model's imputer.
    """
    df = env.join(occ, how="inner").dropna(subset=[config.TARGET, "co2"])
    df["hour"] = df.index.hour + df.index.minute / 60
    df["weekend"] = (df.index.dayofweek >= 5).astype(int)
    return df
