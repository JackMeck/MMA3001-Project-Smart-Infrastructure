"""Loading, cleaning and aligning the environmental and occupancy data.

Raw inputs (Monash Smart Infrastructure extract):
    * Environmental CSV: one row per sensor batch with columns ``sensorid`` and
      ``jsondata``. ``jsondata`` is a JSON list of variables, each holding
      ``(time, value)`` pairs (``time`` = Unix seconds). Entries carrying an
      ``errorCode`` are not measurements.
    * Occupancy CSV (~9 million rows): an EVENT stream with ``floorspaceid``,
      ``headcount`` and ``collecteddate`` (ISO-8601 with UTC offset).

Design decisions:
    * Error-coded and missing readings are discarded.
    * A CO2/temperature/humidity combination making up at least ``placeholder_frac``
      of the records is a placeholder, not a measurement, and is dropped.
    * Each sensor reading is moved to the NEXT grid point (``ceil``) so the value at
      time ``t`` never uses information from after ``t``.
    * Headcount is a held state: the last value is carried forward (up to
      ``occ_ffill_limit_min`` minutes) and averaged over each grid interval.
"""

import glob
import json
import os
import zipfile
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .config import Config


def unzip_data(zip_dir: str, data_dir: str) -> List[str]:
    """Extract the sensor ``.zip`` archives, skipping files that already exist.

    Args:
        zip_dir: Folder containing ``*Sensor*.zip`` archives.
        data_dir: Destination folder (created if needed).

    Returns:
        Names of the files present in ``data_dir`` afterwards.
    """
    os.makedirs(data_dir, exist_ok=True)
    for z in glob.glob(os.path.join(zip_dir, "*Sensor*.zip")):
        with zipfile.ZipFile(z) as f:
            for name in f.namelist():
                if not os.path.exists(os.path.join(data_dir, name)):
                    f.extract(name, data_dir)
    return os.listdir(data_dir)


def parse_env_json(raw: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Parse the nested ``jsondata`` column into one wide row per timestamp.

    Args:
        raw: Environmental rows for ONE sensor with a ``jsondata`` column.
        cfg: Configuration (``env_variables``, ``tz``).

    Returns:
        Frame indexed by naive local timestamp with one numeric column per mapped
        variable. Entries with an ``errorCode``, without a ``value``, or for an
        unmapped variable are skipped; rows with no CO2 value are dropped.

    Raises:
        ValueError: If no usable CO2 readings are found.
    """
    rows: Dict[float, dict] = {}
    for js in raw["jsondata"]:
        for rec in json.loads(js):
            name = cfg.env_variables.get((rec.get("variable") or {}).get("name"))
            if name is None:
                continue
            for v in rec.get("values", []):
                if "value" in v and "errorCode" not in v:
                    rows.setdefault(v["time"], {})[name] = v["value"]
    env = pd.DataFrame.from_dict(rows, orient="index")
    if env.empty or "co2" not in env.columns:
        raise ValueError("No usable CO2 readings found for this sensor.")
    env = env.apply(pd.to_numeric, errors="coerce").sort_index()
    env.index = (pd.to_datetime(pd.to_numeric(env.index), unit="s", utc=True)
                 .tz_convert(cfg.tz).tz_localize(None))
    env = env.dropna(subset=["co2"])
    if env.empty:
        raise ValueError("No usable CO2 readings found for this sensor.")
    return env


def drop_placeholders(env: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, list]:
    """Remove repeated fixed readings and duplicate timestamps.

    Args:
        env: Output of :func:`parse_env_json`.
        cfg: Configuration (``placeholder_cols``, ``placeholder_frac``).

    Returns:
        ``(clean, placeholders)``: the filtered, time-sorted frame and the list of
        reading combinations that were dropped (for the audit).
    """
    key = list(cfg.placeholder_cols)
    counts = env[key].value_counts()
    placeholder = counts[counts / len(env) >= cfg.placeholder_frac].index
    env = env[~env.set_index(key).index.isin(placeholder)]
    env = env[~env.index.duplicated(keep="last")].sort_index()
    return env, list(placeholder)


def to_grid(env: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Snap readings to a regular, causal grid inside the study window.

    Each reading is moved to the next grid point, readings sharing a grid point are
    averaged, and values are carried forward over at most ``env_ffill_limit`` steps.

    Args:
        env: Cleaned, time-sorted environmental frame.
        cfg: Configuration (``window_start``, ``window_end``, ``grid``).

    Returns:
        Frame on a regular grid index.

    Raises:
        ValueError: If no readings fall inside the study window.
    """
    env = env.loc[cfg.window_start:cfg.window_end]
    if env.empty:
        raise ValueError("No sensor readings inside the study window.")
    env.index = env.index.ceil(cfg.grid)
    env = env.groupby(level=0).mean()
    full = pd.date_range(env.index.min(), env.index.max(), freq=cfg.grid)
    return env.reindex(full).ffill(limit=cfg.env_ffill_limit)


def add_sensor_features(env: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add the CO2 change over ``change_lag_steps`` grid steps (past readings only).

    Args:
        env: Gridded environmental frame containing ``co2``.
        cfg: Configuration.

    Returns:
        Copy of ``env`` with an extra ``co2_chg1h`` column.
    """
    env = env.copy()
    env["co2_chg1h"] = env["co2"].diff(cfg.change_lag_steps)
    return env


def load_environment(env_csv: str, cfg: Config) -> Tuple[pd.DataFrame, Dict]:
    """Full environmental pipeline: read -> parse -> clean -> grid -> features.

    Args:
        env_csv: Path of the environmental CSV.
        cfg: Configuration.

    Returns:
        ``(env, audit)``: the gridded sensor frame (including ``co2_chg1h``) and a
        dictionary describing what was removed.

    Raises:
        ValueError: If the sensor has no usable data.
    """
    raw = pd.read_csv(env_csv, usecols=["sensorid", "jsondata"])
    raw = raw[raw["sensorid"].astype(str) == str(cfg.env_sensor_id)]
    parsed = parse_env_json(raw, cfg)
    clean, placeholders = drop_placeholders(parsed, cfg)
    env = add_sensor_features(to_grid(clean, cfg), cfg)
    audit = {"raw_rows": int(len(raw)), "rows_after_error_filter": int(len(parsed)),
             "placeholder_readings_dropped": placeholders,
             "grid_rows": int(len(env)),
             "first": str(env.index.min()), "last": str(env.index.max())}
    return env, audit


def events_to_occupancy(ev: pd.DataFrame, cfg: Config) -> pd.Series:
    """Convert one floorspace's headcount events to a regular occupancy series.

    The headcount is a held state: it is carried forward minute by minute for up to
    ``occ_ffill_limit_min`` minutes, then averaged over each grid interval. Minutes
    with no held value (beyond the limit) are ignored in the average, and the series
    ends at the last event (the final state is not extended into the future).

    Args:
        ev: Events with ``headcount`` and ``collecteddate`` columns.
        cfg: Configuration.

    Returns:
        Series named ``occupancy`` on the grid (mean people per interval).

    Raises:
        ValueError: If there are no valid events.
    """
    t = (pd.to_datetime(ev["collecteddate"], format="ISO8601", utc=True)
         .dt.tz_convert(cfg.tz).dt.tz_localize(None))
    s = pd.Series(pd.to_numeric(ev["headcount"], errors="coerce").to_numpy(),
                  index=t).dropna().sort_index()
    if s.empty:
        raise ValueError("No valid occupancy events.")
    s = s[~s.index.duplicated(keep="last")]
    s = s.resample("1min").last().ffill(limit=cfg.occ_ffill_limit_min)
    return s.resample(cfg.grid, closed="right", label="right").mean().rename("occupancy")


def read_events(occ_csv: str, cfg: Config, floorspace: Optional[str] = None,
                date_range: Optional[Tuple[str, str]] = None) -> pd.DataFrame:
    """Read the large occupancy CSV in chunks, keeping only the rows needed.

    Args:
        occ_csv: Path of the occupancy CSV.
        cfg: Configuration (``chunksize``).
        floorspace: Keep only this floorspace id (all floorspaces if None).
        date_range: Optional ``(first, last)`` ``YYYY-MM-DD`` dates to keep.

    Returns:
        Raw events with ``floorspaceid``, ``headcount`` and ``collecteddate``.
    """
    parts = []
    for c in pd.read_csv(occ_csv, usecols=["floorspaceid", "headcount", "collecteddate"],
                         chunksize=cfg.chunksize):
        if floorspace is not None:
            c = c[c["floorspaceid"] == floorspace]
        if date_range is not None:
            d = c["collecteddate"].astype(str).str[:10]
            c = c[(d >= date_range[0]) & (d <= date_range[1])]
        parts.append(c)
    return pd.concat(parts) if parts else pd.DataFrame(
        columns=["floorspaceid", "headcount", "collecteddate"])


def load_occupancy(occ_csv: str, cfg: Config) -> Tuple[pd.Series, int]:
    """Read and convert the occupancy of the chosen floorspace.

    Args:
        occ_csv: Path of the occupancy CSV.
        cfg: Configuration (``floorspace``).

    Returns:
        ``(occupancy, n_events)``: the gridded occupancy series and the number of raw
        events used.
    """
    ev = read_events(occ_csv, cfg, floorspace=cfg.floorspace)
    return events_to_occupancy(ev, cfg), int(len(ev))


def sensor_columns(env: pd.DataFrame) -> List[str]:
    """Names of the sensor-derived input columns (everything in the gridded frame)."""
    return list(env.columns)


def build_table(env: pd.DataFrame, occ: pd.Series) -> pd.DataFrame:
    """Join sensors and occupancy and add the time-of-week features.

    Args:
        env: Gridded sensor frame (from :func:`load_environment`).
        occ: Gridded occupancy series (from :func:`events_to_occupancy`).

    Returns:
        Modelling table: sensor columns, ``occupancy`` (target), ``hour`` (decimal
        hour of day) and ``weekend`` (0/1). Rows without occupancy or CO2 are dropped.

    Raises:
        ValueError: If sensors and occupancy do not overlap in time.
    """
    df = env.join(occ, how="inner").dropna(subset=["occupancy", "co2"])
    if df.empty:
        raise ValueError("No overlapping rows between the sensor and the floorspace.")
    df["hour"] = df.index.hour + df.index.minute / 60
    df["weekend"] = (df.index.dayofweek >= 5).astype(int)
    return df
