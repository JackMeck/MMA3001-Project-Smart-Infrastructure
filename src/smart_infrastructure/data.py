"""Adapter for the Monash Smart Infrastructure environmental + occupancy extract.

What the raw files look like (from the Colab inspection):
    * Environmental file: one row per sensor message. The ``jsondata`` column holds
      a list of variables (Carbon dioxide, Temperature, Humidity, ...), each with
      ``values`` of ``{"time": epoch_seconds, "value": x}``. Some entries carry an
      ``errorCode`` or have no ``value``; those are not measurements.
    * Occupancy file: an EVENT stream per ``floorspaceid`` (``headcount`` and
      ``occupancystatus`` at ``collecteddate``), not a regular time series.

Design decisions (each is reported in the audit so it can be justified in the report):
    * Readings that carry an ``errorCode`` or have no value are discarded.
    * A reading combination (e.g. CO2/temperature/humidity) that makes up at least
      ``placeholder_frac`` of a sensor's records is a placeholder, not a measurement.
    * Runs of ``stuck_min_run`` identical consecutive readings are a stuck sensor.
    * Environmental readings are snapped to the grid with ``ceil`` so the value
      labelled ``t`` only uses readings at or before ``t`` (strictly causal).
    * Occupancy is a state: the last reported headcount is carried forward for up to
      ``occ_ffill_min`` minutes, then averaged over the grid interval ``(t-step, t]``.
      The environmental features for ``t`` describe the same interval, so the task is
      a *nowcast* of current occupancy.
"""

import json
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import Config


# ------------------------------------------------------------------ config
def monash_config(**overrides) -> Config:
    """Config preset for the Monash extract (15-minute grid, five env variables).

    Args:
        **overrides: Any :class:`Config` field, e.g. ``env_sensor_id``,
            ``floorspace_id``, ``window_start``.

    Returns:
        A :class:`Config`.
    """
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


# --------------------------------------------------------- environmental data
def parse_env_json(df_env: pd.DataFrame, cfg: Config,
                   sensor_ids: Optional[Iterable[str]] = None
                   ) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Parse the nested ``jsondata`` column into one wide row per record and timestamp.

    Args:
        df_env: Raw environmental frame with ``sensorid`` and ``jsondata`` columns.
        cfg: Configuration (``env_variable_map``, ``timezone``).
        sensor_ids: Restrict to these sensors (strongly recommended; saves time).

    Returns:
        ``(records, stats)``: ``records`` has ``sensorid``, ``timestamp`` (naive local
        time) and one column per mapped variable; ``stats`` counts the entries that
        were discarded (bad JSON, error codes, missing values).

    Raises:
        ValueError: If required columns are missing.
    """
    for c in ("sensorid", "jsondata"):
        if c not in df_env.columns:
            raise ValueError(f"Environmental data needs a '{c}' column.")
    sub = df_env
    if sensor_ids is not None:
        sub = df_env[df_env["sensorid"].astype(str).isin([str(s) for s in sensor_ids])]
    stats = {"n_input_rows": int(len(sub)), "n_bad_json": 0,
             "n_error_code_entries": 0, "n_missing_value_entries": 0}
    vmap = cfg.env_variable_map
    rows: List[dict] = []
    for sid, js in zip(sub["sensorid"].astype(str), sub["jsondata"]):
        try:
            recs = json.loads(js)
        except (TypeError, ValueError):
            stats["n_bad_json"] += 1
            continue
        by_time: Dict[float, dict] = defaultdict(dict)
        for rec in recs or []:
            name = vmap.get(((rec or {}).get("variable") or {}).get("name"))
            if name is None:
                continue
            for item in rec.get("values") or []:
                if "errorCode" in item:
                    stats["n_error_code_entries"] += 1
                elif "value" not in item or "time" not in item:
                    stats["n_missing_value_entries"] += 1
                else:
                    by_time[item["time"]][name] = item["value"]
        for t, d in by_time.items():
            rows.append({**d, "sensorid": sid, "epoch": t})
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["sensorid", "timestamp", *vmap.values()]), stats
    for c in vmap.values():
        out[c] = pd.to_numeric(out.get(c), errors="coerce") if c in out else np.nan
    ts = pd.to_datetime(out.pop("epoch"), unit="s", utc=True)
    out["timestamp"] = ts.dt.tz_convert(cfg.timezone).dt.tz_localize(None)
    return out.reset_index(drop=True), stats


def detect_placeholders(df: pd.DataFrame, cols: List[str], frac: float
                        ) -> Tuple[pd.Series, List[dict]]:
    """Flag reading combinations that repeat far too often to be real measurements.

    Args:
        df: Record-level frame.
        cols: Columns forming the reading "fingerprint" (e.g. CO2, temperature, humidity).
        frac: A combination making up at least this share of ``len(df)`` is flagged.

    Returns:
        ``(mask, found)``: boolean mask of flagged rows and a list of dicts with the
        repeated values, their share and row count (for the audit).
    """
    mask = pd.Series(False, index=df.index)
    if not cols or df.empty:
        return mask, []
    sub = df[cols].dropna()
    if sub.empty:
        return mask, []
    counts = sub.value_counts()
    found = []
    for key, n in counts.items():
        if n / len(df) < frac:
            break
        key = key if isinstance(key, tuple) else (key,)
        hit = (df[cols] == pd.Series(key, index=cols)).all(axis=1)
        mask |= hit
        found.append({"values": {c: float(v) for c, v in zip(cols, key)},
                      "rows": int(n), "share": float(n / len(df))})
    return mask, found


def stuck_mask(df: pd.DataFrame, cols: List[str], min_run: int) -> pd.Series:
    """Flag runs of ``min_run`` or more consecutive identical readings.

    Args:
        df: Frame sorted by time (one row per timestamp).
        cols: Columns that must ALL be unchanged for a row to extend a run.
        min_run: Minimum run length (rows) to flag.

    Returns:
        Boolean mask (True = part of a stuck run). Rows with missing values are
        never flagged.
    """
    if not cols or df.empty:
        return pd.Series(False, index=df.index)
    x = df[cols]
    changed = x.ne(x.shift()).any(axis=1)
    run_id = changed.cumsum()
    run_len = run_id.map(run_id.value_counts())
    return (run_len >= min_run) & x.notna().all(axis=1)


def env_to_grid(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Snap readings to a regular grid (causal) and bridge short gaps.

    Each reading is moved to the NEXT grid point (``ceil``), readings sharing a grid
    point are averaged, and values are carried forward for at most
    ``max_ffill_steps`` steps. Long gaps stay missing.

    Args:
        df: Record-level frame with ``timestamp`` and ``cfg.env_cols``.
        cfg: Configuration.

    Returns:
        Frame indexed by grid ``timestamp`` with the environmental columns.
    """
    g = df.set_index("timestamp")[list(cfg.env_cols)].sort_index()
    g.index = g.index.ceil(cfg.resample_rule)
    g = g.groupby(level=0).mean()
    full = pd.date_range(g.index.min(), g.index.max(), freq=cfg.resample_rule)
    g = g.reindex(full).ffill(limit=cfg.max_ffill_steps)
    g.index.name = "timestamp"
    return g


def prepare_env(df_env: pd.DataFrame, cfg: Config, sensor_id: Optional[str] = None
                ) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
    """Parse, validate and grid one environmental sensor.

    Order: parse -> drop error-coded entries -> out-of-range to missing ->
    drop placeholder readings -> one record per timestamp -> drop stuck runs -> grid.

    Args:
        df_env: Raw environmental frame.
        cfg: Configuration.
        sensor_id: Sensor to use (defaults to ``cfg.env_sensor_id``).

    Returns:
        ``(grid, audit, records)``: the gridded sensor data, a JSON-serialisable audit
        of everything that was removed, and the cleaned record-level frame (used to
        re-grid for the resolution study).

    Raises:
        ValueError: If the sensor has no usable records or lacks required variables.
    """
    sid = str(sensor_id or cfg.env_sensor_id)
    if not sid:
        raise ValueError("Set cfg.env_sensor_id.")
    recs, stats = parse_env_json(df_env, cfg, [sid])
    if recs.empty:
        raise ValueError(f"No parsable records for sensor {sid}.")
    cols = list(cfg.env_cols)
    missing = [c for c in cols if c not in recs.columns or recs[c].isna().all()]
    if missing:
        raise ValueError(f"Variables with no valid data for sensor {sid}: {missing}")

    audit: Dict = {"sensor": sid, "n_records": int(len(recs)), **stats}
    oor = {}
    for c in cols:
        lo, hi = cfg.valid_ranges.get(c, (-np.inf, np.inf))
        bad = (recs[c] < lo) | (recs[c] > hi)
        oor[c] = int(bad.sum())
        recs.loc[bad, c] = np.nan
    audit["out_of_range"] = oor

    pcols = [c for c in cfg.placeholder_cols if c in recs.columns]
    pmask, found = detect_placeholders(recs, pcols, cfg.placeholder_frac)
    audit["placeholder_readings"] = found
    audit["n_placeholder_rows"] = int(pmask.sum())
    recs = recs.loc[~pmask]

    n0 = len(recs)
    recs = recs.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    audit["n_duplicate_timestamps_dropped"] = int(n0 - len(recs))

    smask = stuck_mask(recs, pcols, cfg.stuck_min_run)
    audit["n_stuck_rows"] = int(smask.sum())
    recs = recs.loc[~smask].reset_index(drop=True)
    if recs.empty:
        raise ValueError(f"No usable records left for sensor {sid} after cleaning.")

    dt = recs["timestamp"].diff().dt.total_seconds().dropna() / 60.0
    audit["n_usable_records"] = int(len(recs))
    audit["first_usable"] = str(recs["timestamp"].min())
    audit["last_usable"] = str(recs["timestamp"].max())
    audit["interval_min"] = {"median": float(dt.median()) if len(dt) else None,
                             "p90": float(dt.quantile(0.9)) if len(dt) else None,
                             "max": float(dt.max()) if len(dt) else None}
    return env_to_grid(recs, cfg), audit, recs


# ------------------------------------------------------------- occupancy data
def to_events(df: pd.DataFrame, cfg: Config,
              floorspace_ids: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Standardise raw occupancy rows to ``floorspaceid``, ``t`` (naive local), ``headcount``.

    Accepts either the raw frame (``collecteddate`` strings such as
    ``2023-11-24 00:36:52+11``) or a frame that already has a timezone-aware ``t``.

    Args:
        df: Occupancy rows.
        cfg: Configuration (timezone).
        floorspace_ids: Keep only these floorspaces (filtered BEFORE parsing times,
            which matters for the 9M-row file).

    Returns:
        Events sorted by time.

    Raises:
        ValueError: If required columns are missing.
    """
    tcol = "t" if "t" in df.columns else "collecteddate"
    for c in ("floorspaceid", "headcount", tcol):
        if c not in df.columns:
            raise ValueError(f"Occupancy data needs a '{c}' column.")
    d = df[["floorspaceid", "headcount", tcol]]
    if floorspace_ids is not None:
        d = d[d["floorspaceid"].isin([str(f) for f in floorspace_ids])]
    d = d.copy()
    d["floorspaceid"] = d["floorspaceid"].astype(str)
    raw = d.pop(tcol)
    if tcol == "t" and getattr(raw.dt, "tz", None) is None:
        ts = raw  # already naive local time
    else:
        ts = pd.to_datetime(raw, format="ISO8601", utc=True)
        ts = ts.dt.tz_convert(cfg.timezone).dt.tz_localize(None)
    d["t"] = ts
    d["headcount"] = pd.to_numeric(d["headcount"], errors="coerce")
    d = d.dropna(subset=["t", "headcount"])
    return d.sort_values("t").reset_index(drop=True)


def read_occupancy(path: str, cfg: Config, floorspace_ids: Iterable[str],
                   chunksize: int = 1_000_000) -> pd.DataFrame:
    """Read the large occupancy CSV in chunks, keeping only the chosen floorspaces.

    Args:
        path: CSV path.
        cfg: Configuration.
        floorspace_ids: Floorspaces to keep.
        chunksize: Rows per chunk.

    Returns:
        Standardised events (see :func:`to_events`).
    """
    ids = [str(f) for f in floorspace_ids]
    parts = []
    for chunk in pd.read_csv(path, usecols=["floorspaceid", "headcount", "collecteddate"],
                             chunksize=chunksize):
        parts.append(to_events(chunk, cfg, ids))
    if not parts:
        raise ValueError("Occupancy file is empty.")
    return pd.concat(parts, ignore_index=True).sort_values("t").reset_index(drop=True)


def occupancy_to_grid(events: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Convert one floorspace's event stream to a regular occupancy series.

    The headcount is a held state: it is carried forward minute by minute for up to
    ``occ_ffill_min`` minutes after the last event, then averaged over each grid
    interval ``(t - step, t]``.

    Args:
        events: Events for ONE floorspace (``t``, ``headcount``).
        cfg: Configuration.

    Returns:
        Frame indexed by grid ``timestamp`` with ``occupancy`` (mean people over the
        interval) and ``n_events`` (events observed in the interval).
    """
    s = events.set_index("t")["headcount"].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    kw = dict(label="right", closed="right")
    fine = s.resample("1min", **kw).last()
    n_ev = s.resample("1min", **kw).count()
    # extend to the end of the last grid interval (plus the carry-forward limit) so the
    # held state is not truncated at the final event
    end = (fine.index.max() + pd.Timedelta(minutes=int(cfg.occ_ffill_min))).ceil(cfg.resample_rule)
    full = pd.date_range(fine.index.min(), end, freq="1min")
    fine = fine.reindex(full).ffill(limit=int(cfg.occ_ffill_min))
    n_ev = n_ev.reindex(full, fill_value=0)
    out = pd.DataFrame({
        "occupancy": fine.resample(cfg.resample_rule, **kw).mean(),
        "n_events": n_ev.resample(cfg.resample_rule, **kw).sum(),
    })
    out.index.name = "timestamp"
    return out


def audit_occupancy(events: pd.DataFrame, grid: pd.DataFrame, cfg: Config) -> Dict:
    """Summarise one floorspace's event stream and how much was forward-filled.

    Args:
        events: Events for one floorspace.
        grid: Output of :func:`occupancy_to_grid`.
        cfg: Configuration.

    Returns:
        JSON-serialisable dictionary.
    """
    h = events["headcount"]
    gaps = events["t"].diff().dt.total_seconds().dropna() / 60.0
    valid = grid["occupancy"].notna()
    return {
        "n_events": int(len(events)),
        "first_event": str(events["t"].min()), "last_event": str(events["t"].max()),
        "headcount": {"mean": float(h.mean()), "median": float(h.median()),
                      "p95": float(h.quantile(0.95)), "max": float(h.max()),
                      "share_zero": float((h == 0).mean())},
        "event_gap_min": {"median": float(gaps.median()), "p95": float(gaps.quantile(0.95)),
                          "max": float(gaps.max()),
                          "n_gaps_over_ffill_limit": int((gaps > cfg.occ_ffill_min).sum())},
        "share_grid_bins_forward_filled": float((grid.loc[valid, "n_events"] == 0).mean()),
        "ffill_limit_min": int(cfg.occ_ffill_min),
    }


# ------------------------------------------------------------ dataset + runs
def build_dataset(env_grid: pd.DataFrame, occ_grid: pd.DataFrame, cfg: Config,
                  room: str) -> pd.DataFrame:
    """Join sensor and occupancy grids into the modelling table.

    Args:
        env_grid: Output of :func:`env_to_grid`.
        occ_grid: Output of :func:`occupancy_to_grid`.
        cfg: Configuration (``required_env``, optional analysis window).
        room: Label stored in the ``room`` column.

    Returns:
        Clean regular-grid frame ``timestamp, room, <target>, <env cols>``.

    Raises:
        ValueError: If no rows remain.
    """
    d = env_grid.join(occ_grid[["occupancy"]], how="inner")
    d = d.dropna(subset=["occupancy", *cfg.required_env])
    if cfg.window_start:
        d = d[d.index >= pd.Timestamp(cfg.window_start)]
    if cfg.window_end:
        d = d[d.index <= pd.Timestamp(cfg.window_end)]
    if d.empty:
        raise ValueError("No overlapping rows between the sensor and the floorspace.")
    d = d.reset_index()
    d.insert(1, "room", room)
    d = d.rename(columns={"occupancy": cfg.target_col})
    return d[["timestamp", "room", cfg.target_col, *cfg.env_cols]]
