from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import replace
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import Config


class FeatureSet:
    def __init__(self, X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame, groups: Dict[str, List[str]]):
        self.X = X
        self.y = y
        self.meta = meta
        self.groups = groups


def build_features(df: pd.DataFrame, cfg: Config) -> FeatureSet:
    """Build causal trailing-window features from a regular-grid dataset."""
    step = cfg.step_minutes
    steps = []
    for w in cfg.windows_min:
        n = int(round(w / step))
        if n < 1:
            raise ValueError(f"Window {w} min is shorter than the {step} min grid step.")
        steps.append((w, n))

    d = df.sort_values(["room", "timestamp"]).reset_index(drop=True)
    feats: Dict[str, pd.Series] = {}
    groups: Dict[str, List[str]] = {"raw": [], "rolling": [], "trend": [], "time": []}
    g = d.groupby("room", sort=False)
    for c in cfg.env_cols:
        feats[c] = d[c]
        groups["raw"].append(c)
        for w, n in steps:
            roll = g[c].transform(lambda s, n=n: s.rolling(n, min_periods=1).mean())
            diff = d[c] - g[c].shift(n)
            items = [(f"{c}_mean{w}m", roll, "rolling"), (f"{c}_chg{w}m", diff, "trend")]
            if n >= 2:
                std = g[c].transform(lambda s, n=n: s.rolling(n, min_periods=2).std())
                items.append((f"{c}_std{w}m", std, "rolling"))
            for name, ser, grp in items:
                feats[name] = ser
                groups[grp].append(name)

    if cfg.use_time_features:
        hour = d["timestamp"].dt.hour + d["timestamp"].dt.minute / 60.0
        feats["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        feats["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        feats["is_weekend"] = (d["timestamp"].dt.dayofweek >= 5).astype(float)
        groups["time"] = ["hour_sin", "hour_cos", "is_weekend"]

    unknown = set(cfg.feature_groups) - {"raw", "rolling", "trend", "time"}
    if unknown:
        raise ValueError(f"Unknown feature groups: {sorted(unknown)}")
    drop = [c for g, cols in groups.items() if g not in cfg.feature_groups for c in cols]
    groups = {g: cols for g, cols in groups.items() if g in cfg.feature_groups}
    if not groups:
        raise ValueError("No feature groups selected.")
    X = pd.DataFrame(feats).drop(columns=drop)
    meta = d[["timestamp", "room"]].copy()
    y = d[cfg.target_col].astype(float)
    groups = {k: v for k, v in groups.items() if v}
    return FeatureSet(X=X, y=y, meta=meta, groups=groups)


def parse_env_json(df_env: pd.DataFrame, cfg: Config, sensor_ids: Optional[Iterable[str]] = None) -> Tuple[pd.DataFrame, Dict[str, int]]:
    for c in ("sensorid", "jsondata"):
        if c not in df_env.columns:
            raise ValueError(f"Environmental data needs a '{c}' column.")
    sub = df_env
    if sensor_ids is not None:
        sub = df_env[df_env["sensorid"].astype(str).isin([str(s) for s in sensor_ids])]
    stats = {"n_input_rows": int(len(sub)), "n_bad_json": 0, "n_error_code_entries": 0, "n_missing_value_entries": 0}
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


def detect_placeholders(df: pd.DataFrame, cols: List[str], frac: float) -> Tuple[pd.Series, List[dict]]:
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
        found.append({"values": {c: float(v) for c, v in zip(cols, key)}, "rows": int(n), "share": float(n / len(df))})
    return mask, found


def stuck_mask(df: pd.DataFrame, cols: List[str], min_run: int) -> pd.Series:
    if not cols or df.empty:
        return pd.Series(False, index=df.index)
    x = df[cols]
    changed = x.ne(x.shift()).any(axis=1)
    run_id = changed.cumsum()
    run_len = run_id.map(run_id.value_counts())
    return (run_len >= min_run) & x.notna().all(axis=1)


def env_to_grid(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    g = df.set_index("timestamp")[list(cfg.env_cols)].sort_index()
    g.index = g.index.ceil(cfg.resample_rule)
    g = g.groupby(level=0).mean()
    full = pd.date_range(g.index.min(), g.index.max(), freq=cfg.resample_rule)
    g = g.reindex(full).ffill(limit=cfg.max_ffill_steps)
    g.index.name = "timestamp"
    return g


def prepare_env(df_env: pd.DataFrame, cfg: Config, sensor_id: Optional[str] = None) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
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
    audit["interval_min"] = {
        "median": float(dt.median()) if len(dt) else None,
        "p90": float(dt.quantile(0.9)) if len(dt) else None,
        "max": float(dt.max()) if len(dt) else None,
    }
    return env_to_grid(recs, cfg), audit, recs


def to_events(df: pd.DataFrame, cfg: Config, floorspace_ids: Optional[Iterable[str]] = None) -> pd.DataFrame:
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
        ts = raw
    else:
        ts = pd.to_datetime(raw, format="ISO8601", utc=True)
        ts = ts.dt.tz_convert(cfg.timezone).dt.tz_localize(None)
    d["t"] = ts
    d["headcount"] = pd.to_numeric(d["headcount"], errors="coerce")
    d = d.dropna(subset=["t", "headcount"])
    return d.sort_values("t").reset_index(drop=True)


def read_occupancy(path: str, cfg: Config, floorspace_ids: Iterable[str], chunksize: int = 1_000_000) -> pd.DataFrame:
    ids = [str(f) for f in floorspace_ids]
    parts = []
    for chunk in pd.read_csv(path, usecols=["floorspaceid", "headcount", "collecteddate"], chunksize=chunksize):
        parts.append(to_events(chunk, cfg, ids))
    if not parts:
        raise ValueError("Occupancy file is empty.")
    return pd.concat(parts, ignore_index=True).sort_values("t").reset_index(drop=True)


def occupancy_to_grid(events: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    s = events.set_index("t")["headcount"].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    kw = dict(label="right", closed="right")
    fine = s.resample("1min", **kw).last()
    n_ev = s.resample("1min", **kw).count()
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
    h = events["headcount"]
    gaps = events["t"].diff().dt.total_seconds().dropna() / 60.0
    valid = grid["occupancy"].notna()
    return {
        "n_events": int(len(events)),
        "first_event": str(events["t"].min()),
        "last_event": str(events["t"].max()),
        "headcount": {
            "mean": float(h.mean()),
            "median": float(h.median()),
            "p95": float(h.quantile(0.95)),
            "max": float(h.max()),
            "share_zero": float((h == 0).mean()),
        },
        "event_gap_min": {
            "median": float(gaps.median()),
            "p95": float(gaps.quantile(0.95)),
            "max": float(gaps.max()),
            "n_gaps_over_ffill_limit": int((gaps > cfg.occ_ffill_min).sum()),
        },
        "share_grid_bins_forward_filled": float((grid.loc[valid, "n_events"] == 0).mean()),
        "ffill_limit_min": int(cfg.occ_ffill_min),
    }


def build_dataset(env_grid: pd.DataFrame, occ_grid: pd.DataFrame, cfg: Config, room: str) -> pd.DataFrame:
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


def build_modelling_table(env_grid: pd.DataFrame, occ_grid: pd.DataFrame, cfg: Config, room: str) -> pd.DataFrame:
    return build_dataset(env_grid, occ_grid, cfg, room)


def pairing_scan(cfg: Config, df_env: Optional[pd.DataFrame] = None, occ: Optional[pd.DataFrame] = None, floorspace_ids: Optional[List[str]] = None) -> pd.DataFrame:
    from .models import make_models
    from .evaluation import cross_validate, time_block_folds, chronological_split

    ids = [str(f) for f in (floorspace_ids or cfg.floorspace_ids)]
    if not ids:
        raise ValueError("Give floorspace_ids to scan.")
    if df_env is None:
        df_env = pd.read_csv(cfg.env_path, usecols=["sensorid", "jsondata"])
    ev_all = (to_events(occ, cfg, ids) if occ is not None else read_occupancy(cfg.occupancy_path, cfg, ids))
    env_grid, _, _ = prepare_env(df_env, cfg)
    models = make_models(cfg.seed, cfg.env_cols[0], include_schedule=cfg.use_time_features)
    rows = []
    for fid in ids:
        ev = ev_all[ev_all["floorspaceid"] == fid]
        row = {"floorspace": fid[:8], "n_rows": 0}
        try:
            d = build_dataset(env_grid, occupancy_to_grid(ev, cfg), cfg, fid[:8])
        except ValueError:
            rows.append(row)
            continue
        fs = build_features(d, cfg)
        tr, _ = chronological_split(fs.meta, cfg.test_fraction, cfg.cv_gap_min)
        folds = time_block_folds(fs.meta["timestamp"].iloc[tr], cfg.cv_splits, cfg.cv_gap_min)
        y = fs.y.iloc[tr].reset_index(drop=True)
        X_all = fs.X.iloc[tr].reset_index(drop=True)
        env_only = [c for g, v in fs.groups.items() if g != "time" for c in v]
        res = cross_validate({"tree": models["tree"]}, X_all[env_only], y, folds, cfg)
        row.update({
            "n_rows": int(len(fs.y)),
            "spearman_co2": float(X_all[cfg.env_cols[0]].corr(y, method="spearman")),
            "cv_r2_env_only": float(res["r2"].mean()),
            "cv_mae_env_only": float(res["mae"].mean()),
        })
        if "baseline_schedule" in models:
            b = cross_validate({"s": models["baseline_schedule"]}, X_all, y, folds, cfg)
            row["cv_mae_schedule_only"] = float(b["mae"].mean())
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values("cv_r2_env_only", ascending=False, na_position="last").reset_index(drop=True)


__all__ = [
    "FeatureSet",
    "build_features",
    "parse_env_json",
    "detect_placeholders",
    "stuck_mask",
    "env_to_grid",
    "prepare_env",
    "to_events",
    "read_occupancy",
    "occupancy_to_grid",
    "audit_occupancy",
    "build_dataset",
    "build_modelling_table",
    "pairing_scan",
]
