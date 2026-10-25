"""Causal feature construction.

Prediction contract (what one row represents):
    One row = one room at one grid timestamp ``t``. The model estimates the number
    of people present at ``t`` using ONLY environmental measurements observed at or
    before ``t``. Occupancy-derived signals (``cfg.exclude_cols``) and lagged
    occupancy are never used (target leakage, Week 5.6).
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import Config


@dataclass
class FeatureSet:
    """Container returned by :func:`build_features`.

    Attributes:
        X: Feature matrix (may contain NaN; imputation happens inside the model
            pipeline so it is fit on training data only).
        y: Target (people count).
        meta: ``timestamp`` and ``room`` for each row.
        groups: Feature-group name -> list of column names (for ablation).
    """

    X: pd.DataFrame
    y: pd.Series
    meta: pd.DataFrame
    groups: Dict[str, List[str]]


def build_features(df: pd.DataFrame, cfg: Config) -> FeatureSet:
    """Build trailing-window (causal) features from a clean regular-grid frame.

    Feature groups:
        * ``raw``: the current reading of each environmental sensor.
        * ``rolling``: trailing mean and standard deviation over each window.
        * ``trend``: change over each window, ``x(t) - x(t - w)`` (e.g. CO2 rise rate).
        * ``time``: sine/cosine time-of-day and weekend flag (optional).

    Args:
        df: Output of :func:`occupancy.io.clean`.
        cfg: Workflow configuration.

    Returns:
        A :class:`FeatureSet`.

    Raises:
        ValueError: If an excluded column is also listed as an input, or if a
            window is shorter than one grid step.
    """
    leak = set(cfg.exclude_cols) & set(cfg.env_cols)
    if leak:
        raise ValueError(f"Occupancy-derived columns used as inputs (leakage): {sorted(leak)}")
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
            if n >= 2:  # a standard deviation needs at least two samples in the window
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
