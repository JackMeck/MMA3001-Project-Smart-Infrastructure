"""Appendix study: does the sensor-to-room pairing matter?

The sensor is assumed to belong to one room. This module repeats the analysis on
EVERY floorspace in the occupancy file, using the same chronological split and the
same (already chosen) final models with no re-tuning, to show how sensitive the
results are to that assumption. It does not prove which room the sensor is in.
"""

from typing import Callable, Dict

import pandas as pd
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor

from .config import Config
from .data import build_table, events_to_occupancy, read_events
from .evaluation import metrics


def occupancy_all_floorspaces(occ_csv: str, cfg: Config) -> Dict[str, pd.Series]:
    """Gridded occupancy of every floorspace in the study period.

    Args:
        occ_csv: Path of the occupancy CSV.
        cfg: Configuration (``scan_start``, ``scan_end``).

    Returns:
        Dict floorspace id -> occupancy series. Floorspaces with no valid events are
        skipped.
    """
    ev = read_events(occ_csv, cfg, date_range=(cfg.scan_start, cfg.scan_end))
    out = {}
    for fs, g in ev.groupby("floorspaceid"):
        try:
            out[fs] = events_to_occupancy(g, cfg)
        except ValueError:
            continue
    return out


def floorspace_scan(env: pd.DataFrame, occ_all: Dict[str, pd.Series], cfg: Config,
                    train_end) -> pd.DataFrame:
    """Rank floorspaces by CO2-occupancy association on the TRAINING period only.

    Args:
        env: Gridded sensor frame.
        occ_all: Output of :func:`occupancy_all_floorspaces`.
        cfg: Configuration (``scan_min_train``).
        train_end: Last timestamp of the training period.

    Returns:
        Frame indexed by the first 8 characters of the floorspace id with training
        rows, mean occupancy and the Spearman correlation between CO2 and
        occupancy, sorted strongest first.
    """
    rows = []
    for fs, occ in occ_all.items():
        try:
            tr = build_table(env, occ).loc[:train_end]
        except ValueError:
            continue
        if len(tr) < cfg.scan_min_train:
            continue
        rows.append({"floorspace": fs[:8], "train rows": len(tr),
                     "mean occupancy": tr["occupancy"].mean(),
                     "Spearman(CO2, occupancy)": spearmanr(tr["co2"], tr["occupancy"])[0]})
    cols = ["floorspace", "train rows", "mean occupancy", "Spearman(CO2, occupancy)"]
    scan = pd.DataFrame(rows, columns=cols).set_index("floorspace")
    return scan.sort_values("Spearman(CO2, occupancy)", ascending=False)


def floorspace_sensitivity(env: pd.DataFrame, occ_all: Dict[str, pd.Series],
                           models: Dict[str, Callable], features, cfg: Config,
                           train_end, test_start) -> pd.DataFrame:
    """Fit and test the final models on every floorspace (same split, no re-tuning).

    Args:
        env: Gridded sensor frame.
        occ_all: Output of :func:`occupancy_all_floorspaces`.
        models: Final model factories.
        features: Input columns.
        cfg: Configuration (``scan_min_train``, ``scan_min_test``).
        train_end: Last timestamp of the training period.
        test_start: First timestamp of the test period.

    Returns:
        Long frame with ``floorspace``, ``model``, ``test mean occ`` and metrics.
    """
    rows = []
    models = dict(models)
    models["Constant (median)"] = lambda: DummyRegressor(strategy="median")
    for fs, occ in occ_all.items():
        try:
            d = build_table(env, occ)
        except ValueError:
            continue
        tr, te = d.loc[:train_end], d.loc[test_start:]
        if len(tr) < cfg.scan_min_train or len(te) < cfg.scan_min_test:
            continue
        X_a, y_a, X_b, y_b = tr[features], tr["occupancy"], te[features], te["occupancy"]
        for name, make in models.items():
            rows.append({"floorspace": fs[:8], "model": name, "test mean occ": y_b.mean(),
                         **metrics(y_b, make().fit(X_a, y_a).predict(X_b))})
    return pd.DataFrame(rows)
