"""Sensitivity and optimisation studies.

Every study here uses the TRAINING period only (cross-validation or a validation
block carved out of the training period) to inform model choices. The test set is
touched only by the ablation table's separate test columns, which are reported for
information and are not used to select anything.
"""

from typing import Callable, Dict, List, Tuple

import pandas as pd

from .config import Config
from .evaluation import cv_mae, cv_scores, metrics
from .models import linear_on, svr_on


def ablation_variants(features: List[str], sensor_cols: List[str],
                      cfg: Config) -> Dict[str, Callable]:
    """Linear models that see progressively fewer inputs.

    Args:
        features: All inputs (sensor columns + ``hour`` + ``weekend``).
        sensor_cols: Sensor-derived columns only.
        cfg: Configuration (``lean_cols``).

    Returns:
        Ordered dict variant name -> factory.
    """
    return {
        "Full (all sensors + time)": linear_on(features),
        "Lean (CO2 + temp + time)": linear_on(cfg.lean_cols),
        "CO2 + time": linear_on(["co2", "hour", "weekend"]),
        "Time only (no sensors)": linear_on(["hour", "weekend"]),
        "Sensors only (no time)": linear_on(sensor_cols),
    }


def feature_ablation(X_tr, y_tr, X_te, y_te, features: List[str],
                     sensor_cols: List[str], cfg: Config) -> pd.DataFrame:
    """Compare linear models using different input groups.

    Args:
        X_tr: Training inputs.
        y_tr: Training target.
        X_te: Test inputs (reported only; not used for choices).
        y_te: Test target.
        features: All inputs.
        sensor_cols: Sensor columns.
        cfg: Configuration.

    Returns:
        Frame indexed by variant with CV MAE/RMSE and test MAE/RMSE/R2.
    """
    rows = []
    for name, make in ablation_variants(features, sensor_cols, cfg).items():
        cv = cv_scores(make, X_tr, y_tr, cfg)
        te = metrics(y_te, make().fit(X_tr, y_tr).predict(X_te))
        rows.append({"variant": name, "CV MAE": cv["MAE"], "CV RMSE": cv["RMSE"],
                     "Test MAE": te["MAE"], "Test RMSE": te["RMSE"], "Test R2": te["R2"]})
    return pd.DataFrame(rows).set_index("variant")


def leave_one_out(X_tr, y_tr, features: List[str], cfg: Config) -> pd.DataFrame:
    """Drop each input in turn from the full linear model (cross-validation only).

    Args:
        X_tr: Training inputs.
        y_tr: Training target.
        features: All inputs.
        cfg: Configuration.

    Returns:
        Frame indexed by dropped input with CV MAE, CV RMSE and the change in RMSE
        relative to the full model (positive = the input helped).
    """
    base = cv_scores(linear_on(features), X_tr, y_tr, cfg)
    rows = [{"dropped input": "(none, full model)", "CV MAE": base["MAE"],
             "CV RMSE": base["RMSE"], "change in RMSE": 0.0}]
    for col in features:
        s = cv_scores(linear_on([c for c in features if c != col]), X_tr, y_tr, cfg)
        rows.append({"dropped input": col, "CV MAE": s["MAE"], "CV RMSE": s["RMSE"],
                     "change in RMSE": s["RMSE"] - base["RMSE"]})
    return pd.DataFrame(rows).set_index("dropped input")


def svr_grid(X_tr, y_tr, cfg: Config) -> Tuple[pd.DataFrame, Tuple[float, float]]:
    """Grid search of SVR ``C`` and ``epsilon`` on cross-validation MAE.

    Args:
        X_tr: Training inputs.
        y_tr: Training target.
        cfg: Configuration (grids and lean inputs).

    Returns:
        ``(grid, (C_best, epsilon_best))``; ``grid`` has ``C`` as rows and
        ``epsilon`` as columns.
    """
    grid = pd.DataFrame({e: {C: cv_mae(svr_on(cfg.lean_cols, C, e), X_tr, y_tr, cfg)
                             for C in cfg.svr_C_grid} for e in cfg.svr_epsilon_grid})
    grid.index.name, grid.columns.name = "C", "epsilon"
    return grid, grid.stack().idxmin()


def _validation_block(X_tr, y_tr, cfg: Config):
    """Split the training period into an early pool and a late validation block."""
    v0 = int(0.8 * len(X_tr))
    return (X_tr.iloc[:v0 - cfg.gap_steps], y_tr.iloc[:v0 - cfg.gap_steps],
            X_tr.iloc[v0:], y_tr.iloc[v0:])


def training_size_curve(make: Callable, X_tr, y_tr, cfg: Config) -> pd.DataFrame:
    """Validation error against the amount of training data.

    The last 20% of the training period (after a gap) is the validation block; the
    model is trained on the earliest fractions of the remaining pool.

    Args:
        make: Model factory.
        X_tr: Training inputs.
        y_tr: Training target.
        cfg: Configuration (``curve_fractions``).

    Returns:
        Frame indexed ``"<fraction> (<n> rows)"`` with MAE, RMSE and R2.
    """
    X_pool, y_pool, X_val, y_val = _validation_block(X_tr, y_tr, cfg)
    curve = {}
    for frac in cfg.curve_fractions:
        n = int(frac * len(X_pool))
        m = make().fit(X_pool.iloc[:n], y_pool.iloc[:n])
        curve[f"{frac:.0%} ({n} rows)"] = metrics(y_val, m.predict(X_val))
    return pd.DataFrame(curve).T[["MAE", "RMSE", "R2"]]


def data_requirement(models: Dict[str, Callable], X_tr, y_tr, cfg: Config) -> pd.DataFrame:
    """Training and validation MAE of several models against training-set size.

    Args:
        models: Dict name -> factory (the baseline is normally excluded).
        X_tr: Training inputs.
        y_tr: Training target.
        cfg: Configuration.

    Returns:
        Frame indexed ``"<fraction> (<n> rows)"`` with ``<name> val MAE`` and
        ``<name> train MAE`` columns. A growing gap between them signals overfitting.
    """
    X_pool, y_pool, X_val, y_val = _validation_block(X_tr, y_tr, cfg)
    curve = {}
    for frac in cfg.curve_fractions:
        n = int(frac * len(X_pool))
        row = {}
        for name, make in models.items():
            m = make().fit(X_pool.iloc[:n], y_pool.iloc[:n])
            row[f"{name} val MAE"] = metrics(y_val, m.predict(X_val))["MAE"]
            row[f"{name} train MAE"] = metrics(y_pool.iloc[:n], m.predict(X_pool.iloc[:n]))["MAE"]
        curve[f"{frac:.0%} ({n} rows)"] = row
    return pd.DataFrame(curve).T
