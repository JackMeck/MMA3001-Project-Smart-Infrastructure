"""Sensitivity, optimisation, performance and robustness studies.

Each function takes already-prepared data and returns a DataFrame, so results can be
saved, tested or plotted independently. Hyper-parameters and inputs are judged on
cross-validation of the training period only; the test set is touched only by
:func:`final_test`.
"""

from __future__ import annotations

import time
import tracemalloc

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import config
from .data import build_modelling_table
from .evaluation import chronological_split, cv_scores, metrics
from .models import ModelFactory, constant_model, linear_on, svr_on


# ---------------------------------------------------------------- 3a. feature ablation
def feature_variants() -> dict[str, list[str]]:
    """Named input subsets compared in the ablation (all use linear regression)."""
    return {
        "Full (all sensors + time)": config.FEATURES,
        "Lean (CO2 + temp + time)": config.LEAN,
        "CO2 + time": ["co2", "hour", "weekend"],
        "Time only (no sensors)": ["hour", "weekend"],
        "Sensors only (no time)": config.SENSOR_COLS,
    }


def feature_ablation(X_tr, y_tr, X_te, y_te) -> pd.DataFrame:
    """Compare linear models that see different input subsets.

    Args:
        X_tr, y_tr: Training period.
        X_te, y_te: Held-out test period (reported for reference, not used to choose).

    Returns:
        DataFrame indexed by variant with ``CV MAE, CV RMSE, Test MAE, Test RMSE, Test R2``.
    """
    rows = []
    for name, cols in feature_variants().items():
        make = linear_on(cols)
        cv = cv_scores(make, X_tr, y_tr).mean()
        te = metrics(y_te, make().fit(X_tr, y_tr).predict(X_te))
        rows.append({"variant": name, "CV MAE": cv["MAE"], "CV RMSE": cv["RMSE"],
                     "Test MAE": te["MAE"], "Test RMSE": te["RMSE"], "Test R2": te["R2"]})
    return pd.DataFrame(rows).set_index("variant")


def leave_one_out(X_tr, y_tr, features: list[str] = config.FEATURES) -> pd.DataFrame:
    """One-at-a-time input removal from the full linear model (cross-validation only).

    Args:
        X_tr, y_tr: Training period.
        features: Inputs of the full model.

    Returns:
        DataFrame indexed by dropped input with ``CV MAE, CV RMSE, change in MAE,
        change in RMSE`` relative to the full model.
    """
    base = cv_scores(linear_on(features), X_tr, y_tr).mean()
    rows = [{"dropped input": "(none, full model)", "CV MAE": base["MAE"], "CV RMSE": base["RMSE"],
             "change in MAE": 0.0, "change in RMSE": 0.0}]
    for col in features:
        s = cv_scores(linear_on([c for c in features if c != col]), X_tr, y_tr).mean()
        rows.append({"dropped input": col, "CV MAE": s["MAE"], "CV RMSE": s["RMSE"],
                     "change in MAE": s["MAE"] - base["MAE"], "change in RMSE": s["RMSE"] - base["RMSE"]})
    return pd.DataFrame(rows).set_index("dropped input")


# ---------------------------------------------------------------- 3b. SVR settings
SVR_C_GRID = [0.01, 0.03, 0.1, 0.3, 1]
SVR_EPS_GRID = [0, 0.001, 0.01, 0.05, 0.1, 0.2]


def svr_grid(X_tr, y_tr, cols: list[str] = config.LEAN, c_grid=SVR_C_GRID,
             eps_grid=SVR_EPS_GRID) -> tuple[pd.DataFrame, tuple[float, float]]:
    """Grid search of SVR ``C`` and ``epsilon`` on cross-validated MAE.

    Args:
        X_tr, y_tr: Training period.
        cols: Inputs the SVR sees.
        c_grid, eps_grid: Candidate values.

    Returns:
        ``(table, (C_best, eps_best))`` - MAE table (rows C, columns epsilon) and the
        minimising pair.
    """
    table = pd.DataFrame({e: {C: cv_scores(svr_on(cols, C, e), X_tr, y_tr)["MAE"].mean() for C in c_grid}
                          for e in eps_grid})
    table.index.name, table.columns.name = "C", "epsilon"
    return table, table.stack().idxmin()


def data_requirement(models: dict[str, ModelFactory], X_tr, y_tr,
                     fracs=(0.1, 0.25, 0.5, 0.75, 1.0), gap: int = config.GAP) -> pd.DataFrame:
    """Learning curve: error versus amount of training data.

    The last 20 % of the training period is a fixed validation block; models are trained
    on the earliest fractions of what remains (after a gap).

    Args:
        models: Mapping name -> factory.
        X_tr, y_tr: Training period.
        fracs: Fractions of the inner pool to train on.
        gap: Rows between the pool and the validation block.

    Returns:
        DataFrame indexed by "<fraction> (<rows> rows)" with validation and training MAE
        per model.
    """
    v0 = int(0.8 * len(X_tr))
    X_val, y_val = X_tr.iloc[v0:], y_tr.iloc[v0:]
    X_pool, y_pool = X_tr.iloc[:v0 - gap], y_tr.iloc[:v0 - gap]
    curve = {}
    for frac in fracs:
        n = int(frac * len(X_pool))
        row = {}
        for name, make in models.items():
            m = make().fit(X_pool.iloc[:n], y_pool.iloc[:n])
            row[f"{name} val MAE"] = metrics(y_val, m.predict(X_val))["MAE"]
            row[f"{name} train MAE"] = metrics(y_pool.iloc[:n], m.predict(X_pool.iloc[:n]))["MAE"]
        curve[f"{frac:.0%} ({n} rows)"] = row
    return pd.DataFrame(curve).T


# ---------------------------------------------------------------- 5b. computational cost
def _best_time(fn, repeats: int) -> float:
    """Best wall-clock time (s) of ``fn`` over ``repeats`` runs."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def time_models(models: dict[str, ModelFactory], X_tr, y_tr, X_te, repeats: int = 5) -> pd.DataFrame:
    """Measure fit time, prediction time and peak fit memory.

    Args:
        models: Mapping name -> factory.
        X_tr, y_tr: Training data to fit on.
        X_te: Data to predict.
        repeats: Runs per timing (best is reported).

    Returns:
        DataFrame indexed by model with ``fit (ms), predict (ms), predict per 1000 rows
        (ms), peak fit memory (MB)``. Times are machine-dependent and indicative.
    """
    perf = []
    for name, make in models.items():
        fit_s = _best_time(lambda: make().fit(X_tr, y_tr), repeats)
        m = make().fit(X_tr, y_tr)
        pred_s = _best_time(lambda: m.predict(X_te), repeats)
        tracemalloc.start()
        make().fit(X_tr, y_tr)
        peak_mb = tracemalloc.get_traced_memory()[1] / 1e6
        tracemalloc.stop()
        perf.append({"model": name, "fit (ms)": fit_s * 1e3, "predict (ms)": pred_s * 1e3,
                     "predict per 1000 rows (ms)": pred_s * 1e3 / len(X_te) * 1000,
                     "peak fit memory (MB)": peak_mb})
    return pd.DataFrame(perf).set_index("model")


# ---------------------------------------------------------------- 5c. final test
def final_test(models: dict[str, ModelFactory], X_tr, y_tr, X_te, y_te) -> tuple[pd.DataFrame, dict]:
    """Fit on the training period and score once on the held-out test period.

    Run this **once**, after every modelling choice is fixed.

    Args:
        models: Mapping name -> factory.
        X_tr, y_tr: Training period.
        X_te, y_te: Held-out test period.

    Returns:
        ``(table, predictions)`` - metrics per model and clipped predictions per model.
    """
    preds, rows = {}, {}
    for name, make in models.items():
        preds[name] = np.clip(make().fit(X_tr, y_tr).predict(X_te), 0, None)
        rows[name] = metrics(y_te, preds[name])
    return pd.DataFrame(rows).T, preds


def naive_benchmarks(X_tr, y_tr, X_te, y_te) -> pd.DataFrame:
    """Test metrics of constant predictors (mean/median of the training targets)."""
    out = {f"Constant ({s} of training)": metrics(y_te, constant_model(s)().fit(X_tr, y_tr).predict(X_te))
           for s in ("mean", "median")}
    return pd.DataFrame(out).T


# ---------------------------------------------------------------- Appendix: floorspace mapping
def floorspace_tables(env: pd.DataFrame, occ_all: dict[str, pd.Series]) -> dict[str, pd.DataFrame]:
    """Modelling table for every floorspace, using the same sensor ``env``."""
    return {fs: build_modelling_table(env, occ) for fs, occ in occ_all.items()}


def floorspace_scan(tables: dict[str, pd.DataFrame], train_end: str = config.TRAIN_END,
                    min_rows: int = 500) -> pd.DataFrame:
    """Rank floorspaces by CO2-headcount Spearman correlation on the training period only.

    Args:
        tables: Mapping floorspace -> modelling table.
        train_end: Last training timestamp (the scan never sees test data).
        min_rows: Skip floorspaces with fewer training rows.

    Returns:
        DataFrame indexed by short floorspace id, sorted by descending correlation.
    """
    rows = []
    for fs, d in tables.items():
        tr = d.loc[:train_end]
        if len(tr) < min_rows:
            continue
        rows.append({"floorspace": fs[:8], "train rows": len(tr), "mean occupancy": tr[config.TARGET].mean(),
                     "Spearman(CO2, occupancy)": spearmanr(tr["co2"], tr[config.TARGET])[0]})
    return pd.DataFrame(rows).set_index("floorspace").sort_values("Spearman(CO2, occupancy)", ascending=False)


def floorspace_sensitivity(tables: dict[str, pd.DataFrame], models: dict[str, ModelFactory],
                           train_end: str = config.TRAIN_END, test_start: str = config.TEST_START) -> pd.DataFrame:
    """Apply the final models, unchanged, to every floorspace with the main split dates.

    Args:
        tables: Mapping floorspace -> modelling table.
        models: Final models (no re-tuning).
        train_end, test_start: Split dates (identical to the main split).

    Returns:
        Long DataFrame (floorspace, model, test mean occ, MAE, RMSE, R2, bias); a constant
        median predictor is added as a reference.
    """
    models = dict(models)
    models["Constant (median)"] = constant_model("median")
    rows = []
    for fs, d in tables.items():
        tr, te = d.loc[:train_end], d.loc[test_start:]
        if len(tr) < 500 or len(te) < 200:
            continue
        X_a, y_a, X_b, y_b = tr[config.FEATURES], tr[config.TARGET], te[config.FEATURES], te[config.TARGET]
        for name, make in models.items():
            rows.append({"floorspace": fs[:8], "model": name, "test mean occ": y_b.mean(),
                         **metrics(y_b, make().fit(X_a, y_a).predict(X_b))})
    return pd.DataFrame(rows)
