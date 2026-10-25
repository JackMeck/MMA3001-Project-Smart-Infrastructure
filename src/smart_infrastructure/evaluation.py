"""Metrics, chronological splitting, cross-validation, final test and timing.

All error metrics are in people. Splits are chronological: the model is always
trained on the past and scored on the future, with a gap of ``gap_steps`` grid steps
so lagged features cannot straddle the boundary.
"""

import time
import tracemalloc
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from .config import Config


def metrics(y, p) -> Dict[str, float]:
    """Error metrics for predicted headcount.

    Predictions are clipped at zero (a negative headcount is impossible) before
    scoring.

    Args:
        y: Observed people counts.
        p: Predicted people counts.

    Returns:
        Dict with ``MAE`` and ``RMSE`` (people), ``R2``, and ``bias`` =
        mean(pred - observed) (negative = under-estimation).
    """
    p = np.clip(p, 0, None)
    return {"MAE": mean_absolute_error(y, p), "RMSE": mean_squared_error(y, p) ** 0.5,
            "R2": r2_score(y, p), "bias": float(np.mean(p - y))}


def chronological_split(X: pd.DataFrame, y: pd.Series, cfg: Config
                        ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Train on the earliest ``train_fraction`` of rows, test on the rest.

    The last ``gap_steps`` rows before the cut are removed from training.

    Args:
        X: Input frame in time order.
        y: Target in the same order.
        cfg: Configuration (``train_fraction``, ``gap_steps``).

    Returns:
        ``(X_tr, y_tr, X_te, y_te)``.

    Raises:
        ValueError: If ``train_fraction`` is not in (0, 1) or the data is too short.
    """
    if not 0 < cfg.train_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1).")
    cut = int(cfg.train_fraction * len(X))
    if cut - cfg.gap_steps <= 0 or cut >= len(X):
        raise ValueError("Not enough rows for the requested split and gap.")
    return (X.iloc[:cut - cfg.gap_steps], y.iloc[:cut - cfg.gap_steps],
            X.iloc[cut:], y.iloc[cut:])


def splitter(cfg: Config) -> TimeSeriesSplit:
    """Expanding-window, time-ordered cross-validation splitter with a gap.

    Args:
        cfg: Configuration (``cv_splits``, ``gap_steps``).

    Returns:
        A :class:`~sklearn.model_selection.TimeSeriesSplit`.
    """
    return TimeSeriesSplit(n_splits=cfg.cv_splits, gap=cfg.gap_steps)


def cv_fold_scores(make: Callable, X_tr: pd.DataFrame, y_tr: pd.Series,
                   cfg: Config) -> pd.DataFrame:
    """Metrics of one model on each cross-validation fold.

    Args:
        make: Model factory.
        X_tr: Training inputs.
        y_tr: Training target.
        cfg: Configuration.

    Returns:
        One row of :func:`metrics` per fold.
    """
    return pd.DataFrame([
        metrics(y_tr.iloc[va], make().fit(X_tr.iloc[tr], y_tr.iloc[tr]).predict(X_tr.iloc[va]))
        for tr, va in splitter(cfg).split(X_tr)])


def cv_scores(make: Callable, X_tr: pd.DataFrame, y_tr: pd.Series,
              cfg: Config) -> pd.Series:
    """Mean cross-validation metrics of one model (mean over folds)."""
    return cv_fold_scores(make, X_tr, y_tr, cfg).mean()


def cv_mae(make: Callable, X_tr: pd.DataFrame, y_tr: pd.Series, cfg: Config) -> float:
    """Mean cross-validation MAE of one model."""
    return float(cv_scores(make, X_tr, y_tr, cfg)["MAE"])


def cross_validate_models(models: Dict[str, Callable], X_tr: pd.DataFrame,
                          y_tr: pd.Series, cfg: Config) -> pd.DataFrame:
    """Cross-validation table (mean of folds) for several models.

    Args:
        models: Ordered dict name -> factory.
        X_tr: Training inputs.
        y_tr: Training target.
        cfg: Configuration.

    Returns:
        Frame indexed by model name (in the order of ``models``) with mean MAE, RMSE,
        R2 and bias.
    """
    rows = []
    for name, make in models.items():
        for _, r in cv_fold_scores(make, X_tr, y_tr, cfg).iterrows():
            rows.append({"model": name, **r.to_dict()})
    return pd.DataFrame(rows).groupby("model").mean().loc[list(models)]


def time_it(fn: Callable, repeats: int = 5) -> float:
    """Best wall-clock time (seconds) of ``fn()`` over ``repeats`` calls."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def computational_cost(models: Dict[str, Callable], X_tr: pd.DataFrame, y_tr: pd.Series,
                       X_te: pd.DataFrame, repeats: int = 5) -> pd.DataFrame:
    """Fit time, predict time and peak fit memory of each model.

    Args:
        models: Ordered dict name -> factory.
        X_tr: Training inputs.
        y_tr: Training target.
        X_te: Inputs used for the prediction timing.
        repeats: Number of timing repeats (best is reported).

    Returns:
        Frame indexed by model with ``fit (ms)``, ``predict (ms)``,
        ``predict per 1000 rows (ms)`` and ``peak fit memory (MB)``.
    """
    perf = []
    for name, make in models.items():
        fit_s = time_it(lambda: make().fit(X_tr, y_tr), repeats)
        m = make().fit(X_tr, y_tr)
        pred_s = time_it(lambda: m.predict(X_te), repeats)
        tracemalloc.start()
        make().fit(X_tr, y_tr)
        peak_mb = tracemalloc.get_traced_memory()[1] / 1e6
        tracemalloc.stop()
        perf.append({"model": name, "fit (ms)": fit_s * 1e3, "predict (ms)": pred_s * 1e3,
                     "predict per 1000 rows (ms)": pred_s * 1e3 / len(X_te) * 1000,
                     "peak fit memory (MB)": peak_mb})
    return pd.DataFrame(perf).set_index("model")


def final_test(models: Dict[str, Callable], X_tr: pd.DataFrame, y_tr: pd.Series,
               X_te: pd.DataFrame, y_te: pd.Series
               ) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Fit each model on the training period and score it ONCE on the held-out test.

    Args:
        models: Ordered dict name -> factory.
        X_tr: Training inputs.
        y_tr: Training target.
        X_te: Test inputs.
        y_te: Test target.

    Returns:
        ``(test, preds)``: metrics per model and the (zero-clipped) predictions.
    """
    rows, preds = {}, {}
    for name, make in models.items():
        m = make().fit(X_tr, y_tr)
        preds[name] = np.clip(m.predict(X_te), 0, None)
        rows[name] = metrics(y_te, preds[name])
    return pd.DataFrame(rows).T, preds


def mae_improvement(test: pd.DataFrame, baseline: str = "Baseline (schedule)") -> pd.Series:
    """Fractional MAE reduction of each model relative to ``baseline``.

    Args:
        test: Output of :func:`final_test`.
        baseline: Row label of the reference model.

    Returns:
        ``1 - MAE / MAE_baseline`` per model (positive = better than the baseline).
    """
    return 1 - test["MAE"] / test.loc[baseline, "MAE"]


def naive_benchmarks(X_tr: pd.DataFrame, y_tr: pd.Series, X_te: pd.DataFrame,
                     y_te: pd.Series) -> pd.DataFrame:
    """Test-set scores of constant predictors (training mean / median); reference only.

    Args:
        X_tr: Training inputs.
        y_tr: Training target.
        X_te: Test inputs.
        y_te: Test target.

    Returns:
        Frame of metrics, one row per constant strategy.
    """
    out = {}
    for strat in ["mean", "median"]:
        m = DummyRegressor(strategy=strat).fit(X_tr, y_tr)
        out[f"Constant ({strat} of training)"] = metrics(y_te, m.predict(X_te))
    return pd.DataFrame(out).T
