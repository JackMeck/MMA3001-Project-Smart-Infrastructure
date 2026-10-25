"""Metrics, chronological splitting and time-series cross-validation.

All splits are chronological. Occupancy and CO2 are strongly autocorrelated, so
random folds would place near-identical neighbours in training and validation and
overstate accuracy. A ``gap`` (default 8 steps = 2 h, longer than the ~1 h CO2
response lag) separates every training block from its validation/test block.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from . import config
from .models import ModelFactory


def metrics(y, p) -> dict[str, float]:
    """Score predictions against observed headcounts.

    Predictions are clipped at zero first because a negative headcount is meaningless.

    Args:
        y: Observed headcounts (people).
        p: Predicted headcounts (people).

    Returns:
        Dict with ``MAE`` and ``RMSE`` (people), ``R2`` (dimensionless) and ``bias``
        (mean of prediction minus observation, people; positive = over-prediction).
    """
    p = np.clip(p, 0, None)
    return {"MAE": mean_absolute_error(y, p), "RMSE": mean_squared_error(y, p) ** 0.5,
            "R2": r2_score(y, p), "bias": float(np.mean(p - y))}


class Split(NamedTuple):
    """Chronological train/test split (see :func:`chronological_split`)."""
    X_tr: pd.DataFrame
    y_tr: pd.Series
    X_te: pd.DataFrame
    y_te: pd.Series


def chronological_split(df: pd.DataFrame, features: list[str] = config.FEATURES,
                        target: str = config.TARGET, train_frac: float = config.TRAIN_FRAC,
                        gap: int = config.GAP) -> Split:
    """Split a time-ordered table into train and held-out test blocks.

    The first ``train_frac`` of rows (minus ``gap`` rows) train the model; the rest is
    the test set. The ``gap`` rows in between are discarded.

    Args:
        df: Time-ordered modelling table.
        features: Input columns.
        target: Target column.
        train_frac: Fraction of rows before the test block.
        gap: Rows discarded between training and test.

    Returns:
        :class:`Split` with ``X_tr, y_tr, X_te, y_te``.
    """
    X, y = df[features], df[target]
    cut = int(train_frac * len(df))
    return Split(X.iloc[:cut - gap], y.iloc[:cut - gap], X.iloc[cut:], y.iloc[cut:])


def cv_folds(X_tr: pd.DataFrame, n_splits: int = config.N_SPLITS, gap: int = config.GAP):
    """Expanding-window fold indices (each trains on all earlier data).

    Args:
        X_tr: Training table (time-ordered).
        n_splits: Number of folds.
        gap: Rows skipped between each training block and its validation block.

    Returns:
        Iterator of ``(train_idx, val_idx)`` positional index arrays.
    """
    return TimeSeriesSplit(n_splits=n_splits, gap=gap).split(X_tr)


def cv_scores(make: ModelFactory, X_tr: pd.DataFrame, y_tr: pd.Series,
              n_splits: int = config.N_SPLITS, gap: int = config.GAP) -> pd.DataFrame:
    """Per-fold metrics for one model under expanding-window cross-validation.

    Args:
        make: Model factory.
        X_tr, y_tr: Training period inputs and targets.
        n_splits, gap: See :func:`cv_folds`.

    Returns:
        DataFrame with one row per fold and columns ``MAE, RMSE, R2, bias``.
    """
    rows = [metrics(y_tr.iloc[va], make().fit(X_tr.iloc[tr], y_tr.iloc[tr]).predict(X_tr.iloc[va]))
            for tr, va in cv_folds(X_tr, n_splits, gap)]
    return pd.DataFrame(rows)


def cv_table(models: dict[str, ModelFactory], X_tr: pd.DataFrame, y_tr: pd.Series,
             n_splits: int = config.N_SPLITS, gap: int = config.GAP) -> pd.DataFrame:
    """Mean cross-validated metrics for several models.

    Args:
        models: Mapping name -> factory.
        X_tr, y_tr: Training period inputs and targets.
        n_splits, gap: See :func:`cv_folds`.

    Returns:
        DataFrame indexed by model name (input order) with mean fold metrics.
    """
    out = {name: cv_scores(make, X_tr, y_tr, n_splits, gap).mean() for name, make in models.items()}
    return pd.DataFrame(out).T
