"""Candidate predictors: schedule baseline, linear regression and RBF SVR.

All sensor models are scikit-learn pipelines of *column selection -> median
imputation -> standardisation -> regressor*, so imputation and scaling statistics
are learned from each fit's training data only (no leakage of validation/test data).
Every ``*_on`` / ``build_*`` function returns a **factory** (a zero-argument
callable) so that each fold gets a fresh, unfitted model.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from . import config

ModelFactory = Callable[[], BaseEstimator]


class ScheduleBaseline(BaseEstimator, RegressorMixin):
    """Time-of-week baseline that ignores every sensor.

    Predicts the training-set mean headcount for the same (weekend, hour) slot, i.e.
    what a timetable alone would give. Unseen slots fall back to the overall mean.

    Attributes:
        table_: Mapping ``"<weekend>_<rounded hour>"`` -> training mean headcount.
        mean_: Overall training mean (fallback).
    """

    @staticmethod
    def _key(X: pd.DataFrame) -> np.ndarray:
        """Build the slot key for each row from the ``weekend`` and ``hour`` columns."""
        return np.array([f"{int(w)}_{int(round(h))}" for w, h in zip(X["weekend"], X["hour"])])

    def fit(self, X: pd.DataFrame, y) -> "ScheduleBaseline":
        """Learn the mean headcount of each slot.

        Args:
            X: DataFrame with ``weekend`` (0/1) and ``hour`` (0-23.75) columns.
            y: Headcounts (people), same length as ``X``.

        Returns:
            ``self``.
        """
        self.table_ = pd.Series(np.asarray(y)).groupby(self._key(X)).mean().to_dict()
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict by slot lookup.

        Args:
            X: DataFrame with ``weekend`` and ``hour`` columns.

        Returns:
            Array of predicted headcounts (people).
        """
        return np.array([self.table_.get(k, self.mean_) for k in self._key(X)])


def _preprocess(cols: Sequence[str]) -> list:
    """Shared preprocessing steps: select ``cols``, median-impute, standardise."""
    return [ColumnTransformer([("c", "passthrough", list(cols))]),
            SimpleImputer(strategy="median"), StandardScaler()]


def linear_on(cols: Sequence[str]) -> ModelFactory:
    """Factory for ordinary least-squares regression on the listed columns.

    Args:
        cols: Input column names the model may see.

    Returns:
        Zero-argument callable producing a fresh pipeline.
    """
    return lambda: make_pipeline(*_preprocess(cols), LinearRegression())


def svr_on(cols: Sequence[str], C: float, epsilon: float) -> ModelFactory:
    """Factory for an RBF support vector regression on the listed columns.

    ``gamma`` is the scikit-learn default (``"scale"``), not tuned.

    Args:
        cols: Input column names the model may see.
        C: Error penalty (small = smoother model).
        epsilon: Half-width (people) of the zero-loss band.

    Returns:
        Zero-argument callable producing a fresh pipeline.
    """
    return lambda: make_pipeline(*_preprocess(cols), SVR(C=C, epsilon=epsilon))


def candidate_models(sensor_cols: Sequence[str] = config.SENSOR_COLS) -> dict[str, ModelFactory]:
    """First-round candidates (all sensors, no time inputs, untuned SVR).

    Args:
        sensor_cols: Sensor columns the sensor models use.

    Returns:
        Mapping name -> factory for the schedule baseline, linear regression and SVR
        (C = 10, epsilon = 0.5).
    """
    return {
        "Baseline (schedule)": lambda: ScheduleBaseline(),
        "Linear Regression": linear_on(sensor_cols),
        "SVR": svr_on(sensor_cols, C=10, epsilon=0.5),
    }


def final_models(C: float = config.SVR_C, epsilon: float = config.SVR_EPSILON) -> dict[str, ModelFactory]:
    """The models carried into the final evaluation (chosen on cross-validation only).

    Args:
        C, epsilon: SVR settings (defaults are the cross-validated optimum).

    Returns:
        Mapping name -> factory for the schedule baseline, lean linear regression and
        lean SVR.
    """
    return {
        "Baseline (schedule)": lambda: ScheduleBaseline(),
        "Linear Regression": linear_on(config.LEAN),
        "SVR": svr_on(config.LEAN, C=C, epsilon=epsilon),
    }


def constant_model(strategy: str = "median") -> ModelFactory:
    """Factory for a constant predictor (no inputs used).

    Args:
        strategy: ``"median"`` or ``"mean"`` of the training targets.
    """
    return lambda: DummyRegressor(strategy=strategy)
