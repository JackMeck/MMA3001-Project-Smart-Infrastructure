"""Baseline and candidate regression models.

Every model is returned as a *factory* (a function that builds a fresh, unfitted
model) so that each cross-validation fold starts from scratch. Sensor models are
sklearn pipelines: select columns -> median-impute -> standardise -> regress. The
imputer and scaler are fitted on the training rows only, so no information from the
validation or test rows leaks into preprocessing.
"""

from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .config import Config


class ScheduleBaseline(BaseEstimator, RegressorMixin):
    """Predict the training mean for the same (weekend, hour) slot; ignores sensors.

    A sensor model has to beat this to show the sensors add information beyond the
    timetable. Expects a DataFrame with ``weekend`` and ``hour`` columns. Slots not
    seen in training fall back to the overall training mean.
    """

    def _key(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([f"{int(w)}_{int(round(h))}"
                         for w, h in zip(X["weekend"], X["hour"])])

    def fit(self, X: pd.DataFrame, y) -> "ScheduleBaseline":
        """Learn the mean target of each (weekend, hour) slot.

        Args:
            X: Frame with ``weekend`` and ``hour`` columns.
            y: Target values (people).

        Returns:
            ``self``.
        """
        self.table_ = pd.Series(np.asarray(y)).groupby(self._key(X)).mean().to_dict()
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the slot mean (or the overall mean for unseen slots).

        Args:
            X: Frame with ``weekend`` and ``hour`` columns.

        Returns:
            Predicted people count per row.
        """
        return np.array([self.table_.get(k, self.mean_) for k in self._key(X)])


def sensors(cols: List[str]) -> ColumnTransformer:
    """Column selector that passes only ``cols`` through to the model.

    Args:
        cols: Column names to keep.

    Returns:
        A :class:`~sklearn.compose.ColumnTransformer`.
    """
    return ColumnTransformer([("s", "passthrough", list(cols))])


def linear_on(cols: List[str]) -> Callable:
    """Factory of a linear-regression pipeline that only sees ``cols``.

    Args:
        cols: Input columns.

    Returns:
        Zero-argument function returning a fresh unfitted pipeline.
    """
    return lambda: make_pipeline(sensors(cols), SimpleImputer(strategy="median"),
                                 StandardScaler(), LinearRegression())


def svr_on(cols: List[str], C: float, epsilon: float) -> Callable:
    """Factory of an RBF support-vector-regression pipeline that only sees ``cols``.

    Args:
        cols: Input columns.
        C: Regularisation strength (larger = fit the training data harder).
        epsilon: Half-width of the no-penalty tube around the prediction.

    Returns:
        Zero-argument function returning a fresh unfitted pipeline.
    """
    return lambda: make_pipeline(sensors(cols), SimpleImputer(strategy="median"),
                                 StandardScaler(), SVR(C=C, epsilon=epsilon))


def candidate_models(sensor_cols: List[str]) -> Dict[str, Callable]:
    """First-round comparison: schedule baseline, linear regression and SVR on sensors.

    Args:
        sensor_cols: Sensor-derived columns (no time-of-week features).

    Returns:
        Ordered dict name -> factory.
    """
    return {
        "Baseline (schedule)": lambda: ScheduleBaseline(),
        "Linear Regression": linear_on(sensor_cols),
        "SVR": svr_on(sensor_cols, C=10, epsilon=0.5),
    }


def final_models(cfg: Config) -> Dict[str, Callable]:
    """Final models: schedule baseline plus lean linear and lean SVR models.

    The lean inputs and SVR settings come from ``cfg`` and were chosen on
    cross-validation only, never on the test set.

    Args:
        cfg: Configuration (``lean_cols``, ``svr_C``, ``svr_epsilon``).

    Returns:
        Ordered dict name -> factory.
    """
    return {
        "Baseline (schedule)": lambda: ScheduleBaseline(),
        "Linear Regression": linear_on(cfg.lean_cols),
        "SVR": svr_on(cfg.lean_cols, cfg.svr_C, cfg.svr_epsilon),
    }
