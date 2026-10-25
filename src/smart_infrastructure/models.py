"""Baselines and candidate regression models (all taught in Week 5).

Every model is an sklearn ``Pipeline`` whose final step is named ``model`` and whose
imputer/scaler steps are fit on the training data only (no preprocessing leakage).
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


TIME_COLS = ["hour_sin", "hour_cos", "is_weekend"]


class ScheduleBaseline(BaseEstimator, RegressorMixin):
    """Mean occupancy by (weekday/weekend, 15-minute time-of-day slot).

    Uses time of week only, no sensor data. A sensor-based model has to beat this
    to show that the environmental readings add information beyond the timetable.
    Expects the three columns in ``TIME_COLS`` (hour_sin, hour_cos, is_weekend).
    """

    @staticmethod
    def _key(X: np.ndarray) -> np.ndarray:
        angle = np.arctan2(X[:, 0], X[:, 1]) % (2 * np.pi)
        slot = np.round(angle / (2 * np.pi) * 96).astype(int) % 96
        return (X[:, 2] > 0.5).astype(int) * 100 + slot

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.table_ = pd.Series(y).groupby(self._key(X)).mean().to_dict()
        self.mean_ = float(y.mean())
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        keys = self._key(np.asarray(X, float))
        return np.array([self.table_.get(int(k), self.mean_) for k in keys])


def _prep(scale: bool) -> List[tuple]:
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return steps


def make_models(cfg_seed: int = 42, baseline_col: str = "co2",
                include_schedule: bool = False) -> Dict[str, Pipeline]:
    """Create the baselines and candidate models.

    Args:
        cfg_seed: Random seed for stochastic models.
        baseline_col: Single raw sensor column for the simple physical baseline.
        include_schedule: Add the time-of-week-only ``baseline_schedule`` (needs the
            time features).

    Returns:
        Ordered dict name -> unfitted Pipeline. Names starting with ``baseline_``
        are reference models, not selection candidates.
    """
    models = {
        "baseline_mean": Pipeline(
            _prep(False) + [("model", DummyRegressor(strategy="mean"))]
        ),
        "baseline_co2_linear": Pipeline(
            [("select", ColumnTransformer([("c", "passthrough", [baseline_col])])),
             *_prep(False), ("model", LinearRegression())]
        ),
        "linear": Pipeline(_prep(True) + [("model", Ridge(alpha=1.0))]),
        "tree": Pipeline(
            _prep(False)
            + [("model", DecisionTreeRegressor(max_depth=8, min_samples_leaf=20,
                                               random_state=cfg_seed))]
        ),
        "svr": Pipeline(_prep(True) + [("model", SVR(kernel="rbf", C=10.0, epsilon=0.5))]),
        "nn": Pipeline(
            _prep(True)
            + [("model", MLPRegressor(hidden_layer_sizes=(32, 16), early_stopping=True,
                                      max_iter=800, random_state=cfg_seed))]
        ),
    }
    if include_schedule:
        sched = Pipeline([
            ("select", ColumnTransformer([("t", "passthrough", TIME_COLS)])),
            ("model", ScheduleBaseline()),
        ])
        out = {"baseline_mean": models.pop("baseline_mean"), "baseline_schedule": sched}
        out.update(models)
        return out
    return models


def param_grids() -> Dict[str, Dict[str, list]]:
    """Small hyperparameter grids searched on the training folds only."""
    return {
        "linear": {"model__alpha": [0.1, 1.0, 10.0, 100.0]},
        "tree": {"model__max_depth": [3, 5, 8, 12],
                 "model__min_samples_leaf": [5, 20, 50]},
        "svr": {"model__C": [1.0, 10.0], "model__epsilon": [0.1, 0.5]},
        "nn": {"model__alpha": [1e-4, 1e-2]},
    }
