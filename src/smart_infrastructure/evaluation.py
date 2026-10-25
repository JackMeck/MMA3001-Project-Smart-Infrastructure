from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import Config
from .data import FeatureSet


TIME_COLS = ["hour_sin", "hour_cos", "is_weekend"]


class ScheduleBaseline:
    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.table_ = pd.Series(y).groupby(self._key(X)).mean().to_dict()
        self.mean_ = float(y.mean())
        self.n_features_in_ = X.shape[1]
        return self

    @staticmethod
    def _key(X: np.ndarray) -> np.ndarray:
        angle = np.arctan2(X[:, 0], X[:, 1]) % (2 * np.pi)
        slot = np.round(angle / (2 * np.pi) * 96).astype(int) % 96
        return (X[:, 2] > 0.5).astype(int) * 100 + slot

    def predict(self, X):
        keys = self._key(np.asarray(X, float))
        return np.array([self.table_.get(int(k), self.mean_) for k in keys])


def _prep(scale: bool) -> List[tuple]:
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return steps


def make_models(cfg_seed: int = 42, baseline_col: str = "co2", include_schedule: bool = False) -> Dict[str, object]:
    from sklearn.compose import ColumnTransformer
    from sklearn.dummy import DummyRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor

    models = {
        "baseline_mean": Pipeline(_prep(False) + [("model", DummyRegressor(strategy="mean"))]),
        "baseline_co2_linear": Pipeline([
            ("select", ColumnTransformer([("c", "passthrough", [baseline_col])])),
            *_prep(False),
            ("model", LinearRegression()),
        ]),
        "linear": Pipeline(_prep(True) + [("model", Ridge(alpha=1.0))]),
        "tree": Pipeline(_prep(False) + [("model", DecisionTreeRegressor(max_depth=8, min_samples_leaf=20, random_state=cfg_seed))]),
        "svr": Pipeline(_prep(True) + [("model", SVR(kernel="rbf", C=10.0, epsilon=0.5))]),
        "nn": Pipeline(_prep(True) + [("model", MLPRegressor(hidden_layer_sizes=(32, 16), early_stopping=True, max_iter=800, random_state=cfg_seed))]),
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
    return {
        "linear": {"model__alpha": [0.1, 1.0, 10.0, 100.0]},
        "tree": {"model__max_depth": [3, 5, 8, 12], "model__min_samples_leaf": [5, 20, 50]},
        "svr": {"model__C": [1.0, 10.0], "model__epsilon": [0.1, 0.5]},
        "nn": {"model__alpha": [1e-4, 1e-2]},
    }


def chronological_split(meta: pd.DataFrame, test_fraction: float, gap_min: float) -> Tuple[np.ndarray, np.ndarray]:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in (0, 1).")
    ts = meta["timestamp"]
    cutoff = ts.sort_values().iloc[int((1 - test_fraction) * (len(ts) - 1))]
    gap = pd.Timedelta(minutes=gap_min)
    train = np.where(ts < cutoff - gap)[0]
    test = np.where(ts >= cutoff)[0]
    return train, test


def time_block_folds(timestamps: pd.Series, n_splits: int, gap_min: float) -> List[Tuple[np.ndarray, np.ndarray]]:
    ts = pd.Series(timestamps).reset_index(drop=True)
    edges = ts.quantile(np.linspace(0, 1, n_splits + 2)).to_numpy()
    gap = pd.Timedelta(minutes=gap_min)
    folds = []
    for i in range(1, n_splits + 1):
        lo, hi = edges[i], edges[i + 1]
        val = np.where((ts >= lo) & ((ts < hi) if i < n_splits else (ts <= hi)))[0]
        train = np.where(ts < lo - gap)[0]
        if len(train) and len(val):
            folds.append((train, val))
    return folds


def leave_one_room_out(meta: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    rooms = meta["room"].to_numpy()
    for r in np.unique(rooms):
        yield np.where(rooms != r)[0], np.where(rooms == r)[0]


def regression_metrics(y_true, y_pred, cfg: Config) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.clip(np.asarray(y_pred, dtype=float), 0.0, None)
    err = yp - yt
    thr = cfg.occupied_threshold
    occ_t, occ_p = yt >= thr, yp >= thr
    recall = float((occ_t & occ_p).sum() / occ_t.sum()) if occ_t.any() else np.nan
    fa = float((~occ_t & occ_p).sum() / (~occ_t).sum()) if (~occ_t).any() else np.nan
    return {
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "r2": float(r2_score(yt, yp)) if len(yt) > 1 else np.nan,
        "bias": float(err.mean()),
        "under_rate": float((err < -0.5).mean()),
        "asym_cost": float(np.mean(cfg.cost_under * np.clip(-err, 0, None) + cfg.cost_over * np.clip(err, 0, None))),
        "occ_recall": recall,
        "occ_false_alarm": fa,
    }


def cross_validate(models: Dict, X: pd.DataFrame, y: pd.Series, folds, cfg: Config) -> pd.DataFrame:
    rows = []
    for name, est in models.items():
        for k, (tr, va) in enumerate(folds):
            m = clone(est).fit(X.iloc[tr], y.iloc[tr])
            met = regression_metrics(y.iloc[va], m.predict(X.iloc[va]), cfg)
            rows.append({"model": name, "fold": k, **met})
    return pd.DataFrame(rows)


def residual_tables(fs: FeatureSet, idx: np.ndarray, y_pred, cfg: Config) -> Dict[str, pd.DataFrame]:
    d = fs.meta.iloc[idx].copy()
    d["y"] = fs.y.iloc[idx].to_numpy()
    d["pred"] = np.clip(np.asarray(y_pred, float), 0, None)
    d["err"] = d["pred"] - d["y"]
    d["abs_err"] = d["err"].abs()
    d["occ_band"] = pd.cut(d["y"], [-0.5, 0.5, 5.5, 15.5, 1e9], labels=["0", "1-5", "6-15", ">15"])
    d["hour_band"] = pd.cut(d["timestamp"].dt.hour, [-1, 5, 8, 12, 17, 20, 23], labels=["00-05", "06-08", "09-12", "13-17", "18-20", "21-23"])
    out = {}
    for key, col in (("by_occupancy", "occ_band"), ("by_hour", "hour_band")):
        out[key] = (
            d.groupby(col, observed=True)
            .agg(n=("y", "size"), mae=("abs_err", "mean"), bias=("err", "mean"))
            .reset_index()
        )
    return out


def plot_diagnostics(y_true, y_pred, timestamps, path: str, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    yt = np.asarray(y_true, float)
    yp = np.clip(np.asarray(y_pred, float), 0, None)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    ax[0].scatter(yt, yp, s=4, alpha=0.3)
    lim = [0, max(yt.max(), yp.max()) * 1.05]
    ax[0].plot(lim, lim, "k--", lw=1)
    ax[0].set(xlabel="Observed occupancy", ylabel="Predicted", title="Observed vs predicted")
    ax[1].scatter(yp, yp - yt, s=4, alpha=0.3)
    ax[1].axhline(0, color="k", lw=1)
    ax[1].set(xlabel="Predicted", ylabel="Residual (pred - obs)", title="Residuals")
    ts = pd.Series(timestamps).reset_index(drop=True)
    n = min(len(ts), int(7 * 24 * 60 / max(1, (ts.diff().median().total_seconds() / 60))))
    ax[2].plot(ts[:n], yt[:n], label="observed", lw=1)
    ax[2].plot(ts[:n], yp[:n], label="predicted", lw=1)
    ax[2].set(title="First test week", ylabel="People")
    ax[2].legend()
    ax[2].tick_params(axis="x", rotation=30)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def feature_group_ablation(est, fs: FeatureSet, train_idx, folds, cfg: Config) -> pd.DataFrame:
    X = fs.X.iloc[train_idx].reset_index(drop=True)
    y = fs.y.iloc[train_idx].reset_index(drop=True)

    def cv_rmse(cols: List[str]) -> float:
        out = []
        for tr, va in folds:
            m = clone(est).fit(X.iloc[tr][cols], y.iloc[tr])
            out.append(regression_metrics(y.iloc[va], m.predict(X.iloc[va][cols]), cfg)["rmse"])
        return float(np.mean(out))

    all_cols = list(X.columns)
    base = cv_rmse(all_cols)
    rows = [{"dropped_group": "none", "n_features": len(all_cols), "cv_rmse": base, "delta_rmse": 0.0}]

    def add(label: str, drop: List[str]) -> None:
        keep = [c for c in all_cols if c not in drop]
        if keep and len(keep) < len(all_cols):
            r = cv_rmse(keep)
            rows.append({"dropped_group": label, "n_features": len(keep), "cv_rmse": r, "delta_rmse": r - base})

    for g, cols in fs.groups.items():
        add(g, cols)
    sensor_cols = [c for g, cols in fs.groups.items() if g != "time" for c in cols]
    add("ALL sensor features (time only)", sensor_cols)
    for v in cfg.env_cols:
        add(f"variable: {v}", [c for c in sensor_cols if c == v or c.startswith(v + "_")])
    return pd.DataFrame(rows)


def hyperparameter_sweep(est, param: str, values, X, y, folds, cfg: Config) -> pd.DataFrame:
    rows = []
    for v in values:
        r, a = [], []
        for tr, va in folds:
            m = clone(est).set_params(**{param: v}).fit(X.iloc[tr], y.iloc[tr])
            met = regression_metrics(y.iloc[va], m.predict(X.iloc[va]), cfg)
            r.append(met["rmse"])
            a.append(met["mae"])
        rows.append({"value": v, "cv_rmse": float(np.mean(r)), "cv_mae": float(np.mean(a))})
    return pd.DataFrame(rows)


def training_size_curve(est, X, y, folds, cfg: Config, fractions=(0.1, 0.25, 0.5, 0.75, 1.0)) -> pd.DataFrame:
    tr_full, va = folds[-1]
    rows = []
    for f in fractions:
        tr = tr_full[: max(10, int(len(tr_full) * f))]
        m = clone(est).fit(X.iloc[tr], y.iloc[tr])
        met = regression_metrics(y.iloc[va], m.predict(X.iloc[va]), cfg)
        rows.append({"fraction": f, "n_train": len(tr), "val_rmse": met["rmse"], "val_mae": met["mae"]})
    return pd.DataFrame(rows)


def noise_robustness(fitted, clean_df: pd.DataFrame, cfg: Config, test_start, levels=(0.0, 0.05, 0.1, 0.2, 0.5), seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for lv in levels:
        d = clean_df.copy()
        for c in cfg.env_cols:
            d[c] = d[c] + rng.normal(0, lv * d[c].std(), len(d))
        fs = build_features(d, cfg)
        mask = (fs.meta["timestamp"] >= test_start).to_numpy()
        met = regression_metrics(fs.y[mask], fitted.predict(fs.X[mask]), cfg)
        rows.append({"noise_frac": lv, "rmse": met["rmse"], "mae": met["mae"], "bias": met["bias"]})
    return pd.DataFrame(rows)


def resolution_sweep(clean_builder, est, cfg: Config, rules=("1min", "5min", "15min", "30min")) -> pd.DataFrame:
    rows = []
    for rule in rules:
        c = replace(cfg, resample_rule=rule)
        c = replace(c, windows_min=[w for w in cfg.windows_min if w >= c.step_minutes])
        if not c.windows_min:
            rows.append({"rule": rule, "n_rows": 0, "cv_rmse": np.nan, "cv_mae": np.nan, "note": "no window >= grid step"})
            continue
        fs = build_features(clean_builder(c), c)
        tr, _ = chronological_split(fs.meta, c.test_fraction, c.cv_gap_min)
        Xtr = fs.X.iloc[tr].reset_index(drop=True)
        ytr = fs.y.iloc[tr].reset_index(drop=True)
        folds = time_block_folds(fs.meta["timestamp"].iloc[tr], c.cv_splits, c.cv_gap_min)
        r, a = [], []
        for t_i, v_i in folds:
            m = clone(est).fit(Xtr.iloc[t_i], ytr.iloc[t_i])
            met = regression_metrics(ytr.iloc[v_i], m.predict(Xtr.iloc[v_i]), c)
            r.append(met["rmse"])
            a.append(met["mae"])
        rows.append({"rule": rule, "n_rows": len(fs.y), "cv_rmse": float(np.mean(r)), "cv_mae": float(np.mean(a)), "note": f"windows={c.windows_min}"})
    return pd.DataFrame(rows)


def time_fit_predict(est, X_train, y_train, X_test, repeats: int = 3) -> Dict[str, float]:
    import time as _time
    import tracemalloc

    fits, preds, peaks = [], [], []
    for _ in range(repeats):
        m = clone(est)
        tracemalloc.start()
        t0 = _time.perf_counter()
        m.fit(X_train, y_train)
        t1 = _time.perf_counter()
        m.predict(X_test)
        t2 = _time.perf_counter()
        peaks.append(tracemalloc.get_traced_memory()[1] / 1e6)
        tracemalloc.stop()
        fits.append(t1 - t0)
        preds.append(t2 - t1)
    p = float(np.median(preds))
    return {"fit_s": float(np.median(fits)), "predict_s": p, "predict_us_per_row": 1e6 * p / max(1, len(X_test)), "peak_mem_mb": float(np.max(peaks))}


def predict_flops(fitted_pipeline, n_rows: int, n_features: int) -> Dict[str, float]:
    m = fitted_pipeline.named_steps["model"]
    name = type(m).__name__
    p = n_features
    if name in ("LinearRegression", "Ridge"):
        per_row, params = 2 * p, p + 1
    elif name == "DecisionTreeRegressor":
        per_row, params = m.get_depth(), 5 * m.tree_.node_count
    elif name == "SVR":
        sv = m.support_vectors_.shape[0]
        per_row, params = sv * (3 * p + 2), sv * (p + 1)
    elif name == "MLPRegressor":
        sizes = [p, *m.hidden_layer_sizes, 1]
        per_row = 2 * sum(a * b for a, b in zip(sizes[:-1], sizes[1:]))
        params = sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))
    elif name == "ScheduleBaseline":
        per_row, params = 4, 2 * len(m.table_)
    elif name == "DummyRegressor":
        per_row, params = 1, 1
    else:
        return {"flops": np.nan, "bytes": np.nan, "arith_intensity": np.nan}
    flops = float(per_row * n_rows)
    nbytes = 8.0 * (n_rows * p + params + n_rows)
    return {"flops": flops, "bytes": nbytes, "arith_intensity": flops / nbytes}


def profile_call(fn, *args, top: int = 8, **kwargs) -> str:
    import cProfile
    import io as _io
    import pstats

    pr = cProfile.Profile()
    pr.enable()
    fn(*args, **kwargs)
    pr.disable()
    buf = _io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(top)
    return buf.getvalue()


def performance_table(models: Dict, X_train, y_train, X_test, repeats: int = 3) -> pd.DataFrame:
    rows = []
    for name, est in models.items():
        t = time_fit_predict(est, X_train, y_train, X_test, repeats)
        fitted = clone(est).fit(X_train, y_train)
        n_feat = fitted.named_steps["model"].n_features_in_ if hasattr(fitted.named_steps["model"], "n_features_in_") else X_train.shape[1]
        f = predict_flops(fitted, len(X_test), n_feat)
        rows.append({"model": name, **t, **f})
    return pd.DataFrame(rows)


__all__ = [
    "ScheduleBaseline",
    "make_models",
    "param_grids",
    "chronological_split",
    "time_block_folds",
    "leave_one_room_out",
    "regression_metrics",
    "cross_validate",
    "residual_tables",
    "plot_diagnostics",
    "feature_group_ablation",
    "hyperparameter_sweep",
    "training_size_curve",
    "noise_robustness",
    "resolution_sweep",
    "time_fit_predict",
    "predict_flops",
    "profile_call",
    "performance_table",
]
