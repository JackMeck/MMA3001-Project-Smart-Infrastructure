"""Validation, metrics, sensitivity studies and performance estimates.

Combines four related concerns used to judge a model:

* chronological split, time-block / leave-one-room-out cross-validation, metrics and
  residual diagnostics;
* sensitivity and robustness studies (feature-group ablation, hyper-parameter sweep,
  training-size curve, noise robustness, time-resolution sweep);
* timing, memory, profiling and analytic FLOP estimates.

All studies that could influence model choice use the TRAINING folds only. The noise
study runs after selection on the held-out test period and does not feed back into
selection. FLOP counts are analytic ESTIMATES for the *prediction* step and should be
reported as order-of-magnitude figures next to measured timings.
"""

import cProfile
import io as _io
import pstats
import time
import tracemalloc
from dataclasses import replace
from typing import Callable, Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import Config
from .features import FeatureSet, build_features


# ------------------------------------------------------------------ splits
def chronological_split(
    meta: pd.DataFrame, test_fraction: float, gap_min: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Train on earlier data, test on the latest ``test_fraction`` of time.

    A buffer of ``gap_min`` minutes before the cut-off is removed from training so
    that overlapping rolling windows cannot straddle the boundary.

    Args:
        meta: Frame with a ``timestamp`` column (one row per sample).
        test_fraction: Fraction of unique timestamps reserved for the final test.
        gap_min: Training/test buffer in minutes.

    Returns:
        ``(train_idx, test_idx)`` positional index arrays.

    Raises:
        ValueError: If ``test_fraction`` is not in (0, 1).
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in (0, 1).")
    ts = meta["timestamp"]
    cutoff = ts.sort_values().iloc[int((1 - test_fraction) * (len(ts) - 1))]
    gap = pd.Timedelta(minutes=gap_min)
    train = np.where(ts < cutoff - gap)[0]
    test = np.where(ts >= cutoff)[0]
    return train, test


def time_block_folds(
    timestamps: pd.Series, n_splits: int, gap_min: float
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Expanding-window time-ordered CV folds.

    Time is cut into ``n_splits + 1`` equal-count blocks; fold ``i`` trains on all
    blocks before block ``i + 1`` (minus a gap) and validates on block ``i + 1``.

    Args:
        timestamps: Timestamps of the training rows (positional order).
        n_splits: Number of folds.
        gap_min: Buffer (minutes) removed from the end of each training window.

    Returns:
        List of ``(train_idx, val_idx)`` positional index arrays (indices refer to
        positions within ``timestamps``).
    """
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
    """Grouped split: each room held out in turn (a 'new room' generalisation test).

    Args:
        meta: Frame with a ``room`` column.

    Yields:
        ``(train_idx, val_idx)`` with zero shared rooms.
    """
    rooms = meta["room"].to_numpy()
    for r in np.unique(rooms):
        yield np.where(rooms != r)[0], np.where(rooms == r)[0]


# ----------------------------------------------------------------- metrics
def regression_metrics(y_true, y_pred, cfg: Config) -> Dict[str, float]:
    """Error metrics in people, plus decision-oriented measures.

    Metrics:
        * ``mae``, ``rmse`` (people), ``r2`` (supporting only).
        * ``bias``: mean(pred - true); negative = under-estimation.
        * ``under_rate``: fraction of samples with pred < true - 0.5.
        * ``asym_cost``: mean of ``cost_under*max(true-pred,0) + cost_over*max(pred-true,0)``.
        * ``occ_recall`` / ``occ_false_alarm``: detection of 'occupied' (>= ``occupied_threshold``)
          using the SAME threshold on observed and predicted values; NaN if a class is absent.

    Args:
        y_true: Observed counts.
        y_pred: Predicted counts (clipped at zero before scoring).
        cfg: Workflow configuration (cost weights).

    Returns:
        Dictionary of metric name -> value.
    """
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
        "asym_cost": float(np.mean(cfg.cost_under * np.clip(-err, 0, None)
                                   + cfg.cost_over * np.clip(err, 0, None))),
        "occ_recall": recall,
        "occ_false_alarm": fa,
    }


def cross_validate(models: Dict, X: pd.DataFrame, y: pd.Series,
                   folds, cfg: Config) -> pd.DataFrame:
    """Score each model on each fold (fresh clone per fold).

    Args:
        models: name -> unfitted estimator.
        X: Training features.
        y: Training target.
        folds: Iterable of ``(train_idx, val_idx)``.
        cfg: Workflow configuration.

    Returns:
        Long DataFrame with columns ``model``, ``fold`` and the metrics.
    """
    rows = []
    for name, est in models.items():
        for k, (tr, va) in enumerate(folds):
            m = clone(est).fit(X.iloc[tr], y.iloc[tr])
            met = regression_metrics(y.iloc[va], m.predict(X.iloc[va]), cfg)
            rows.append({"model": name, "fold": k, **met})
    return pd.DataFrame(rows)


# -------------------------------------------------------------- diagnostics
def residual_tables(fs: FeatureSet, idx: np.ndarray, y_pred, cfg: Config
                    ) -> Dict[str, pd.DataFrame]:
    """Regional residual analysis (where are the errors?).

    Args:
        fs: Feature set.
        idx: Positional indices of the evaluated rows.
        y_pred: Predictions for those rows.
        cfg: Workflow configuration.

    Returns:
        Dict with ``by_occupancy`` (error by true-count band) and ``by_hour``
        (error by time-of-day band) tables of count, MAE and bias.
    """
    d = fs.meta.iloc[idx].copy()
    d["y"] = fs.y.iloc[idx].to_numpy()
    d["pred"] = np.clip(np.asarray(y_pred, float), 0, None)
    d["err"] = d["pred"] - d["y"]
    d["abs_err"] = d["err"].abs()
    d["occ_band"] = pd.cut(d["y"], [-0.5, 0.5, 5.5, 15.5, 1e9],
                           labels=["0", "1-5", "6-15", ">15"])
    d["hour_band"] = pd.cut(d["timestamp"].dt.hour, [-1, 5, 8, 12, 17, 20, 23],
                            labels=["00-05", "06-08", "09-12", "13-17", "18-20", "21-23"])
    out = {}
    for key, col in (("by_occupancy", "occ_band"), ("by_hour", "hour_band")):
        out[key] = (d.groupby(col, observed=True)
                    .agg(n=("y", "size"), mae=("abs_err", "mean"), bias=("err", "mean"))
                    .reset_index())
    return out


def plot_diagnostics(y_true, y_pred, timestamps, path: str, title: str = "") -> None:
    """Save observed-vs-predicted, residual-vs-predicted and a time-series panel.

    Args:
        y_true: Observed counts.
        y_pred: Predicted counts.
        timestamps: Timestamps aligned with the values.
        path: Output PNG path.
        title: Figure title.
    """
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


# ------------------------------------------------------------------ sensitivity
def feature_group_ablation(est, fs: FeatureSet, train_idx, folds, cfg: Config) -> pd.DataFrame:
    """Remove features and report the cross-validated change in RMSE.

    Rows:
        * ``none``: all features (reference).
        * one row per feature group (raw / rolling / trend / time) dropped alone.
        * ``ALL sensor features (time only)``: raw + rolling + trend dropped TOGETHER.
          This is the fair test of whether the sensors add anything beyond time of day,
          because the three sensor groups are built from the same readings and each can
          stand in for the others when only one is removed.
        * ``variable: <name>``: every feature derived from one sensor variable dropped.

    A positive ``delta_rmse`` means the removed features were helping.

    Args:
        est: Unfitted selected model.
        fs: Feature set.
        train_idx: Positional indices of the training rows.
        folds: Time-block folds over the training rows.
        cfg: Workflow configuration.

    Returns:
        Table with ``dropped_group``, ``n_features``, ``cv_rmse``, ``delta_rmse``.
    """
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
    rows = [{"dropped_group": "none", "n_features": len(all_cols),
             "cv_rmse": base, "delta_rmse": 0.0}]

    def add(label: str, drop: List[str]) -> None:
        keep = [c for c in all_cols if c not in drop]
        if keep and len(keep) < len(all_cols):
            r = cv_rmse(keep)
            rows.append({"dropped_group": label, "n_features": len(keep),
                         "cv_rmse": r, "delta_rmse": r - base})

    for g, cols in fs.groups.items():
        add(g, cols)
    sensor_cols = [c for g, cols in fs.groups.items() if g != "time" for c in cols]
    add("ALL sensor features (time only)", sensor_cols)
    for v in cfg.env_cols:
        add(f"variable: {v}", [c for c in sensor_cols if c == v or c.startswith(v + "_")])
    return pd.DataFrame(rows)


def hyperparameter_sweep(est, param: str, values, X, y, folds, cfg: Config) -> pd.DataFrame:
    """Sweep one hyperparameter and report mean CV RMSE/MAE.

    Args:
        est: Unfitted pipeline.
        param: Pipeline parameter name, e.g. ``"model__max_depth"``.
        values: Values to try.
        X: Training features.
        y: Training target.
        folds: Time-block folds.
        cfg: Workflow configuration.

    Returns:
        Table of ``value``, ``cv_rmse``, ``cv_mae``.
    """
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


def training_size_curve(est, X, y, folds, cfg: Config,
                        fractions=(0.1, 0.25, 0.5, 0.75, 1.0)) -> pd.DataFrame:
    """Data-requirement study: train on the earliest fraction, validate on the last fold.

    Args:
        est: Unfitted pipeline.
        X: Training features.
        y: Training target.
        folds: Time-block folds (the last fold's validation block is used).
        cfg: Workflow configuration.
        fractions: Fractions of the last fold's training window to use.

    Returns:
        Table of ``fraction``, ``n_train``, ``val_rmse``, ``val_mae``.
    """
    tr_full, va = folds[-1]
    rows = []
    for f in fractions:
        tr = tr_full[: max(10, int(len(tr_full) * f))]
        m = clone(est).fit(X.iloc[tr], y.iloc[tr])
        met = regression_metrics(y.iloc[va], m.predict(X.iloc[va]), cfg)
        rows.append({"fraction": f, "n_train": len(tr),
                     "val_rmse": met["rmse"], "val_mae": met["mae"]})
    return pd.DataFrame(rows)


def noise_robustness(fitted, clean_df: pd.DataFrame, cfg: Config, test_start,
                     levels=(0.0, 0.05, 0.1, 0.2, 0.5), seed: int = 0) -> pd.DataFrame:
    """Add Gaussian noise (fraction of each sensor's std) to RAW inputs and re-score.

    Noise is added before feature construction so rolling/trend features respond
    consistently. Evaluated on rows at or after ``test_start``.

    Args:
        fitted: Fitted pipeline.
        clean_df: Clean regular-grid frame (all periods).
        cfg: Workflow configuration.
        test_start: First timestamp of the held-out test period.
        levels: Noise levels as fractions of each sensor's standard deviation.
        seed: RNG seed.

    Returns:
        Table of ``noise_frac``, ``rmse``, ``mae``, ``bias``.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for lv in levels:
        d = clean_df.copy()
        for c in cfg.env_cols:
            d[c] = d[c] + rng.normal(0, lv * d[c].std(), len(d))
        fs = build_features(d, cfg)
        mask = (fs.meta["timestamp"] >= test_start).to_numpy()
        met = regression_metrics(fs.y[mask], fitted.predict(fs.X[mask]), cfg)
        rows.append({"noise_frac": lv, "rmse": met["rmse"], "mae": met["mae"],
                     "bias": met["bias"]})
    return pd.DataFrame(rows)


def resolution_sweep(clean_builder, est, cfg: Config,
                     rules=("1min", "5min", "15min", "30min")) -> pd.DataFrame:
    """Input-resolution study: change the grid spacing and re-run CV on training data.

    Args:
        clean_builder: Callable ``cfg -> clean frame`` (re-cleans raw data on the new grid).
        est: Unfitted pipeline.
        cfg: Workflow configuration.
        rules: Pandas resample rules to try.

    Returns:
        Table of ``rule``, ``n_rows``, ``cv_rmse``, ``cv_mae``, ``note``. Windows shorter
        than the grid step are dropped, and rows differ between rules, so scores are
        indicative rather than like-for-like.
    """

    rows = []
    for rule in rules:
        c = replace(cfg, resample_rule=rule)
        # windows shorter than the grid step cannot be formed at this resolution
        c = replace(c, windows_min=[w for w in cfg.windows_min if w >= c.step_minutes])
        if not c.windows_min:
            rows.append({"rule": rule, "n_rows": 0, "cv_rmse": np.nan, "cv_mae": np.nan,
                         "note": "no window >= grid step"})
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
        rows.append({"rule": rule, "n_rows": len(fs.y), "cv_rmse": float(np.mean(r)),
                     "cv_mae": float(np.mean(a)), "note": f"windows={c.windows_min}"})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ performance
def time_fit_predict(est, X_train, y_train, X_test, repeats: int = 3) -> Dict[str, float]:
    """Measure fit time, predict time and peak Python-heap memory.

    Args:
        est: Unfitted estimator (cloned for every repeat).
        X_train: Training features.
        y_train: Training target.
        X_test: Features to predict.
        repeats: Repeats; the median time is reported.

    Returns:
        ``fit_s``, ``predict_s``, ``predict_us_per_row``, ``peak_mem_mb``.
        (``tracemalloc`` sees NumPy/Python allocations, not all native C memory,
        so ``peak_mem_mb`` is a lower bound for libsvm/BLAS-heavy models.)
    """
    fits, preds, peaks = [], [], []
    for _ in range(repeats):
        m = clone(est)
        tracemalloc.start()
        t0 = time.perf_counter()
        m.fit(X_train, y_train)
        t1 = time.perf_counter()
        m.predict(X_test)
        t2 = time.perf_counter()
        peaks.append(tracemalloc.get_traced_memory()[1] / 1e6)
        tracemalloc.stop()
        fits.append(t1 - t0)
        preds.append(t2 - t1)
    p = float(np.median(preds))
    return {"fit_s": float(np.median(fits)), "predict_s": p,
            "predict_us_per_row": 1e6 * p / max(1, len(X_test)),
            "peak_mem_mb": float(np.max(peaks))}


def predict_flops(fitted_pipeline, n_rows: int, n_features: int) -> Dict[str, float]:
    """Estimate prediction FLOPs and arithmetic intensity (FLOP per byte moved).

    Assumptions: float64 (8 B); bytes = reading the inputs once + the model
    parameters once + writing the outputs. Counts:
        * Linear/Ridge: ``2 p`` per row (multiply-add).
        * Tree: ``depth`` comparisons per row (not true FLOPs; counted as 1 each).
        * SVR (RBF): per row, per support vector ``3 p + 2`` (distance, scale, exp).
        * MLP: ``2 * sum(fan_in * fan_out)`` per row.

    Args:
        fitted_pipeline: Fitted pipeline whose last step is ``model``.
        n_rows: Number of rows predicted.
        n_features: Number of input features reaching the model.

    Returns:
        ``flops``, ``bytes``, ``arith_intensity`` (NaN if unsupported).
    """
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
        per_row, params = 4, 2 * len(m.table_)  # one table lookup per row
    elif name == "DummyRegressor":
        per_row, params = 1, 1
    else:
        return {"flops": np.nan, "bytes": np.nan, "arith_intensity": np.nan}
    flops = float(per_row * n_rows)
    nbytes = 8.0 * (n_rows * p + params + n_rows)
    return {"flops": flops, "bytes": nbytes, "arith_intensity": flops / nbytes}


def profile_call(fn: Callable, *args, top: int = 8, **kwargs) -> str:
    """Run ``fn`` under cProfile and return the top cumulative-time lines.

    Args:
        fn: Callable to profile.
        top: Number of rows to report.

    Returns:
        Text table from :mod:`pstats`.
    """
    pr = cProfile.Profile()
    pr.enable()
    fn(*args, **kwargs)
    pr.disable()
    buf = _io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(top)
    return buf.getvalue()


def performance_table(models: Dict, X_train, y_train, X_test, repeats: int = 3) -> pd.DataFrame:
    """Timing, memory and FLOP estimates for each model.

    Args:
        models: name -> unfitted (possibly tuned) pipeline.
        X_train: Training features.
        y_train: Training target.
        X_test: Test features.
        repeats: Timing repeats.

    Returns:
        One row per model.
    """
    rows = []
    for name, est in models.items():
        t = time_fit_predict(est, X_train, y_train, X_test, repeats)
        fitted = clone(est).fit(X_train, y_train)
        n_feat = fitted.named_steps["model"].n_features_in_ if hasattr(
            fitted.named_steps["model"], "n_features_in_") else X_train.shape[1]
        f = predict_flops(fitted, len(X_test), n_feat)
        rows.append({"model": name, **t, **f})
    return pd.DataFrame(rows)
