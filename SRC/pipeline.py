"""End-to-end workflow: audit -> clean -> features -> select -> final test -> studies.

Usage example:

```python
from smart_infrastructure import monash_config
from smart_infrastructure.pipeline import run_monash

cfg = monash_config(env_sensor_id="6012002000869",
                    floorspace_id="941f0352-bb69-47d0-a9c5-ac29eb4f7319")
result = run_monash(cfg, out_dir="results/run1")
```

The held-out test period is scored exactly once, after model selection.
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV

from .config import Config
from .data import (audit_occupancy, build_dataset, env_to_grid, occupancy_to_grid,
                   prepare_env, read_occupancy, to_events)
from .evaluation import (chronological_split, cross_validate, feature_group_ablation,
                         hyperparameter_sweep, leave_one_room_out, noise_robustness,
                         performance_table, plot_diagnostics, regression_metrics,
                         residual_tables, resolution_sweep, time_block_folds,
                         training_size_curve)
from .features import build_features
from .models import make_models, param_grids


def run_from_clean(cfg: Config, out_dir: str, clean_df: pd.DataFrame, report: Dict,
                   clean_builder=None, quick: bool = False) -> Dict:
    """Features -> model selection -> final test -> studies, from a clean grid frame.

    The held-out test period is scored exactly once, after model selection.

    Args:
        cfg: Workflow configuration.
        out_dir: Output directory (created if needed).
        clean_df: Regular-grid frame ``timestamp, room, <target>, <env cols>``.
        report: Audit dictionary written to ``audit.json``.
        clean_builder: Optional ``cfg -> clean frame`` used by the resolution study.
        quick: Skip the slow resolution sweep and performance repeats.

    Returns:
        Summary dictionary (selected model, final test metrics, paths).
    """
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    cfg.to_json(os.path.join(out_dir, "config_used.json"))
    with open(os.path.join(out_dir, "audit.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    # 1. features + splits
    fs = build_features(clean_df, cfg)
    tr, te = chronological_split(fs.meta, cfg.test_fraction, cfg.cv_gap_min)
    Xtr = fs.X.iloc[tr].reset_index(drop=True)
    ytr = fs.y.iloc[tr].reset_index(drop=True)
    folds = time_block_folds(fs.meta["timestamp"].iloc[tr], cfg.cv_splits, cfg.cv_gap_min)

    # 3. model selection on training folds only
    models = make_models(cfg.seed, baseline_col=cfg.env_cols[0],
                         include_schedule="time" in fs.groups)
    if cfg.env_cols[0] not in fs.X.columns:       # CO2-only baseline needs the raw group
        models.pop("baseline_co2_linear")
    grids = param_grids()
    scoring = "neg_root_mean_squared_error" if cfg.selection_metric == "rmse" \
        else "neg_mean_absolute_error"
    tuned = {}
    for name, est in models.items():
        if name in grids:
            gs = GridSearchCV(est, grids[name], cv=folds, scoring=scoring, n_jobs=-1)
            gs.fit(Xtr, ytr)
            tuned[name] = clone(est).set_params(**gs.best_params_)
        else:
            tuned[name] = est
    cv = cross_validate(tuned, Xtr, ytr, folds, cfg)
    cv.to_csv(os.path.join(out_dir, "cv_folds.csv"), index=False)
    summary = cv.groupby("model").agg(["mean", "std"]).drop(columns="fold", level=0)
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.to_csv(os.path.join(out_dir, "cv_summary.csv"))
    cands = summary.loc[[m for m in summary.index if not m.startswith("baseline_")]]
    best = cands[f"{cfg.selection_metric}_mean"].idxmin()

    # 4. final test: fit on all training data, score ONCE
    final = clone(tuned[best]).fit(Xtr, ytr)
    Xte, yte = fs.X.iloc[te], fs.y.iloc[te]
    rows = []
    for name in tuned:
        m = final if name == best else clone(tuned[name]).fit(Xtr, ytr)
        rows.append({"model": name, "selected": name == best,
                     **regression_metrics(yte, m.predict(Xte), cfg)})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "final_test_metrics.csv"), index=False)
    pred = final.predict(Xte)
    pd.DataFrame({"timestamp": fs.meta["timestamp"].iloc[te].to_numpy(),
                  "room": fs.meta["room"].iloc[te].to_numpy(),
                  "observed": yte.to_numpy(),
                  "predicted": np.clip(pred, 0, None)}).to_csv(
        os.path.join(out_dir, "test_predictions.csv"), index=False)
    for k, t in residual_tables(fs, te, pred, cfg).items():
        t.to_csv(os.path.join(out_dir, f"residuals_{k}.csv"), index=False)
    plot_diagnostics(yte, pred, fs.meta["timestamp"].iloc[te],
                     os.path.join(out_dir, "figures", "final_test_diagnostics.png"),
                     title=f"Final test - {best}")

    # 5. grouped generalisation (only meaningful with >= 2 rooms)
    if fs.meta["room"].nunique() >= 2:
        g_rows = []
        for tr_i, va_i in leave_one_room_out(fs.meta):
            m = clone(tuned[best]).fit(fs.X.iloc[tr_i], fs.y.iloc[tr_i])
            g_rows.append({"held_out_room": fs.meta["room"].iloc[va_i[0]],
                           **regression_metrics(fs.y.iloc[va_i], m.predict(fs.X.iloc[va_i]), cfg)})
        pd.DataFrame(g_rows).to_csv(os.path.join(out_dir, "leave_one_room_out.csv"), index=False)

    # 6. sensitivity / optimisation (training folds; noise study post-selection)
    feature_group_ablation(tuned[best], fs, tr, folds, cfg).to_csv(
        os.path.join(out_dir, "sens_feature_groups.csv"), index=False)
    training_size_curve(tuned[best], Xtr, ytr, folds, cfg).to_csv(
        os.path.join(out_dir, "sens_training_size.csv"), index=False)
    if "tree" in tuned:
        hyperparameter_sweep(tuned["tree"], "model__max_depth", [2, 3, 5, 8, 12, 16],
                             Xtr, ytr, folds, cfg).to_csv(
            os.path.join(out_dir, "sens_tree_depth.csv"), index=False)
    test_start = fs.meta["timestamp"].iloc[te].min()
    noise_robustness(final, clean_df, cfg, test_start).to_csv(
        os.path.join(out_dir, "sens_noise.csv"), index=False)
    if not quick and clean_builder is not None:
        resolution_sweep(clean_builder, tuned[best], cfg,
                         rules=tuple(cfg.resolution_rules)).to_csv(
            os.path.join(out_dir, "sens_resolution.csv"), index=False)

    # 7. performance
    performance_table(tuned, Xtr, ytr, Xte, repeats=1 if quick else 3).to_csv(
        os.path.join(out_dir, "performance.csv"), index=False)

    sel = next(r for r in rows if r["selected"])
    result = {"selected_model": best, "n_train": int(len(tr)), "n_test": int(len(te)),
              "final_test": sel, "out_dir": out_dir,
              "train_period": [str(fs.meta["timestamp"].iloc[tr].min()),
                               str(fs.meta["timestamp"].iloc[tr].max())],
              "test_period": [str(fs.meta["timestamp"].iloc[te].min()),
                              str(fs.meta["timestamp"].iloc[te].max())]}
    for ref in ("baseline_mean", "baseline_schedule"):
        r0 = next((r for r in rows if r["model"] == ref), None)
        if r0 is not None and r0["mae"] > 0:
            result[f"mae_improvement_vs_{ref}"] = 1.0 - sel["mae"] / r0["mae"]
    with open(os.path.join(out_dir, "run_summary.json"), "w") as fh:
        json.dump(result, fh, indent=2, default=float)
    return result


def pairing_scan(cfg: Config, df_env: Optional[pd.DataFrame] = None,
                 occ: Optional[pd.DataFrame] = None,
                 floorspace_ids: Optional[List[str]] = None) -> pd.DataFrame:
    """Compare one sensor against several candidate floorspaces (training data only).

    For each floorspace an environment-only decision tree (no time-of-day features)
    is cross-validated on the training period. The final test period is never used.
    A floorspace that really shares the sensor's room should be predicted much better
    than the others. Every floorspace follows a similar weekday timetable, so raw
    correlations cannot separate them; this scan can, but it is evidence, not proof.
    Confirm with the official sensor-to-room mapping.

    Args:
        cfg: Configuration (``env_sensor_id`` and the candidate ``floorspace_ids``).
        df_env: Raw environmental frame (read from ``cfg.env_path`` if None).
        occ: Raw occupancy frame (read from ``cfg.occupancy_path`` if None).
        floorspace_ids: Candidates (defaults to ``cfg.floorspace_ids``).

    Returns:
        One row per floorspace, best first: rows, Spearman(co2, occupancy), CV R2 and
        MAE of the environment-only tree, and CV MAE of the schedule-only baseline.
    """
    ids = [str(f) for f in (floorspace_ids or cfg.floorspace_ids)]
    if not ids:
        raise ValueError("Give floorspace_ids to scan.")
    if df_env is None:
        df_env = pd.read_csv(cfg.env_path, usecols=["sensorid", "jsondata"])
    ev_all = (to_events(occ, cfg, ids) if occ is not None
              else read_occupancy(cfg.occupancy_path, cfg, ids))
    env_grid, _, _ = prepare_env(df_env, cfg)
    models = make_models(cfg.seed, cfg.env_cols[0], include_schedule=cfg.use_time_features)
    rows = []
    for fid in ids:
        ev = ev_all[ev_all["floorspaceid"] == fid]
        row = {"floorspace": fid[:8], "n_rows": 0}
        try:
            d = build_dataset(env_grid, occupancy_to_grid(ev, cfg), cfg, fid[:8])
        except ValueError:
            rows.append(row)
            continue
        fs = build_features(d, cfg)
        tr, _ = chronological_split(fs.meta, cfg.test_fraction, cfg.cv_gap_min)
        folds = time_block_folds(fs.meta["timestamp"].iloc[tr], cfg.cv_splits, cfg.cv_gap_min)
        y = fs.y.iloc[tr].reset_index(drop=True)
        X_all = fs.X.iloc[tr].reset_index(drop=True)
        env_only = [c for g, v in fs.groups.items() if g != "time" for c in v]
        res = cross_validate({"tree": models["tree"]}, X_all[env_only], y, folds, cfg)
        row.update({"n_rows": int(len(fs.y)),
                    "spearman_co2": float(X_all[cfg.env_cols[0]].corr(y, method="spearman")),
                    "cv_r2_env_only": float(res["r2"].mean()),
                    "cv_mae_env_only": float(res["mae"].mean())})
        if "baseline_schedule" in models:
            b = cross_validate({"s": models["baseline_schedule"]}, X_all, y, folds, cfg)
            row["cv_mae_schedule_only"] = float(b["mae"].mean())
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values("cv_r2_env_only", ascending=False, na_position="last"
                           ).reset_index(drop=True)


def run_monash(cfg: Config, out_dir: str, df_env: Optional[pd.DataFrame] = None,
               occ: Optional[pd.DataFrame] = None, quick: bool = False) -> Dict:
    """Full workflow for one sensor / floorspace pair on the Monash extract.

    Args:
        cfg: Configuration with ``env_sensor_id`` and ``floorspace_id`` set.
        out_dir: Output directory.
        df_env: Raw environmental frame (read from ``cfg.env_path`` if None).
        occ: Raw occupancy frame (read, chunked, from ``cfg.occupancy_path`` if None).
        quick: Skip the slow resolution sweep.

    Returns:
        Summary dictionary (see :func:`run_from_clean`).
    """

    if not cfg.floorspace_id:
        raise ValueError("Set cfg.floorspace_id.")
    if df_env is None:
        df_env = pd.read_csv(cfg.env_path, usecols=["sensorid", "jsondata"])
    events = (to_events(occ, cfg, [cfg.floorspace_id]) if occ is not None
              else read_occupancy(cfg.occupancy_path, cfg, [cfg.floorspace_id]))
    if events.empty:
        raise ValueError(f"No occupancy events for floorspace {cfg.floorspace_id}.")
    env_grid, env_audit, recs = prepare_env(df_env, cfg)
    occ_grid = occupancy_to_grid(events, cfg)
    room = cfg.floorspace_id[:8]
    clean_df = build_dataset(env_grid, occ_grid, cfg, room)
    report = {
        "pairing": {"env_sensor_id": cfg.env_sensor_id, "floorspace_id": cfg.floorspace_id,
                    "note": "pairing is an assumption unless confirmed by the official mapping"},
        "environment": env_audit,
        "occupancy": audit_occupancy(events, occ_grid, cfg),
        "dataset": {"rows": int(len(clean_df)), "start": str(clean_df["timestamp"].min()),
                    "end": str(clean_df["timestamp"].max()), "grid": cfg.resample_rule},
    }

    def builder(c: Config) -> pd.DataFrame:
        return build_dataset(env_to_grid(recs, c), occupancy_to_grid(events, c), c, room)

    return run_from_clean(cfg, out_dir, clean_df, report, clean_builder=builder, quick=quick)
