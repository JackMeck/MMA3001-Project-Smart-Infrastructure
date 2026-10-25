from __future__ import annotations

import os
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV

from .config import Config
from .data import FeatureSet, build_dataset, build_features, occupancy_to_grid, prepare_env, read_occupancy, to_events, env_to_grid, pairing_scan
from .evaluation import (
    cross_validate,
    feature_group_ablation,
    hyperparameter_sweep,
    noise_robustness,
    performance_table,
    plot_diagnostics,
    regression_metrics,
    residual_tables,
    resolution_sweep,
    time_block_folds,
    training_size_curve,
    chronological_split,
    leave_one_room_out,
)
from .models import make_models, param_grids


def run_from_clean(cfg: Config, out_dir: str, clean_df: pd.DataFrame, report: Dict, clean_builder=None, quick: bool = False) -> Dict:
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    cfg.to_json(os.path.join(out_dir, "config_used.json"))
    with open(os.path.join(out_dir, "audit.json"), "w") as fh:
        import json as _json

        _json.dump(report, fh, indent=2, default=str)

    fs = build_features(clean_df, cfg)
    tr, te = chronological_split(fs.meta, cfg.test_fraction, cfg.cv_gap_min)
    Xtr = fs.X.iloc[tr].reset_index(drop=True)
    ytr = fs.y.iloc[tr].reset_index(drop=True)
    folds = time_block_folds(fs.meta["timestamp"].iloc[tr], cfg.cv_splits, cfg.cv_gap_min)

    models = make_models(cfg.seed, baseline_col=cfg.env_cols[0], include_schedule="time" in fs.groups)
    if cfg.env_cols[0] not in fs.X.columns:
        models.pop("baseline_co2_linear")
    grids = param_grids()
    scoring = "neg_root_mean_squared_error" if cfg.selection_metric == "rmse" else "neg_mean_absolute_error"
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

    final = clone(tuned[best]).fit(Xtr, ytr)
    Xte, yte = fs.X.iloc[te], fs.y.iloc[te]
    rows = []
    for name in tuned:
        m = final if name == best else clone(tuned[name]).fit(Xtr, ytr)
        rows.append({"model": name, "selected": name == best, **regression_metrics(yte, m.predict(Xte), cfg)})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "final_test_metrics.csv"), index=False)
    pred = final.predict(Xte)
    pd.DataFrame({
        "timestamp": fs.meta["timestamp"].iloc[te].to_numpy(),
        "room": fs.meta["room"].iloc[te].to_numpy(),
        "observed": yte.to_numpy(),
        "predicted": np.clip(pred, 0, None),
    }).to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)
    for k, t in residual_tables(fs, te, pred, cfg).items():
        t.to_csv(os.path.join(out_dir, f"residuals_{k}.csv"), index=False)
    plot_diagnostics(yte, pred, fs.meta["timestamp"].iloc[te], os.path.join(out_dir, "figures", "final_test_diagnostics.png"), title=f"Final test - {best}")

    if fs.meta["room"].nunique() >= 2:
        g_rows = []
        for tr_i, va_i in leave_one_room_out(fs.meta):
            m = clone(tuned[best]).fit(fs.X.iloc[tr_i], fs.y.iloc[tr_i])
            g_rows.append({"held_out_room": fs.meta["room"].iloc[va_i[0]], **regression_metrics(fs.y.iloc[va_i], m.predict(fs.X.iloc[va_i]), cfg)})
        pd.DataFrame(g_rows).to_csv(os.path.join(out_dir, "leave_one_room_out.csv"), index=False)

    feature_group_ablation(tuned[best], fs, tr, folds, cfg).to_csv(os.path.join(out_dir, "sens_feature_groups.csv"), index=False)
    training_size_curve(tuned[best], Xtr, ytr, folds, cfg).to_csv(os.path.join(out_dir, "sens_training_size.csv"), index=False)
    if "tree" in tuned:
        hyperparameter_sweep(tuned["tree"], "model__max_depth", [2, 3, 5, 8, 12, 16], Xtr, ytr, folds, cfg).to_csv(os.path.join(out_dir, "sens_tree_depth.csv"), index=False)
    test_start = fs.meta["timestamp"].iloc[te].min()
    noise_robustness(final, clean_df, cfg, test_start).to_csv(os.path.join(out_dir, "sens_noise.csv"), index=False)
    if not quick and clean_builder is not None:
        resolution_sweep(clean_builder, tuned[best], cfg, rules=tuple(cfg.resolution_rules)).to_csv(os.path.join(out_dir, "sens_resolution.csv"), index=False)

    performance_table(tuned, Xtr, ytr, Xte, repeats=1 if quick else 3).to_csv(os.path.join(out_dir, "performance.csv"), index=False)

    sel = next(r for r in rows if r["selected"])
    result = {
        "selected_model": best,
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "final_test": sel,
        "out_dir": out_dir,
        "train_period": [str(fs.meta["timestamp"].iloc[tr].min()), str(fs.meta["timestamp"].iloc[tr].max())],
        "test_period": [str(fs.meta["timestamp"].iloc[te].min()), str(fs.meta["timestamp"].iloc[te].max())],
    }
    for ref in ("baseline_mean", "baseline_schedule"):
        r0 = next((r for r in rows if r["model"] == ref), None)
        if r0 is not None and r0["mae"] > 0:
            result[f"mae_improvement_vs_{ref}"] = 1.0 - sel["mae"] / r0["mae"]
    with open(os.path.join(out_dir, "run_summary.json"), "w") as fh:
        import json as _json

        _json.dump(result, fh, indent=2, default=float)
    return result


def run_monash(cfg: Config, out_dir: str, df_env: pd.DataFrame | None = None, occ: pd.DataFrame | None = None, quick: bool = False) -> Dict:
    if not cfg.floorspace_id:
        raise ValueError("Set cfg.floorspace_id.")
    if df_env is None:
        df_env = pd.read_csv(cfg.env_path, usecols=["sensorid", "jsondata"])
    events = to_events(occ, cfg, [cfg.floorspace_id]) if occ is not None else read_occupancy(cfg.occupancy_path, cfg, [cfg.floorspace_id])
    if events.empty:
        raise ValueError(f"No occupancy events for floorspace {cfg.floorspace_id}.")
    env_grid, env_audit, recs = prepare_env(df_env, cfg)
    occ_grid = occupancy_to_grid(events, cfg)
    room = cfg.floorspace_id[:8]
    clean_df = build_dataset(env_grid, occ_grid, cfg, room)
    report = {
        "pairing": {"env_sensor_id": cfg.env_sensor_id, "floorspace_id": cfg.floorspace_id, "note": "pairing is an assumption unless confirmed by the official mapping"},
        "environment": env_audit,
        "occupancy": audit_occupancy(events, occ_grid, cfg),
        "dataset": {"rows": int(len(clean_df)), "start": str(clean_df["timestamp"].min()), "end": str(clean_df["timestamp"].max()), "grid": cfg.resample_rule},
    }

    def builder(c: Config) -> pd.DataFrame:
        return build_dataset(env_to_grid(recs, c), occupancy_to_grid(events, c), c, room)

    return run_from_clean(cfg, out_dir, clean_df, report, clean_builder=builder, quick=quick)


__all__ = ["run_from_clean", "run_monash"]
