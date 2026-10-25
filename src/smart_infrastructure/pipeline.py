"""End-to-end workflow for the final project.

Steps (same order as the final notebook):

1. load and clean the environmental and occupancy data, build the modelling table;
2. chronological train/test split and first cross-validation comparison;
3. sensitivity studies (feature ablation, SVR settings, training-data requirement);
4. final models, cross-validation, computational cost, ONE final test, plots;
5. appendix: repeat on every floorspace.

Usage from Python::

    from smart_infrastructure import Config, run_pipeline

    result = run_pipeline(Config(), env_csv="data/5EnvSensor_MayToDec2024_180kRows.csv",
                          occ_csv="data/5occupancySensor_MayToDec2024_9MRows.csv",
                          out_dir="results")
    print(result["test"])

Usage from the command line::

    python -m smart_infrastructure --data-dir data --out results

The held-out test period is scored exactly once, after all choices are made.
"""

import argparse
import json
import os
from typing import Dict, Optional

import pandas as pd

from .config import Config
from .data import (build_table, load_environment, load_occupancy, sensor_columns,
                   unzip_data)
from .evaluation import (chronological_split, computational_cost, cross_validate_models,
                         final_test, mae_improvement, naive_benchmarks)
from .floorspace import (floorspace_scan, floorspace_sensitivity,
                         occupancy_all_floorspaces)
from .models import candidate_models, final_models
from .plots import plot_test_results
from .sensitivity import (data_requirement, feature_ablation, leave_one_out, svr_grid,
                          training_size_curve)


def _say(verbose: bool, title: str, obj) -> None:
    """Print a titled table when ``verbose``."""
    if verbose:
        print(f"\n{title}")
        print(obj.round(3).to_string() if hasattr(obj, "round") else obj)


def run_pipeline(cfg: Config, env_csv: str, occ_csv: str, out_dir: Optional[str] = None,
                 appendix: bool = True, timing_repeats: int = 5,
                 verbose: bool = False) -> Dict:
    """Run the whole analysis and return every result table.

    Args:
        cfg: Configuration.
        env_csv: Path of the environmental CSV.
        occ_csv: Path of the occupancy CSV.
        out_dir: If given, tables are saved as CSV, the test plot as PNG and the
            settings/audit as JSON in this folder (created if needed).
        appendix: Also run the all-floorspace appendix study (slow: reads the whole
            occupancy file again).
        timing_repeats: Repeats for the timing study (best is reported).
        verbose: Print each table as it is produced.

    Returns:
        Dict with keys ``audit``, ``table``, ``split``, ``cv_candidates``,
        ``ablation``, ``leave_one_out``, ``svr_grid``, ``svr_best``, ``curve_linear``,
        ``curve_final``, ``cv_final``, ``cost``, ``test``, ``preds``, ``improvement``,
        ``naive`` and, when ``appendix`` is True, ``scan`` and
        ``floorspace_sensitivity``.

    Raises:
        ValueError: If the data are unusable (no CO2, no occupancy, no overlap, or too
            few rows to split).
    """
    # ---- 1. data ------------------------------------------------------------------
    env, audit = load_environment(env_csv, cfg)
    occ, n_events = load_occupancy(occ_csv, cfg)
    audit["occupancy_events"] = n_events
    audit["occupancy_rows"] = int(occ.notna().sum())
    df = build_table(env, occ)
    sensor_cols = sensor_columns(env)
    features = sensor_cols + ["hour", "weekend"]
    X, y = df[features], df["occupancy"]
    audit["table_shape"] = list(df.shape)
    _say(verbose, "DATA AUDIT", json.dumps(audit, indent=2, default=str))
    _say(verbose, "MODELLING TABLE SUMMARY", df.describe())

    # ---- 2. split + first comparison ------------------------------------------------
    X_tr, y_tr, X_te, y_te = chronological_split(X, y, cfg)
    split = {"train": [str(X_tr.index.min()), str(X_tr.index.max())],
             "test": [str(X_te.index.min()), str(X_te.index.max())]}
    cv_candidates = cross_validate_models(candidate_models(sensor_cols), X_tr, y_tr, cfg)
    _say(verbose, "CROSS-VALIDATION (training period, mean of folds)", cv_candidates)

    # ---- 3. sensitivity / optimisation (training period only) -------------------------
    ablation = feature_ablation(X_tr, y_tr, X_te, y_te, features, sensor_cols, cfg)
    loo = leave_one_out(X_tr, y_tr, features, cfg)
    grid, best = svr_grid(X_tr, y_tr, cfg)
    curve_linear = training_size_curve(candidate_models(sensor_cols)["Linear Regression"],
                                       X_tr, y_tr, cfg)
    models = final_models(cfg)
    curve_final = data_requirement(
        {k: v for k, v in models.items() if k != "Baseline (schedule)"}, X_tr, y_tr, cfg)
    for title, obj in [("FEATURE ABLATION (linear model)", ablation),
                       ("LEAVE-ONE-OUT (full linear model, CV)", loo),
                       ("SVR (lean inputs): CV MAE, C (rows) x epsilon (columns)", grid),
                       ("LINEAR MODEL: validation error vs training data", curve_linear),
                       ("DATA REQUIREMENT: FINAL MODELS", curve_final)]:
        _say(verbose, title, obj)

    # ---- 4. final evaluation ----------------------------------------------------------
    cv_final = cross_validate_models(models, X_tr, y_tr, cfg)
    cost = computational_cost(models, X_tr, y_tr, X_te, timing_repeats)
    test, preds = final_test(models, X_tr, y_tr, X_te, y_te)
    improvement = mae_improvement(test)
    naive = naive_benchmarks(X_tr, y_tr, X_te, y_te)
    for title, obj in [("CROSS-VALIDATION OF FINAL MODELS", cv_final),
                       ("COMPUTATIONAL COST", cost), ("FINAL TEST", test),
                       ("MAE IMPROVEMENT OVER BASELINE", improvement),
                       ("NAIVE CONSTANT BENCHMARKS (test set)", naive)]:
        _say(verbose, title, obj)

    result = {"audit": audit, "table": df, "split": split, "cv_candidates": cv_candidates,
              "ablation": ablation, "leave_one_out": loo, "svr_grid": grid,
              "svr_best": best, "curve_linear": curve_linear, "curve_final": curve_final,
              "cv_final": cv_final, "cost": cost, "test": test, "preds": preds,
              "improvement": improvement, "naive": naive}

    # ---- 5. appendix ------------------------------------------------------------------
    if appendix:
        occ_all = occupancy_all_floorspaces(occ_csv, cfg)
        train_end, test_start = X_tr.index.max(), X_te.index.min()   # identical split
        result["scan"] = floorspace_scan(env, occ_all, cfg, train_end)
        result["floorspace_sensitivity"] = floorspace_sensitivity(
            env, occ_all, models, features, cfg, train_end, test_start)
        _say(verbose, "FLOORSPACE SCAN (training period only)", result["scan"])

    if out_dir:
        save_results(result, cfg, out_dir, y_te)
    return result


def save_results(result: Dict, cfg: Config, out_dir: str, y_te: pd.Series) -> None:
    """Write result tables (CSV), the test plot (PNG) and settings/audit (JSON).

    Args:
        result: Output of :func:`run_pipeline`.
        cfg: Configuration used for the run.
        out_dir: Output folder (created if needed).
        y_te: Observed test headcount (for the plot).
    """
    from dataclasses import asdict

    os.makedirs(out_dir, exist_ok=True)
    for name, obj in result.items():
        if isinstance(obj, (pd.DataFrame, pd.Series)) and name != "table":
            obj.to_csv(os.path.join(out_dir, f"{name}.csv"))
    with open(os.path.join(out_dir, "run_info.json"), "w") as fh:
        json.dump({"config": asdict(cfg), "audit": result["audit"],
                   "split": result["split"], "svr_best": list(result["svr_best"])},
                  fh, indent=2, default=str)
    plot_test_results(result["test"], result["preds"], y_te,
                      path=os.path.join(out_dir, "test_results.png"))


def main(argv=None) -> None:
    """Command-line entry point (``python -m smart_infrastructure``)."""
    p = argparse.ArgumentParser(description="Room-occupancy nowcasting workflow.")
    p.add_argument("--data-dir", required=True,
                   help="Folder holding the two CSV files (or the .zip archives).")
    p.add_argument("--zip-dir", help="Folder with *Sensor*.zip files to extract first.")
    p.add_argument("--out", default="results", help="Output folder.")
    p.add_argument("--no-appendix", action="store_true", help="Skip the all-floorspace study.")
    a = p.parse_args(argv)
    cfg = Config()
    if a.zip_dir:
        unzip_data(a.zip_dir, a.data_dir)
    run_pipeline(cfg, os.path.join(a.data_dir, cfg.env_csv_name),
                 os.path.join(a.data_dir, cfg.occ_csv_name), out_dir=a.out,
                 appendix=not a.no_appendix, verbose=True)


if __name__ == "__main__":
    main()
