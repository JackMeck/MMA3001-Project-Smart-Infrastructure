"""End-to-end workflow reproducing every table and figure in the report.

Run from the command line::

    python scripts/run_pipeline.py --data-dir data --output-dir results

The steps mirror the report sections: data preparation, candidate comparison,
ablation, SVR tuning, data requirement, final cross-validation, timing, the single
final test, constant benchmarks and the five-floorspace sensitivity study.
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")  # headless: figures are saved, not shown

from . import config
from .data import (add_co2_change, build_modelling_table, extract_zips, load_all_occupancy,
                   load_environment, load_occupancy)
from .evaluation import chronological_split, cv_table
from .experiments import (data_requirement, feature_ablation, final_test, floorspace_scan,
                          floorspace_sensitivity, floorspace_tables, leave_one_out,
                          naive_benchmarks, svr_grid, time_models)
from .models import candidate_models, final_models
from .plotting import plot_test_results


def _show(title: str, table, out_dir: str, name: str, digits: int = 3) -> None:
    """Print a titled table and save it as ``<out_dir>/<name>.csv``."""
    print(f"\n{title}")
    print(table.round(digits).to_string())
    table.to_csv(os.path.join(out_dir, f"{name}.csv"))


def run(data_dir: str = "data", output_dir: str = "results", zip_dir: str | None = None,
        appendix: bool = True) -> dict:
    """Run the full workflow.

    Args:
        data_dir: Folder holding (or receiving) the two CSV files
            ``5EnvSensor_MayToDec2024_180kRows.csv`` and
            ``5occupancySensor_MayToDec2024_9MRows.csv``.
        output_dir: Folder for CSV tables and the test figure.
        zip_dir: Optional folder with the original ``*Sensor*.zip`` files to unpack
            into ``data_dir`` first.
        appendix: Also run the five-floorspace study (reads the 9M-row file again).

    Returns:
        Dict of the main result tables (``cv_final``, ``test``, ``perf`` ...).

    Raises:
        FileNotFoundError: If a data CSV is missing.
    """
    os.makedirs(output_dir, exist_ok=True)
    if zip_dir:
        print("data files:", extract_zips(zip_dir, data_dir))
    env_csv = os.path.join(data_dir, "5EnvSensor_MayToDec2024_180kRows.csv")
    occ_csv = os.path.join(data_dir, "5occupancySensor_MayToDec2024_9MRows.csv")
    for p in (env_csv, occ_csv):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found - see data/README.md")

    # 1. data
    env = add_co2_change(load_environment(env_csv))
    occ = load_occupancy(occ_csv)
    df = build_modelling_table(env, occ)
    print(df.shape)
    _show("DESCRIPTIVE STATISTICS", df.describe(), output_dir, "describe", 2)

    sp = chronological_split(df)
    X_tr, y_tr, X_te, y_te = sp
    print("train", X_tr.index.min(), "->", X_tr.index.max(), "| test", X_te.index.min(), "->", X_te.index.max())

    # 2. candidates
    cand = candidate_models()
    _show("CROSS-VALIDATION OF FIRST-ROUND CANDIDATES", cv_table(cand, X_tr, y_tr), output_dir, "cv_candidates")

    # 3. sensitivity / optimisation (cross-validation only)
    _show("FEATURE ABLATION (linear model)", feature_ablation(X_tr, y_tr, X_te, y_te), output_dir, "ablation")
    _show("LEAVE-ONE-OUT (full linear model)", leave_one_out(X_tr, y_tr), output_dir, "leave_one_out")
    grid, (c_best, e_best) = svr_grid(X_tr, y_tr)
    _show("SVR (lean inputs): CV MAE vs C (rows) and epsilon (columns)", grid, output_dir, "svr_grid")
    print(f"best SVR: C = {c_best}, epsilon = {e_best}  (final models use C={config.SVR_C}, "
          f"epsilon={config.SVR_EPSILON}; update config.py if the grid or data change)")

    final = final_models()
    curve_models = {"Linear (all sensors, no time)": candidate_models()["Linear Regression"], **final}
    curve_models.pop("Baseline (schedule)")
    _show("DATA REQUIREMENT", data_requirement(curve_models, X_tr, y_tr), output_dir, "data_requirement")

    # 4-5. final evaluation
    cv_final = cv_table(final, X_tr, y_tr)
    _show("CROSS-VALIDATION OF FINAL MODELS", cv_final, output_dir, "cv_final")
    perf = time_models(final, X_tr, y_tr, X_te)
    _show(f"COMPUTATIONAL COST (train rows = {len(X_tr)}, test rows = {len(X_te)}; best of 5)", perf,
          output_dir, "performance")
    test, preds = final_test(final, X_tr, y_tr, X_te, y_te)
    _show("FINAL TEST (held-out data, run once)", test, output_dir, "final_test")
    print("\nMAE improvement over baseline:")
    print((1 - test["MAE"] / test.loc["Baseline (schedule)", "MAE"]).round(3).to_string())
    plot_test_results(test, y_te, preds, os.path.join(output_dir, "test_results.png"))
    _show("NAIVE CONSTANT BENCHMARKS (test set)", naive_benchmarks(X_tr, y_tr, X_te, y_te), output_dir, "naive")
    results = {"cv_final": cv_final, "test": test, "perf": perf}

    # appendix: unknown sensor-to-room mapping
    if appendix:
        tables = floorspace_tables(env, load_all_occupancy(occ_csv))
        scan = floorspace_scan(tables)
        _show("FLOORSPACE SCAN (training period only)", scan, output_dir, "floorspace_scan")
        print("\nselected by this rule:", scan.index[0], "| used in main workflow:", config.FLOORSPACE[:8])
        sens = floorspace_sensitivity(tables, final)
        sens.to_csv(os.path.join(output_dir, "floorspace_sensitivity.csv"), index=False)
        for metric in ("MAE", "R2"):
            print(f"\nSENSITIVITY ({metric}, test set)")
            print(sens.pivot(index="floorspace", columns="model", values=metric).round(3).to_string())
        results["sensitivity"] = sens
    return results


def main(argv=None) -> None:
    """Command-line entry point (see module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", default="data", help="folder with the two CSV files")
    ap.add_argument("--zip-dir", default=None, help="folder with the original zip files to unpack")
    ap.add_argument("--output-dir", default="results", help="folder for tables and figures")
    ap.add_argument("--skip-appendix", action="store_true", help="skip the five-floorspace study")
    a = ap.parse_args(argv)
    run(a.data_dir, a.output_dir, a.zip_dir, appendix=not a.skip_appendix)
