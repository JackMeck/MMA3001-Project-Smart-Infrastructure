import json
import os

import pandas as pd
import pytest

from smart_infrastructure import Config, run_pipeline
from smart_infrastructure.pipeline import main
from synth import make_raw_files


@pytest.fixture(scope="module")
def light_cfg():
    """Smaller sweeps so the end-to-end run is quick; algorithms are unchanged."""
    return Config(svr_C_grid=[0.3, 1.0], svr_epsilon_grid=[0.01], curve_fractions=[0.5, 1.0])


@pytest.fixture(scope="module")
def run(tmp_path_factory, raw_files, light_cfg):
    out = tmp_path_factory.mktemp("out")
    return run_pipeline(light_cfg, *raw_files, out_dir=str(out), timing_repeats=1), out


def test_all_result_tables_present(run):
    res, _ = run
    for k in ["audit", "table", "split", "cv_candidates", "ablation", "leave_one_out",
              "svr_grid", "svr_best", "curve_linear", "curve_final", "cv_final", "cost",
              "test", "preds", "improvement", "naive", "scan", "floorspace_sensitivity"]:
        assert k in res, k


def test_test_period_strictly_after_training_period(run):
    res, _ = run
    assert pd.Timestamp(res["split"]["train"][1]) < pd.Timestamp(res["split"]["test"][0])


def test_final_models_beat_naive_constant_by_a_wide_margin(run):
    res, _ = run
    best_naive = res["naive"]["MAE"].min()
    assert res["test"].loc["Linear Regression", "MAE"] < 0.5 * best_naive
    assert res["test"].loc["SVR", "MAE"] < 0.5 * best_naive


def test_baseline_improvement_is_zero_for_baseline(run):
    res, _ = run
    assert res["improvement"]["Baseline (schedule)"] == 0.0


def test_predictions_aligned_with_test_rows(run):
    res, _ = run
    n_test = len(res["preds"]["SVR"])
    assert n_test == len(res["table"]) - int(0.8 * len(res["table"]))
    assert all((p >= 0).all() for p in res["preds"].values())


def test_outputs_written(run):
    res, out = run
    names = set(os.listdir(out))
    assert {"test.csv", "cv_final.csv", "scan.csv", "run_info.json", "test_results.png"} <= names
    info = json.loads((out / "run_info.json").read_text())
    assert info["config"]["floorspace"] == Config().floorspace
    assert os.path.getsize(out / "test_results.png") > 1000


def test_deterministic_given_same_input(raw_files, light_cfg):
    a = run_pipeline(light_cfg, *raw_files, appendix=False, timing_repeats=1)
    b = run_pipeline(light_cfg, *raw_files, appendix=False, timing_repeats=1)
    pd.testing.assert_frame_equal(a["test"], b["test"])
    pd.testing.assert_frame_equal(a["cv_final"], b["cv_final"])


def test_wrong_sensor_raises_clear_error(raw_files, light_cfg):
    from dataclasses import replace
    with pytest.raises(ValueError):
        run_pipeline(replace(light_cfg, env_sensor_id="nope"), *raw_files, appendix=False)


def test_unknown_floorspace_raises(raw_files, light_cfg):
    from dataclasses import replace
    with pytest.raises(ValueError):
        run_pipeline(replace(light_cfg, floorspace="no-such-room"), *raw_files, appendix=False)


def test_command_line_entry_point(tmp_path, monkeypatch, light_cfg, capsys):
    c = Config()
    make_raw_files(str(tmp_path), days=40, env_name=c.env_csv_name, occ_name=c.occ_csv_name)
    monkeypatch.setattr("smart_infrastructure.pipeline.Config", lambda: light_cfg)
    main(["--data-dir", str(tmp_path), "--out", str(tmp_path / "o"), "--no-appendix"])
    assert (tmp_path / "o" / "test.csv").exists()
    assert "FINAL TEST" in capsys.readouterr().out
