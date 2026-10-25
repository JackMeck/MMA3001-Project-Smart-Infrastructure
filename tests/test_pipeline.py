import json
import pandas as pd
import pytest

from smart_infrastructure import monash_config, run_from_clean, run_monash


def test_run_from_clean_end_to_end(tmp_path, cfg, clean_df):
    out = tmp_path / "run"
    res = run_from_clean(cfg, str(out), clean_df, {"note": "synthetic"}, quick=True)
    assert res["selected_model"] in {"linear", "tree", "svr", "nn"}
    assert res["final_test"]["mae"] < 5
    assert res["n_train"] > 0 and res["n_test"] > 0
    # train period must finish before the held-out test period starts
    assert pd.Timestamp(res["train_period"][1]) < pd.Timestamp(res["test_period"][0])
    for name in ("config_used.json", "audit.json", "run_summary.json", "performance.csv"):
        assert (out / name).exists(), name
    assert json.loads((out / "run_summary.json").read_text())["selected_model"] \
        == res["selected_model"]


def test_run_from_clean_beats_mean_baseline(tmp_path, cfg, clean_df):
    res = run_from_clean(cfg, str(tmp_path), clean_df, {}, quick=True)
    assert res["mae_improvement_vs_baseline_mean"] > 0.5


def test_run_monash_requires_floorspace(tmp_path):
    cfg = monash_config(env_sensor_id="s")      # floorspace_id deliberately unset
    with pytest.raises(ValueError, match="floorspace_id"):
        run_monash(cfg, str(tmp_path))


def test_run_monash_no_events_raises(tmp_path):
    cfg = monash_config(env_sensor_id="s", floorspace_id="nope")
    occ = pd.DataFrame({"floorspaceid": ["other"], "headcount": [1],
                        "collecteddate": ["2024-03-04 09:00:00+11"]})
    env = pd.DataFrame({"sensorid": ["s"], "jsondata": ["{}"]})
    with pytest.raises(ValueError, match="No occupancy events"):
        run_monash(cfg, str(tmp_path), df_env=env, occ=occ)
