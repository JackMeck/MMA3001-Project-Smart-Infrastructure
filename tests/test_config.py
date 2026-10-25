
from smart_infrastructure import Config


def test_defaults_match_final_notebook():
    c = Config()
    assert c.env_sensor_id == "6012002000869"
    assert c.floorspace == "941f0352-bb69-47d0-a9c5-ac29eb4f7319"
    assert (c.window_start, c.window_end) == ("2024-02-29", "2024-06-14")
    assert c.tz == "Australia/Melbourne" and c.grid == "15min"
    assert (c.train_fraction, c.gap_steps, c.cv_splits) == (0.8, 8, 5)
    assert c.lean_cols == ["co2", "temperature", "hour", "weekend"]
    assert (c.svr_C, c.svr_epsilon) == (0.3, 0.01)


def test_step_minutes():
    assert Config().step_minutes == 15.0
    assert Config(grid="1h").step_minutes == 60.0


def test_mutable_defaults_not_shared_between_instances():
    a, b = Config(), Config()
    a.lean_cols.append("voc")
    assert "voc" not in b.lean_cols
