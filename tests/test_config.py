import pytest

from smart_infrastructure import Config, monash_config


def test_json_round_trip(tmp_path, cfg):
    path = tmp_path / "cfg.json"
    cfg.to_json(str(path))
    loaded = Config.from_json(str(path))
    assert loaded == cfg


def test_unknown_json_key_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"not_a_real_setting": 1}')
    with pytest.raises(TypeError):
        Config.from_json(str(path))


def test_step_minutes_from_resample_rule():
    assert monash_config(resample_rule="15min").step_minutes == 15.0
    assert monash_config(resample_rule="1h").step_minutes == 60.0


def test_monash_overrides_applied():
    c = monash_config(env_sensor_id="abc", floorspace_id="xyz")
    assert (c.env_sensor_id, c.floorspace_id) == ("abc", "xyz")
