import numpy as np
import pytest
from dataclasses import replace

from smart_infrastructure import build_features


def test_feature_groups_and_alignment(cfg, clean_df):
    fs = build_features(clean_df, cfg)
    assert len(fs.X) == len(fs.y) == len(fs.meta) == len(clean_df)
    assert set(fs.groups) <= {"raw", "rolling", "trend", "time"}
    assert all(c in fs.X.columns for cols in fs.groups.values() for c in cols)


def test_features_are_causal(cfg, clean_df):
    """Changing a FUTURE row must not change any earlier feature value."""
    base = build_features(clean_df, cfg).X
    altered = clean_df.copy()
    altered.loc[altered.index[-1], "co2"] += 1000.0
    after = build_features(altered, cfg).X
    np.testing.assert_allclose(base.iloc[:-1].to_numpy(), after.iloc[:-1].to_numpy(),
                               equal_nan=True)


def test_target_never_in_features(cfg, clean_df):
    fs = build_features(clean_df, cfg)
    assert cfg.target_col not in fs.X.columns


def test_leakage_columns_rejected(cfg, clean_df):
    bad = replace(cfg, exclude_cols=["co2"])
    with pytest.raises(ValueError, match="leakage"):
        build_features(clean_df, bad)


def test_unknown_feature_group_rejected(cfg, clean_df):
    with pytest.raises(ValueError, match="Unknown feature groups"):
        build_features(clean_df, replace(cfg, feature_groups=["raw", "bogus"]))


def test_window_shorter_than_grid_rejected(cfg, clean_df):
    with pytest.raises(ValueError, match="shorter"):
        build_features(clean_df, replace(cfg, windows_min=[1]))
