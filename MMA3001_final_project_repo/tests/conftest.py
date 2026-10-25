import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, os.path.dirname(__file__))

import pytest

from smart_infrastructure import Config
from smart_infrastructure.data import (build_table, load_environment, load_occupancy)
from synth import make_raw_files


@pytest.fixture(scope="session")
def raw_files(tmp_path_factory):
    return make_raw_files(str(tmp_path_factory.mktemp("raw")))


@pytest.fixture(scope="session")
def cfg():
    return Config(window_end="2024-06-14")


@pytest.fixture(scope="session")
def prepared(raw_files, cfg):
    env, audit = load_environment(raw_files[0], cfg)
    occ, n_events = load_occupancy(raw_files[1], cfg)
    df = build_table(env, occ)
    return env, occ, df, audit
