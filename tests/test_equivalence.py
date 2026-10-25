"""Regression test: the package must reproduce the final notebook's data preparation.

``reference_table`` below is the notebook's data-preparation code (cells 2-4) copied
line for line (only the constants are read from ``Config``). If the refactoring ever
changes a number, this test fails.
"""
import json

import pandas as pd

from smart_infrastructure import Config
from smart_infrastructure.data import build_table, load_environment, load_occupancy


def reference_table(ENV_CSV, OCC_CSV, c):
    ENV_SENSOR_ID, FLOORSPACE, TZ = c.env_sensor_id, c.floorspace, c.tz
    WINDOW_START, WINDOW_END = c.window_start, c.window_end
    VARS = {'Carbon dioxide': 'co2', 'Temperature': 'temperature', 'Humidity': 'humidity',
            'Light Volatile Organic Compounds': 'voc', 'Particulate matter 2.5': 'pm2_5'}
    raw = pd.read_csv(ENV_CSV, usecols=['sensorid', 'jsondata'])
    raw = raw[raw['sensorid'].astype(str) == ENV_SENSOR_ID]
    rows = {}
    for js in raw['jsondata']:
        for rec in json.loads(js):
            name = VARS.get(rec.get('variable', {}).get('name'))
            if name is None:
                continue
            for v in rec.get('values', []):
                if 'value' in v and 'errorCode' not in v:
                    rows.setdefault(v['time'], {})[name] = v['value']
    env = pd.DataFrame.from_dict(rows, orient='index').apply(pd.to_numeric, errors='coerce').sort_index()
    env.index = pd.to_datetime(pd.to_numeric(env.index), unit='s', utc=True).tz_convert(TZ).tz_localize(None)
    env = env.dropna(subset=['co2'])
    key = ['co2', 'temperature', 'humidity']
    counts = env[key].value_counts()
    placeholder = counts[counts / len(env) >= 0.05].index
    env = env[~env.set_index(key).index.isin(placeholder)]
    env = env[~env.index.duplicated(keep='last')].sort_index()
    env = env.loc[WINDOW_START:WINDOW_END]
    env.index = env.index.ceil('15min')
    env = env.groupby(level=0).mean()
    env = env.reindex(pd.date_range(env.index.min(), env.index.max(), freq='15min')).ffill(limit=3)
    parts = [c_[c_['floorspaceid'] == FLOORSPACE]
             for c_ in pd.read_csv(OCC_CSV, usecols=['floorspaceid', 'headcount', 'collecteddate'], chunksize=1_000_000)]
    ev = pd.concat(parts)
    t = pd.to_datetime(ev['collecteddate'], format='ISO8601', utc=True).dt.tz_convert(TZ).dt.tz_localize(None)
    occ = pd.Series(pd.to_numeric(ev['headcount'], errors='coerce').to_numpy(), index=t).dropna().sort_index()
    occ = occ[~occ.index.duplicated(keep='last')]
    occ = occ.resample('1min').last().ffill(limit=4320)
    occ = occ.resample('15min', closed='right', label='right').mean().rename('occupancy')
    env['co2_chg1h'] = env['co2'].diff(4)
    df = env.join(occ, how='inner').dropna(subset=['occupancy', 'co2'])
    df['hour'] = df.index.hour + df.index.minute / 60
    df['weekend'] = (df.index.dayofweek >= 5).astype(int)
    return df


def test_package_reproduces_notebook_modelling_table(raw_files):
    c = Config()
    expected = reference_table(raw_files[0], raw_files[1], c)
    env, _ = load_environment(raw_files[0], c)
    occ, _ = load_occupancy(raw_files[1], c)
    got = build_table(env, occ)
    assert list(got.columns) == list(expected.columns)
    pd.testing.assert_frame_equal(got, expected)
    assert len(got) > 1000
