"""Builds small synthetic raw files in the REAL file layouts (no real data needed).

Environmental CSV: columns ``sensorid, jsondata`` (one JSON batch per sensor-day).
Occupancy CSV: columns ``floorspaceid, headcount, collecteddate``.
"""
import json

import numpy as np
import pandas as pd

SENSOR = "6012002000869"
ROOM = "941f0352-bb69-47d0-a9c5-ac29eb4f7319"
OTHER_ROOM = "aaaaaaaa-0000-0000-0000-000000000000"
TZ = "Australia/Melbourne"
NAMES = {"co2": "Carbon dioxide", "temperature": "Temperature", "humidity": "Humidity",
         "voc": "Light Volatile Organic Compounds", "pm2_5": "Particulate matter 2.5"}


def true_occupancy(ts, rng):
    """Weekday working-hours pattern (0 at night / weekends)."""
    h = np.asarray(ts.hour + ts.minute / 60)
    day = np.clip(np.sin(np.pi * (h - 7) / 12), 0, None) * np.asarray(ts.dayofweek < 5)
    return np.round(np.clip(20 * day + rng.normal(0, 1.5, len(ts)), 0, None))


def make_raw_files(folder, start="2024-03-04", days=70, seed=0, placeholder=True,
                   env_name="env.csv", occ_name="occ.csv"):
    """Write ``env.csv`` and ``occ.csv`` into ``folder``; return their paths."""
    rng = np.random.default_rng(seed)
    local = pd.date_range(start, periods=days * 24 * 12, freq="5min", tz=TZ)
    occ = true_occupancy(local, rng)
    epoch = np.asarray((local - pd.Timestamp("1970-01-01", tz="UTC")).total_seconds()).astype("int64")
    n = len(local)
    vals = {"co2": 420 + 12 * occ + rng.normal(0, 15, n),
            "temperature": 21 + 0.05 * occ + rng.normal(0, 0.3, n),
            "humidity": 45 + rng.normal(0, 2, n),
            "voc": 80 + rng.normal(0, 5, n), "pm2_5": 4 + rng.normal(0, 0.5, n)}
    if placeholder:                      # a fixed repeated reading on ~10% of records
        ph = rng.random(n) < 0.10
        vals["co2"][ph], vals["temperature"][ph], vals["humidity"][ph] = 450.0, 20.0, 40.0
    day_id = local.normalize()
    rows = []
    for d in pd.unique(day_id):
        idx = np.where(day_id == d)[0]
        recs = []
        for col, name in NAMES.items():
            values = []
            for i in idx:
                if rng.random() < 0.01:
                    values.append({"time": int(epoch[i]), "errorCode": 3})
                else:
                    values.append({"time": int(epoch[i]), "value": float(vals[col][i])})
            recs.append({"variable": {"name": name}, "values": values})
        rows.append({"sensorid": SENSOR, "jsondata": json.dumps(recs)})
    # another sensor that must be ignored
    rows.append({"sensorid": "999", "jsondata": json.dumps(
        [{"variable": {"name": "Carbon dioxide"}, "values": [{"time": int(epoch[0]), "value": 1.0}]}])})
    env_path = f"{folder}/{env_name}"
    pd.DataFrame(rows).to_csv(env_path, index=False)

    # occupancy events: report on change + a periodic heartbeat, two floorspaces
    events = []
    for room, scale in ((ROOM, 1.0), (OTHER_ROOM, 0.5)):
        o = np.round(occ * scale)
        last = None
        for i in range(0, n, 2):                      # every 10 minutes
            if o[i] != last or i % 36 == 0:
                last = o[i]
                off = int(local[i].utcoffset().total_seconds() // 3600)
                events.append({"floorspaceid": room, "headcount": int(o[i]),
                               "collecteddate": local[i].strftime("%Y-%m-%d %H:%M:%S") + f"+{off:02d}"})
    occ_path = f"{folder}/{occ_name}"
    pd.DataFrame(events).to_csv(occ_path, index=False)
    return env_path, occ_path
