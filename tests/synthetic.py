"""Tiny synthetic fleet for detector unit tests (no dataset download needed)."""
import numpy as np
import pandas as pd

from straightrows.constants import TRACTORS
from straightrows.data import Dataset

START = pd.Timestamp("2026-05-06T12:00:00Z")
FIELD_LAT, FIELD_LON = 46.9035, -97.199


def fleet(hours=3, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range(START, periods=hours * 3600, freq="s")
    n = len(index)
    wide = {}
    # tractors drive north-south rows 700 m long, 12 m apart, 300 m between machines
    t = np.arange(n)
    leg = 700 / 4.5
    phase = (t % (2 * leg)) / leg
    north = np.where(phase < 1, phase, 2 - phase) * 700 - 350
    lat = pd.DataFrame({k: FIELD_LAT + north / 110570 for k in TRACTORS}, index=index)
    lon = pd.DataFrame({k: FIELD_LON + (i * 300 - 300 + (t // leg) * 12) / 76000 for i, k in enumerate(TRACTORS)},
                       index=index)
    wide["lat"], wide["lon"] = lat, lon
    wide["cn0"] = pd.DataFrame(45 + rng.normal(0, 0.6, (n, 3)), index=index, columns=list(TRACTORS))
    wide["min_cn0"] = wide["cn0"] - 5 - np.abs(rng.normal(0, 1, (n, 3)))
    wide["xte"] = pd.DataFrame(np.abs(rng.normal(0.02, 0.01, (n, 3))), index=index, columns=list(TRACTORS))
    wide["speed"] = pd.DataFrame(4.5, index=index, columns=list(TRACTORS))
    wide["link"] = pd.DataFrame(1.0, index=index, columns=list(TRACTORS))
    wide["rtk"] = pd.DataFrame(1.0, index=index, columns=list(TRACTORS))
    wide["sats"] = pd.DataFrame(12.0, index=index, columns=list(TRACTORS))
    s4 = pd.Series(0.05, index=pd.date_range(START - pd.Timedelta("1h"), periods=(hours + 2) * 4, freq="15min"))
    kp = pd.Series(2.0, index=pd.date_range(START - pd.Timedelta("6h"), periods=(hours + 8) // 3 + 3, freq="3h"))
    dst = pd.Series(-10.0, index=pd.date_range(START - pd.Timedelta("6h"), periods=hours + 8, freq="h"))
    return Dataset(tele=None, wide=wide, kp=kp, dst=dst, s4=s4, advisories=[], rfi=[])


def at(ds, minute):
    return ds.wide["cn0"].index[0] + pd.Timedelta(minutes=minute)
