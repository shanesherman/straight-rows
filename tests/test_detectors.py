import math

import numpy as np
import pandas as pd
import pytest

from straightrows import detectors, evidence, localize, signals
from straightrows.constants import ACTION, THRESHOLDS as TH
from straightrows.context import KM_PER_DEG_LAT

from .synthetic import at, fleet

# A fixed S4 response so synthetic tests do not depend on fitting data.
RESPONSE = signals.S4Response(np.array([0.0, 0.3, 0.45, 0.8, 1.5]), np.array([0.0, 0.0, 10.0, 16.0, 16.0]))


def run(ds):
    sig = signals.build(ds, TH, response=RESPONSE)
    return sig, {
        "sw": detectors.space_weather(sig, ds, TH), "jam": detectors.jamming(sig, ds, TH),
        "spoof": detectors.spoofing(sig, ds, TH), "eq": detectors.equipment(sig, ds, TH),
    }


def test_intervals_bridge_and_min_length():
    idx = pd.date_range("2026-05-06", periods=20, freq="s", tz="UTC")
    mask = pd.Series([0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1], index=idx).astype(bool)
    runs = detectors.intervals(mask, bridge_s=1, min_s=3)
    assert runs == [(idx[1], idx[6]), (idx[16], idx[19] + pd.Timedelta("1s"))]


def test_quiet_fleet_has_no_detections():
    _, found = run(fleet())
    assert not any(found.values())


def test_correction_link_fault_on_one_machine():
    ds = fleet()
    a, b = at(ds, 40), at(ds, 90)
    rng = np.random.default_rng(1)
    drop = rng.random(len(ds.wide["link"].loc[a:b])) < 0.3
    ds.wide["link"].loc[a:b, "T2"] = np.where(drop, 0.0, 1.0)
    ds.wide["rtk"].loc[a:b, "T2"] = np.where(drop, 0.0, 1.0)
    _, found = run(ds)
    (c,) = found["eq"]
    assert c.tractors == ["T2"] and c.metrics["kind"] == "correction_link"
    assert abs((c.start - a).total_seconds()) < 60 and abs((c.end - b).total_seconds()) < 60
    assert not found["jam"] and not found["sw"] and not found["spoof"]


def test_spoofing_ramp_detected_with_onset():
    ds = fleet()
    a, b = at(ds, 60), at(ds, 100)
    ramp = np.minimum(np.arange(len(ds.wide["xte"].loc[a:b])) / 60 * 0.25, 8.0)
    ds.wide["xte"].loc[a:b, "T1"] += ramp
    _, found = run(ds)
    (c,) = found["spoof"]
    assert c.tractors == ["T1"]
    assert abs((c.start - a).total_seconds()) < 120
    assert not found["jam"] and not found["eq"]


def test_space_weather_versus_jamming_uses_s4():
    # identical uniform 10 dB loss: with S4 0.45 it is space weather, with quiet S4 it is interference
    for s4, expected in [(0.45, "sw"), (0.05, "jam")]:
        ds = fleet()
        a, b = at(ds, 50), at(ds, 130)
        ds.wide["cn0"].loc[a:b] -= 10
        ds.wide["min_cn0"].loc[a:b] -= 10
        ds.s4.loc[a - pd.Timedelta("15min"):b] = s4
        _, found = run(ds)
        assert found[expected], f"S4={s4} should give {expected}"
        other = "jam" if expected == "sw" else "sw"
        assert not found[other]


def test_jamming_inside_storm_is_a_separate_cause():
    ds = fleet()
    a, b = at(ds, 20), at(ds, 160)
    ds.wide["cn0"].loc[a:b] -= 10
    ds.s4.loc[a - pd.Timedelta("15min"):b] = 0.45
    ja, jb = at(ds, 70), at(ds, 100)
    ds.wide["cn0"].loc[ja:jb] -= 12
    _, found = run(ds)
    assert len(found["sw"]) == 1 and len(found["jam"]) == 1
    jam = found["jam"][0]
    assert abs((jam.start - ja).total_seconds()) < 60
    assert set(jam.tractors) == {"T1", "T2", "T3"}


def test_localize_recovers_synthetic_emitter():
    ds = fleet()
    a, b = at(ds, 30), at(ds, 80)
    e_lat, e_lon = 46.8990, -97.1930
    lat, lon = ds.wide["lat"].loc[a:b], ds.wide["lon"].loc[a:b]
    k_lon = 111.32 * math.cos(math.radians(46.9))
    rng_km = np.hypot((lat - e_lat) * KM_PER_DEG_LAT, (lon - e_lon) * k_lon)
    loss = 8 - 20 * np.log10(rng_km)
    ds.wide["cn0"].loc[a:b] -= loss
    sig, found = run(ds)
    (jam,) = found["jam"]
    loc = localize.localize(ds, sig, jam.start, jam.end)
    error_km = math.hypot((loc["lat"] - e_lat) * KM_PER_DEG_LAT, (loc["lon"] - e_lon) * k_lon)
    assert error_km < 0.15 and loc["rms_db"] < 0.5


def test_detection_json_contract():
    ds = fleet()
    a, b = at(ds, 40), at(ds, 90)
    ds.wide["link"].loc[a:b, "T3"] = np.tile([0.0, 1.0, 1.0], len(ds.wide["link"].loc[a:b]) // 3 + 1)[
        :len(ds.wide["link"].loc[a:b])]
    sig, found = run(ds)
    (d,) = evidence.build(found["eq"], ds, sig)
    j = d.to_json()
    assert j["action"] == ACTION[j["class"]]
    assert 0.05 <= j["confidence"] <= 0.97
    assert j["evidence"] and all(isinstance(s, str) and s for s in j["evidence"])
    assert any(s.startswith("OPERATOR:") for s in j["evidence"])


@pytest.mark.parametrize("cls", list(ACTION))
def test_every_class_has_an_action(cls):
    assert ACTION[cls]
