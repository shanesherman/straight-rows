"""Space Domain Awareness context: Kp, Dst, S4, advisories and RFI reports."""
import math

import pandas as pd

from .constants import TRACTORS

KM_PER_DEG_LAT = 110.57


def _at(series: pd.Series, ts):
    before = series.loc[:ts]
    return (before.index[-1], float(before.iloc[-1])) if len(before) else (None, float("nan"))


def space_context(ds, start, end):
    lookback = start - pd.Timedelta("6h")
    kp_ts, kp_now = _at(ds.kp, start)
    kp_win = ds.kp.loc[lookback:end]
    s4_ts, s4_now = _at(ds.s4, start)
    s4_during = ds.s4.loc[start:end - pd.Timedelta("1s")]
    s4_before = ds.s4.loc[start - pd.Timedelta("2h"):start - pd.Timedelta("1s")]
    dst_win = ds.dst.loc[lookback:end]
    return {
        "kp_at_onset": kp_now, "kp_at_onset_ts": kp_ts,
        "kp_max_6h": float(kp_win.max()) if len(kp_win) else float("nan"),
        "kp_max_ts": kp_win.idxmax() if len(kp_win) else None,
        "s4_at_onset": s4_now, "s4_at_onset_ts": s4_ts,
        "s4_mean": float(s4_during.mean()) if len(s4_during) else s4_now,
        "s4_max": float(s4_during.max()) if len(s4_during) else s4_now,
        "s4_before_mean": float(s4_before.mean()) if len(s4_before) else float("nan"),
        "dst_min": float(dst_win.min()) if len(dst_win) else float("nan"),
        "dst_min_ts": dst_win.idxmin() if len(dst_win) else None,
    }


def advisories_in_effect(ds, start, end, kind=None):
    """Advisories issued up to 48 h before the episode (storm advisories cover 24-48 h)."""
    out = []
    for a in ds.advisories:
        if start - pd.Timedelta("48h") <= a["ts"] <= end and (kind is None or a["type"] == kind):
            out.append(a)
    return out


def fleet_position(ds, start, end, tractors=TRACTORS):
    lat = ds.wide["lat"].loc[start:end, list(tractors)].mean().mean()
    lon = ds.wide["lon"].loc[start:end, list(tractors)].mean().mean()
    return float(lat), float(lon)


def distance_km(lat1, lon1, lat2, lon2):
    dy = (lat2 - lat1) * KM_PER_DEG_LAT
    dx = (lon2 - lon1) * KM_PER_DEG_LAT * math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot(dx, dy)


def rfi_matches(ds, start, end, tractors=TRACTORS):
    """RFI reports that could describe this episode.

    Reports arrive 10-30 min late, can miss events, and can be false alarms, so
    a match is corroboration, not proof. A report counts if it was issued between
    5 min after onset and 35 min after the end, and its uncertainty ellipse (taken
    conservatively as a circle of the semi-major axis) reaches the fleet.
    """
    lat, lon = fleet_position(ds, start, end, tractors)
    out = []
    for r in ds.rfi:
        if not (start + pd.Timedelta("5min") <= r["ts"] <= end + pd.Timedelta("35min")):
            continue
        d = distance_km(lat, lon, r["centerLat"], r["centerLon"])
        out.append({
            "id": r["id"], "ts": r["ts"], "confidence": r["confidence"],
            "delay_min": (r["ts"] - start).total_seconds() / 60,
            "distance_km": d, "semi_major_km": r["semiMajor_km"],
            "covers_fleet": d <= r["semiMajor_km"], "text": r["text"],
        })
    return out


def rfi_nearby(ds, start, end, slack="2h"):
    """Any RFI report within `slack` of the episode (used to discount or rule out)."""
    slack = pd.Timedelta(slack)
    return [r for r in ds.rfi if start - slack <= r["ts"] <= end + slack]
