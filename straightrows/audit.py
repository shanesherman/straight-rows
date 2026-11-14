"""Audit the data sources, and test the fleet as a sensor network.

Two questions an aggregator of many data providers has to answer:

1. How far can each feed be trusted? The RFI-report feed is scored against the
   fleet's own telemetry: a report is corroborated when a jamming detection
   explains it (and, where the emitter was geolocated, the estimate falls in the
   report's ellipse), discounted when space weather explains the loss it saw,
   and uncorroborated otherwise. Jamming the fleet detected that no report
   describes counts as a miss. Precision and recall of the feed follow.

2. How many machines does it take to locate an emitter? Each jamming episode is
   geolocated again from every subset of the affected tractors and compared with
   the full-fleet estimate and with the RFI report.
"""
import itertools

import pandas as pd

from . import context as ctx
from .localize import localize
from .pipeline import analyse
from .constants import DEV_END, DEV_START, EVAL_END, EVAL_START


def all_decisions(ds):
    """Decisions over the full 72 h (development + evaluation windows)."""
    dev, sig = analyse(ds, DEV_START, DEV_END)
    ev, _ = analyse(ds, EVAL_START, EVAL_END)
    return [d for d in dev + ev if d.cls != "MULTIPATH"], sig


def _inside(report, lat, lon):
    return ctx.distance_km(lat, lon, report["centerLat"], report["centerLon"]) <= report["semiMajor_km"]


def source_audit(ds, decisions):
    jams = [d for d in decisions if d.cls == "JAMMING"]
    storms = [d for d in decisions if d.cls == "SPACE_WEATHER"]
    rows, explained = [], set()
    for r in ds.rfi:
        row = {"source": r["id"], "ts": str(r["ts"]), "report_confidence": r["confidence"]}
        match = next((d for d in jams if d.start + pd.Timedelta("5min") <= r["ts"] <= d.end + pd.Timedelta("35min")), None)
        storm = next((d for d in storms
                      if d.start - pd.Timedelta("2h") <= r["ts"] <= d.end + pd.Timedelta("2h")), None)
        if match is None and storm is None:
            # a storm that began while machines were parked is still a storm
            s4 = ds.s4.loc[:r["ts"]]
            if len(s4) and s4.iloc[-1] >= 0.25:
                storm = True
        if match is not None:
            explained.add(match.start)
            loc = match.metrics.get("localization")
            row.update(verdict="CORROBORATED", episode=f"JAMMING {match.start:%Y-%m-%d %H:%M}Z",
                       delay_min=round((r["ts"] - match.start).total_seconds() / 60, 1))
            if loc:
                row.update(estimate_km_from_centre=round(ctx.distance_km(
                    loc["lat"], loc["lon"], r["centerLat"], r["centerLon"]), 2),
                    estimate_inside_ellipse=_inside(r, loc["lat"], loc["lon"]))
        elif storm is not None:
            row.update(verdict="DISCOUNTED",
                       reason="loss at that time is uniform across the fleet and explained by S4 (space weather)")
        else:
            row.update(verdict="UNCORROBORATED", reason="no matching loss in the fleet telemetry")
        rows.append(row)
    missed = [{"episode": f"JAMMING {d.start:%Y-%m-%d %H:%M}Z", "tractors": d.tractors,
               "verdict": "MISSED_BY_FEED",
               "estimated_source": (None if not d.metrics.get("localization") else
                                    [round(d.metrics["localization"]["lat"], 4),
                                     round(d.metrics["localization"]["lon"], 4)])}
              for d in jams if d.start not in explained]
    corroborated = sum(r["verdict"] == "CORROBORATED" for r in rows)
    score = {
        "feed": "regional RFI monitor",
        "reports": len(rows), "corroborated": corroborated,
        "precision": round(corroborated / len(rows), 2) if rows else None,
        "jamming_detected": len(jams), "jamming_reported": len(jams) - len(missed),
        "recall": round((len(jams) - len(missed)) / len(jams), 2) if jams else None,
    }
    return {"reports": rows, "unreported": missed, "trust": score}


def fleet_sensor_study(ds, sig, decisions):
    out = []
    for d in (d for d in decisions if d.cls == "JAMMING"):
        full = d.metrics.get("localization")
        if not full:
            continue
        report = next((ds_r for m in d.metrics.get("rfi", []) for ds_r in ds.rfi if ds_r["id"] == m["id"]), None)
        for k in range(1, len(d.tractors) + 1):
            for subset in itertools.combinations(d.tractors, k):
                est = localize(ds, sig, d.start, d.end, list(subset))
                if not est:
                    continue
                row = {"episode": f"{d.start:%Y-%m-%d %H:%M}Z", "tractors": "+".join(subset), "machines": k,
                       "fit_rms_db": round(est["rms_db"], 2),
                       "km_from_full_fleet": round(ctx.distance_km(est["lat"], est["lon"], full["lat"], full["lon"]), 3)}
                if report:
                    row.update(report=report["id"], inside_report_ellipse=_inside(report, est["lat"], est["lon"]))
                out.append(row)
    return out


def run(ds):
    decisions, sig = all_decisions(ds)
    return {"source_audit": source_audit(ds, decisions), "fleet_sensor_study": fleet_sensor_study(ds, sig, decisions)}
