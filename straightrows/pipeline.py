"""End-to-end: feeds -> signals -> candidates -> decisions -> submission."""
import pandas as pd

from . import detectors, evidence, signals
from .constants import DEV_END, DEV_START, THRESHOLDS
from .official_scorer import score

MAX_DETECTIONS = 500


def analyse(ds, start, end, th=None, use_sda=True):
    """Detect and attribute every episode whose onset falls in [start, end)."""
    th = {**THRESHOLDS, **(th or {})}
    sig = signals.build(ds, th, fit_window=(DEV_START, DEV_END - pd.Timedelta("1s")), use_sda=use_sda)
    mp, mp_mask = detectors.multipath(sig, ds, th)
    candidates = (
        detectors.space_weather(sig, ds, th) + detectors.jamming(sig, ds, th)
        + detectors.spoofing(sig, ds, th) + detectors.equipment(sig, ds, th, mp_mask) + mp
    )
    kept = []
    for c in candidates:
        if start <= c.start < end:
            c.end = min(c.end, end)
            kept.append(c)
    kept.sort(key=lambda c: (c.start, c.cls, c.tractors))
    decisions = evidence.build(kept, ds, sig)
    return decisions, sig


def submission(decisions, handle, repo_url, display_name=None):
    participant = {"github_handle": handle}
    if display_name:
        participant["display_name"] = display_name
    detections = [d.to_json() for d in decisions]
    if len(detections) > MAX_DETECTIONS:
        raise ValueError(f"{len(detections)} detections exceed the {MAX_DETECTIONS} limit")
    return {"participant": participant, "repository_url": repo_url, "detections": detections}


def evaluate_dev(ds, th=None, use_sda=True):
    """Score the development window against its 12 labels with the official scorer."""
    decisions, _ = analyse(ds, DEV_START, DEV_END, th, use_sda=use_sda)
    sub = {"detections": [d.to_json() for d in decisions]}
    return score(sub, {"episodes": ds.labels}), decisions
