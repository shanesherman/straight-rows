"""End-to-end: feeds -> signals -> candidates -> decisions -> submission."""
import hashlib
import platform
from pathlib import Path

import numpy as np
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


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance(data_dir, decisions, sig, th=None):
    """Everything needed to tell whether two submissions came from the same inputs and code."""
    data_dir = Path(data_dir)
    package = Path(__file__).resolve().parent
    code = hashlib.sha256()
    for f in sorted(package.glob("*.py")):
        code.update(f.name.encode() + b"\0" + f.read_bytes())
    counts = {}
    for d in decisions:
        counts[d.cls] = counts.get(d.cls, 0) + 1
    return {
        "code_sha256": code.hexdigest(),
        "inputs_sha256": {str(f.relative_to(data_dir)): _sha256(f)
                          for f in sorted(data_dir.rglob("*")) if f.is_file() and "labels" not in f.parts and not f.name.startswith(".")},
        "labels_used": "development labels only (threshold choice and evaluation); never the evaluation window",
        "thresholds": {**THRESHOLDS, **(th or {})},
        "s4_response": {"s4": [round(float(x), 3) for x in sig.response.knots_s4],
                        "loss_db": [round(float(x), 2) for x in sig.response.knots_db]},
        "detections": counts,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }


def evaluate_dev(ds, th=None, use_sda=True):
    """Score the development window against its 12 labels with the official scorer."""
    decisions, _ = analyse(ds, DEV_START, DEV_END, th, use_sda=use_sda)
    sub = {"detections": [d.to_json() for d in decisions]}
    return score(sub, {"episodes": ds.labels}), decisions
