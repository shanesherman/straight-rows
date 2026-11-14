"""Assurance study on the labelled development window.

1. Comparison with the official starter baseline (same scorer, same labels).
2. Ablation: farm telemetry only (no S4 in the regional model).
3. Threshold sensitivity: every numeric threshold scaled by 0.7 / 0.85 / 1.15 / 1.3.
4. Noise injection: extra Gaussian noise on C/N0 and cross-track error.
5. Feed degradation: S4 shifted +/-30 min, RFI reports removed.

Thresholds were tuned on this same window, so a 40/40 dev score is in-sample.
This study measures how much margin each decision has, which is the honest
proxy available for out-of-sample behaviour with only 12 labels.
"""
import copy
import json
import time

import numpy as np
import pandas as pd

from . import pipeline
from .constants import DEV_END, DEV_START, THRESHOLDS
from .official_scorer import score

INTEGER_KEYS = {"smooth_s", "regional_window_s", "link_window_s", "drop_window_s", "mp_pre_s", "mp_post_s",
                "sw_min_minutes", "jam_min_minutes", "spoof_min_minutes", "link_min_minutes",
                "fault_min_minutes", "jam_bridge_s"}


def _summary(result):
    return {
        "score": round(result["diagnostic_score"], 2), "macro_f1": round(result["macro_f1"], 3),
        "timing": round(result["mean_timing_credit"], 3), "false_positives": len(result["false_positives"]),
        "matched": sum(e["correct"] for e in result["episodes"]),
    }


def _dev(ds, th=None, use_sda=True):
    result, decisions = pipeline.evaluate_dev(ds, th, use_sda=use_sda)
    return _summary(result), decisions


def starter_baseline(ds, data_dir):
    from benchmarks import starter_baseline as base
    base.EVAL_START, base.EVAL_END = DEV_START, DEV_END
    dets = base.detect(data_dir)
    return _summary(score({"detections": dets}, {"episodes": ds.labels}))


def _scaled(key, value, factor):
    v = value * factor
    if key in INTEGER_KEYS:
        v = max(1, int(round(v)))
        if key.endswith("_s") and key not in ("jam_bridge_s", "mp_pre_s", "mp_post_s") and v % 2 == 0:
            v += 1
    return v


def sensitivity(ds, factors=(0.7, 0.85, 1.15, 1.3)):
    rows = []
    for key, value in THRESHOLDS.items():
        for f in factors:
            s, _ = _dev(ds, {key: _scaled(key, value, f)})
            rows.append({"threshold": key, "factor": f, **s})
    return rows


def with_noise(ds, cn0_sigma=0.0, xte_sigma=0.0, seed=7):
    rng = np.random.default_rng(seed)
    noisy = copy.copy(ds)
    noisy.wide = dict(ds.wide)
    if cn0_sigma:
        # independent noise on mean and minimum C/N0, so the mean-minus-min gap used by the
        # multipath and antenna-fault detectors is perturbed too
        noisy.wide["cn0"] = ds.wide["cn0"] + rng.normal(0, cn0_sigma, ds.wide["cn0"].shape)
        noisy.wide["min_cn0"] = ds.wide["min_cn0"] + rng.normal(0, cn0_sigma, ds.wide["min_cn0"].shape)
    if xte_sigma:
        noisy.wide["xte"] = ds.wide["xte"] + rng.normal(0, xte_sigma, ds.wide["xte"].shape)
    return noisy


def run(ds, output, data_dir="data"):
    t0 = time.time()
    report = {}
    full, decisions = _dev(ds)
    report["full_system"] = full
    report["starter_baseline"] = starter_baseline(ds, data_dir)
    report["farm_only_ablation"], _ = _dev(ds, use_sda=False)

    noise = []
    for cn0_sigma, xte_sigma in [(1, 0), (2, 0), (3, 0), (0, 0.05), (0, 0.1), (2, 0.1)]:
        s, _ = _dev(with_noise(ds, cn0_sigma, xte_sigma))
        noise.append({"cn0_sigma_db": cn0_sigma, "xte_sigma_m": xte_sigma, **s})
    report["noise_injection"] = noise

    feeds = []
    for minutes in (-30, -15, 15, 30):
        shifted = copy.copy(ds)
        shifted.s4 = ds.s4.copy()
        shifted.s4.index = shifted.s4.index + pd.Timedelta(minutes=minutes)
        s, _ = _dev(shifted)
        feeds.append({"perturbation": f"S4 shifted {minutes:+d} min", **s})
    no_rfi = copy.copy(ds)
    no_rfi.rfi = []
    s, dec = _dev(no_rfi)
    base_conf = {(d.cls, d.start): d.confidence for d in decisions if d.cls == "JAMMING"}
    new_conf = {(d.cls, d.start): d.confidence for d in dec if d.cls == "JAMMING"}
    feeds.append({"perturbation": "all RFI reports removed", **s,
                  "jamming_confidence": {str(k[1]): [base_conf.get(k), new_conf.get(k)] for k in base_conf}})
    report["feed_degradation"] = feeds

    sens = sensitivity(ds)
    report["threshold_sensitivity"] = sens
    worst = {}
    for row in sens:
        k = row["threshold"]
        worst[k] = min(worst.get(k, 99), row["score"])
    report["threshold_worst_case"] = worst
    report["confidence_on_dev"] = [
        {"class": d.cls, "start": str(d.start), "tractors": d.tractors, "confidence": d.confidence}
        for d in decisions if d.cls != "MULTIPATH"
    ]
    report["runtime_s"] = round(time.time() - t0, 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: v for k, v in report.items() if k not in ("threshold_sensitivity",)}, indent=1, default=str))
    return report
