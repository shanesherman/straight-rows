"""Regression tests on the official dataset (skipped if data/ has not been downloaded)."""
import json
from pathlib import Path

import pytest

from straightrows import data, pipeline
from straightrows.constants import EVAL_END, EVAL_START
from straightrows.official_validator import validate

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
pytestmark = pytest.mark.skipif(not (DATA / "farm/tractor_telemetry.csv.gz").exists(),
                                reason="dataset not downloaded (python scripts/get_data.py)")


@pytest.fixture(scope="module")
def ds():
    return data.load(DATA)


def test_development_window_scores_full_marks(ds):
    result, decisions = pipeline.evaluate_dev(ds)
    assert all(e["correct"] and e["action_agreement"] for e in result["episodes"])
    assert result["false_positives"] == []
    assert result["diagnostic_score"] == pytest.approx(40.0)


def test_onsets_within_full_credit(ds):
    result, _ = pipeline.evaluate_dev(ds)
    assert max(abs(e["onset_error_s"]) for e in result["episodes"]) <= 300


def test_overlapping_causes_are_separate(ds):
    _, decisions = pipeline.evaluate_dev(ds)
    major = [d for d in decisions if d.cls != "MULTIPATH"]
    t2 = [d.cls for d in major if d.tractors == ["T2"] and d.start.day == 4 and d.start.hour in (12, 13)]
    assert sorted(t2) == ["EQUIPMENT_FAULT", "SPOOFING"]


def test_submission_is_valid_deterministic_and_committed(ds):
    decisions, _ = pipeline.analyse(ds, EVAL_START, EVAL_END)
    sub = pipeline.submission(decisions, "shanesherman", "https://github.com/shanesherman/straight-rows")
    assert validate(sub) == []
    again, _ = pipeline.analyse(ds, EVAL_START, EVAL_END)
    assert [d.to_json() for d in again] == sub["detections"]
    committed = ROOT / "submission.json"
    if committed.exists():
        assert json.loads(committed.read_text())["detections"] == sub["detections"]


def test_source_audit_scores_the_rfi_feed(ds):
    from straightrows import audit
    decisions, _ = audit.all_decisions(ds)
    sa = audit.source_audit(ds, decisions)
    verdicts = {r["source"]: r["verdict"] for r in sa["reports"]}
    assert verdicts == {"RFI-001": "CORROBORATED", "RFI-002": "CORROBORATED",
                        "RFI-003": "CORROBORATED", "RFI-004": "DISCOUNTED"}
    assert all(r["estimate_inside_ellipse"] for r in sa["reports"] if r["verdict"] == "CORROBORATED")
    assert len(sa["unreported"]) == 2
    assert sa["trust"]["precision"] == 0.75 and sa["trust"]["recall"] == 0.6


def test_single_tractor_geolocates_every_jammer(ds):
    from straightrows import audit
    decisions, sig = audit.all_decisions(ds)
    single = [r for r in audit.fleet_sensor_study(ds, sig, decisions) if r["machines"] == 1]
    assert len(single) == 14
    assert max(r["km_from_full_fleet"] for r in single) < 0.2
    assert all(r["inside_report_ellipse"] for r in single if "report" in r)


def test_provenance_is_deterministic(ds):
    decisions, sig = pipeline.analyse(ds, EVAL_START, EVAL_END)
    a = pipeline.provenance(DATA, decisions, sig)
    b = pipeline.provenance(DATA, decisions, sig)
    assert a == b
    assert set(a["inputs_sha256"]) == {"farm/tractor_telemetry.csv.gz", "sda/constellation_advisories.json",
                                       "sda/rfi_reports.json", "sda/scintillation_series.json", "sda/sgi_indices.json"}
