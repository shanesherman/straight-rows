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
