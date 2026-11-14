# Five-minute demo script

The guide asks the demo to show **an episode, the evidence behind its cause and action, an ambiguous case or
limitation, and how the code reproduces the submitted output**. Timings are targets.

Before you start: open `out/report_eval.html` in a browser and keep a terminal in the repo root with the venv active.

## 0:00 – 0:30 · The question and the answer

> "My tractor is drifting: is it the Sun, an adversary, or my own hardware?"

- The system reads 777,600 telemetry rows plus the SDA feeds and, in about three seconds, outputs
  seven episodes plus every headland reflection, each with a cause, an action, a confidence, and evidence an
  operator can check.
- Show the summary table at the top of the report.

## 0:30 – 1:30 · One idea that separates the Sun from an adversary

Scroll to the **timeline**. Point at the dashed "S4-consistent regional level" line.

- The Sun lowers C/N0 **uniformly** on every tractor, and by an amount that **S4 predicts**. I fitted that S4→loss
  curve on the development storm (S4 0.44 → ~10 dB, 0.78 → ~16 dB).
- Anything beyond that level is **excess**, which has to come from something local: interference, an antenna,
  a reflection.
- The decomposition is what lets the system call **jamming inside a storm** (TRN-05 in the dev data) a separate
  cause. Without the S4 feed, that episode disappears into the storm: the farm-only ablation drops from 40 to
  36.6 on the dev diagnostic.

## 1:30 – 2:45 · One episode end to end: E02 jamming at 02:12Z

Open card **E02**.

1. Chart: C/N0 drops 10–12 dB in one step on all three machines while S4 is 0.08 and Kp is 2.5, so space
   weather is ruled out.
2. The loss oscillates with each row pass. That is range to the emitter changing as the tractors drive.
3. **Geolocation:** a free-space path-loss model fits 363 samples with 0.29 dB RMS and places the source about
   0.5 km ESE of the field centre. The RFI-003 report arrives 27 minutes later (inside the stated 10–30 minute
   delay), and our estimate lies inside its uncertainty ellipse. The ellipse was never used in the fit, so this
   is independent confirmation.
4. Action `HALT_AND_REPORT`, and the report includes the estimated emitter coordinates.

Then point at **E05 (18:30Z)**: jamming with **no RFI report**. Reports are incomplete, so the evidence lists
that as an AGAINST item, but the telemetry alone still geolocates the source.

## 2:45 – 3:45 · Ambiguity and limitations

- **Spoofing (E04, E07):** the guidance monitor drifts by up to 8 m while C/N0, satellites and RTK fix all look
  healthy. The receiver is confidently wrong. Reported lat/lon is *not* displaced in this data, so the system
  relies on the separate guidance-monitor channel; a real deployment needs IMU or odometry cross-checks.
- **Multipath:** every north-headland turn shows a ~17 s reflection, 486 passes in the evaluation day. Each is
  reported as a separate ≤3 min `CONTINUE_FALSE_ALARM`, because the rules exempt recurrent short multipath and
  only one is scored. Say plainly that this is a reporting choice made with the scorer in mind; an operator UI
  would collapse them into a single "known reflector" alert.
- Thresholds were tuned on the 12 development labels, so 40/40 on dev is in-sample. The robustness study
  measures the margin instead: ±30 % threshold sweeps, injected noise, S4 shifted ±30 min, RFI removed.

## 3:45 – 4:45 · Reproduce the submission live

```bash
python -m straightrows submit --data data --output /tmp/check.json \
  --github-handle shanesherman --repo-url https://github.com/shanesherman/straight-rows
python -m straightrows validate /tmp/check.json
diff <(python -c "import json;print(json.load(open('/tmp/check.json'))['detections'])") \
     <(python -c "import json;print(json.load(open('submission.json'))['detections'])") && echo IDENTICAL
python -m straightrows evaluate --data data     # 12/12 dev episodes, 40.00/40
python -m pytest -q                              # 20 tests
```

## 4:45 – 5:00 · Close

Detect, attribute, act, and show your work. Every recommendation carries the measurements and SDA observations
behind it, plus what argues against it.

---

# Q&A preparation

**Why not a machine-learning classifier?** There are 12 labelled episodes, and each class has 1–3 examples.
A learned model would memorise them. Physically motivated features (common-mode vs excess loss, peer
comparison, ramp shape) generalise better and produce the explanation directly. The labels were used to choose
thresholds and to fit the one learned component, the S4 response curve.

**How is confidence computed?** Each class has explicit checks. A passing check adds its weight to the
log-odds; a failing one subtracts its penalty. Absent RFI corroboration only costs a little, because reports are
known to be incomplete. Confidence is the logistic of that sum, capped at 0.97. On the dev window every
detection is correct, so calibration can't be fitted honestly from 12 labels; the cap and the AGAINST items
are the guard against overconfidence.

**What if the jammer were equidistant from all machines?** The loss would look uniform. The system then asks
whether S4 explains it. If S4 is quiet, a uniform loss is still attributed to interference (unit test
`test_space_weather_versus_jamming_uses_s4`).

**What if S4 lags the farm impact?** Acceptance uses a ±30 min band, and shifting S4 by ±30 min leaves the
dev score at 40/40.

**Which tractors count as affected by jamming?** A machine counts if its mean excess loss is ≥4.5 dB, or if it
lost RTK fix it previously held. In TRN-04, T2 lost about 3 dB but kept RTK, and the label agrees it was
unaffected.

**Why does the space-weather episode start at 11:00 on May 5 when C/N0 fell at 09:45?** The machines were
parked until 11:00, and scored episodes concern working tractors. The evidence states both times.

**How would you change X live?** All thresholds live in `straightrows/constants.py`. Each detector is one
function in `straightrows/detectors.py`. Each class's evidence and confidence is one function in
`straightrows/evidence.py`.
