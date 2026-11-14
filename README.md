# Straight Rows: is it the Sun, an adversary, or my own hardware?

A detection, attribution and decision system for degraded GNSS guidance on a three-tractor fleet near
Casselton, ND, built for the Bluestaq × NDSU **Straight Rows** hackathon. It fuses 1 Hz farm telemetry with
Space Domain Awareness feeds (Kp, Dst, S4 scintillation, constellation advisories, RF-interference reports).
For each episode it outputs the cause, the affected machines, an operator action, a calibrated confidence, and
evidence a person can check, including what argues *against* the call.

| | Result |
|---|---|
| Development window (12 labelled episodes, official scorer) | **12/12 correct class and action, all onsets within 2 min, 0 false positives, diagnostic 40.00/40** |
| Starter baseline, same scorer, same labels | 7/12, 5 false positives, diagnostic 20.15/40 |
| Evaluation window (submitted) | 7 episodes + 486 recurrent headland multipath passes |
| Runtime | ~3 s on a laptop, deterministic, no network or credentials |

> **In-sample caveat.** Thresholds were chosen by studying the development window, so 40/40 there is
> in-sample. The [assurance study](#assurance-and-rigor) measures how much margin each decision has instead.

## Reproduce

Python 3.11 or newer (tested on 3.14). Dependencies are pinned in `requirements.txt`.

```bash
git clone https://github.com/shanesherman/straight-rows.git
cd straight-rows
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/get_data.py                             # official dataset; SHA-256 verified into ./data
```

The dataset is not committed. `scripts/get_data.py` downloads `StraightRows_Student_Package.zip` from the
official release and checks its SHA-256 against `DATASET_SHA256`. If you already have the zip:
`python scripts/get_data.py --zip PATH`.

**Regenerate the submitted `submission.json`** (byte-identical detections; `tests/test_dataset.py` checks this):

```bash
python -m straightrows submit --data data --output submission.json \
  --github-handle shanesherman --repo-url https://github.com/shanesherman/straight-rows
python -m straightrows validate submission.json        # the official validator, vendored
```

`submit` also writes `submission.provenance.json`: SHA-256 of the code and of every input file, the thresholds,
the fitted S4 response, and library versions. If two runs report the same hashes, they used the same code and the same inputs.

```bash
```

Everything else:

```bash
python -m straightrows evaluate --data data            # score the dev window with the official scorer
python -m straightrows evaluate --data data --farm-only  # ablation without SDA feeds
python -m straightrows report --data data              # out/report_eval.html (a copy is committed at docs/report_eval.html)
python -m straightrows report --data data --window dev # out/report_dev.html
python -m straightrows robustness --data data          # out/robustness.json (~5 min)
python -m straightrows audit --data data               # RFI-feed trust score + single-tractor geolocation
python -m pytest -q                                    # 20 tests (synthetic unit tests + dataset regression)
```

## Evaluation-window results (submitted)

| # | Time (UTC, 2026-05-06) | Tractors | Cause | Action | Conf. | Key evidence |
|---|---|---|---|---|---|---|
| E01 | 01:00:02–02:00:00 | T1 | EQUIPMENT_FAULT (antenna) | SERVICE_MACHINE_OTHERS_CONTINUE | 0.94 | deep dropouts in 17 % of seconds on T1, 0 % on neighbours; mean C/N0 normal |
| E02 | 02:12:00–02:54:00 | T1,T2,T3 | JAMMING | HALT_AND_REPORT | 0.97 | 9.6–12.3 dB excess loss (step up to 10.9 dB in 60 s) with S4 0.08 / Kp 2.5; RFI-003 27 min later; geolocated emitter falls inside RFI-003 ellipse |
| E03 | 13:00:00–15:59:57 | T1,T2,T3 | SPACE_WEATHER | HALT_OR_DEGRADED_MODE | 0.97 | uniform 10.1 dB loss (spread 0.3 dB); S4 0.46 predicts 10.2 dB; Kp 6.8 at 12:00Z |
| E04 | 16:30:01–17:12:00 | T1 | SPOOFING | HALT_CROSSCHECK_AND_REPORT | 0.96 | XTE ramps 0.19 m/min to 8.1 m while C/N0 and RTK fix match neighbours |
| E05 | 18:30:00–19:00:00 | T1,T2,T3 | JAMMING | HALT_AND_REPORT | 0.97 | 5.7–11.9 dB excess loss (step up to 16.8 dB), S4 quiet; **no RFI report**, but telemetry alone geolocates a source 0.6 km NW |
| E06 | 19:48:02–20:29:25 | T2 | EQUIPMENT_FAULT (correction link) | SERVICE_MACHINE_OTHERS_CONTINUE | 0.94 | T2 link down 30 % of the time, neighbours 0 %; signal strength normal |
| E07 | 21:48:01–22:24:00 | T3 | SPOOFING | HALT_CROSSCHECK_AND_REPORT | 0.96 | XTE ramps to 5.1 m with healthy C/N0 and RTK |
| MP | 486 passes across working hours (00:00–03:00 and 11:00–24:00) | each single tractor | MULTIPATH | CONTINUE_FALSE_ALARM | 0.82–0.91 | ~17 s min-C/N0 dip at every north-headland turn; in clear conditions peak XTE has median 0.27 m (294 of 296 passes under 0.5 m); passes during the storm, jamming or spoofing inherit that elevated XTE, fail the 'harmless' check and score 0.82 |

Each detection's `evidence` list in `submission.json` reads like this (E05, abbreviated):

```
JAMMING on T1,T2,T3, 2026-05-06 18:30Z to 19:00Z: RF interference beyond what space weather explains.
SUPPORTS: C/N0 loss beyond what regional S4 explains: T1 8.8 dB, T2 5.7 dB, T3 11.9 dB
SUPPORTS: Abrupt onset at 18:30Z: up to 16.8 dB extra loss within 60 s (...)
SUPPORTS: Space weather quiet: S4 0.09 at 18:30Z, Kp 2.3; scintillation cannot explain the loss
AGAINST: No RFI report matches; reports are incomplete, so attribution rests on the telemetry
SUPPORTS: (...) a free-space point-emitter model (20 dB/decade) fits 255 samples with 0.23 dB RMS, placing
          the source at 46.9058, -97.2057 (+/- 250 m), 0.6 km at bearing 312 deg from the field centre
IMPACT: T1 RTK_FIXED 100%->19%, (...)
OPERATOR: Halt GNSS-guided work on T1,T2,T3; report the interference (estimated source 46.9058, -97.2057
          +/- 250 m) with times to the regional RFI monitor/authorities; resume after C/N0 is back to nominal (...)
```

## Approach

```
 telemetry (1 Hz × 3)        SDA feeds
        │                 S4 · Kp · Dst · advisories · RFI
        ▼                        │
 signals.py ── smoothed C/N0 ────┤
   deficit = 45 − C/N0           │  S4 → expected loss (fitted on the dev storm)
   regional = uniform loss that S4 can explain  ◄─┘
   excess   = deficit − regional   (per tractor)
        │
        ▼
 detectors.py   one detector per physical mechanism, independent so causes can overlap
   SPACE_WEATHER  regional > 3 dB while working
   JAMMING        excess > 3 dB for ≥ 3 min; affected = excess ≥ 4.5 dB or lost RTK; abruptness scored as evidence
   SPOOFING       XTE above cleanest neighbour + storm noise, own C/N0 within 2 dB of regional; onset by ramp fit
   EQUIPMENT      one machine's correction-link loss or deep-dropout rate vs its neighbours
   MULTIPATH      ~17 s min-C/N0 rise at each north-headland turn
        │
        ▼
 localize.py    path-loss geolocation of interference sources from the fleet's own C/N0
 context.py     Kp/Dst/S4 at onset, advisories in effect, RFI reports matched by delay and distance
 evidence.py    explicit SUPPORTS/AGAINST checks → log-odds → confidence; operator guidance;
                overlapping detections grouped into operational events
```

### 1. Separate the Sun from everything else

Scintillation is regional: at farm scale it lowers C/N0 by the same amount on every machine, and by an amount
the regional S4 proxy predicts. The system measures the **uniform** part of the fleet's C/N0 loss (seconds when
all three machines agree within 1 dB) and accepts it as **regional** only when S4 explains it within 4 dB. A
±30 min time band allows for S4's lag. The S4→loss response is a monotone curve fitted on the development
window: S4 0.44 → 10.2 dB, 0.66 → 14.1 dB, 0.78 → 16.1 dB. Everything above the regional level is **excess** and
must come from something local.

This makes the system robust to the case the guide warns about: **causes overlap**. In TRN-05 a jammer switches on
in the middle of a storm and hits all three machines almost equally. A purely farm-side "uniform = space
weather" rule would swallow it. Here S4 predicts 10 dB, the fleet loses 23 dB, and the extra 13 dB becomes a
separate JAMMING detection. Removing the SDA feed costs 3.4 points on the dev diagnostic (below).

### 2. Mechanism-specific detectors

- **Jamming:** excess loss > 3 dB sustained for ≥ 3 min while working. Whether it appeared in one step (a
  switched-on emitter) rather than building gradually is scored as a confidence check, not a gate. A tractor
  counts as affected if its excess is ≥ 4.5 dB or it lost an RTK fix it held before. That matches TRN-04, where
  T2 lost about 3 dB, kept RTK, and is labelled unaffected.
- **Spoofing:** the guidance monitor's cross-track error drifts away from the cleanest neighbour while the
  machine's own C/N0 stays within 2 dB of the regional level (the detection gate). That RTK fix also stays as
  good as the neighbours' is scored as evidence: the receiver is confidently wrong. Storms raise everyone's guidance noise, so the drift threshold adds the fleet's median XTE. Onset is
  found by fitting the linear ramp and extrapolating back to zero. Reported lat/lon is not displaced in this
  dataset, so position is not used as a spoofing cue (see limitations).
- **Equipment fault:** always judged relative to the neighbours in the same field, which experience the same
  sky and the same RF environment. A correction-link fault shows as `base_link_ok` dropping on one machine
  only. An antenna or cable fault shows as intermittent deep dropouts (weakest satellite >15 dB below the mean)
  on one machine only, with mean C/N0 near normal.
- **Multipath:** each north-headland turn produces a ~17 s dip of the weakest satellite (window = turn −8 s
  to +9 s, reproducing both development labels exactly) with negligible guidance impact.

### 3. Geolocating the interferer

As tractors drive their rows, their range to an emitter changes and so does their excess loss. Under
free-space propagation, `excess = A − 20·log10(range)`. A coarse-to-fine grid search over 16 km × 16 km
(solving A in closed form) fits the telemetry alone, and the fit is good (0.16–0.29 dB RMS on four of five jamming
episodes). The RFI reports were **never used in the fit**, so they serve as an independent check:

| Episode | Estimated source vs RFI report |
|---|---|
| TRN-03 (dev) | 0.53 km from RFI-001 centre, ellipse semi-major 3.44 km: inside |
| TRN-04 (dev) | 0.60 km from RFI-002 centre, semi-major 1.63 km: inside |
| E02 (eval) | 1.77 km from RFI-003 centre, semi-major 3.50 km: inside |
| TRN-05, E05 | no report filed; telemetry-only estimate is given to the operator |

The ±250 m uncertainty is a deliberate floor. The statistical radius is smaller, but samples are correlated and
real propagation (terrain, antenna patterns) is not free-space.

### 4. Decisions, confidence and evidence

Each class has a short list of explicit checks (for space weather: uniform across machines, S4 explains the
magnitude, Kp ≥ 5, no local excess, guidance impact, advisory in effect). A passing check adds its weight to the
log-odds; a failing one subtracts its penalty. Missing RFI corroboration costs only a little, because reports
are known to be incomplete. Confidence is the logistic of the sum, clipped to [0.05, 0.97]. Every check
becomes an evidence line marked `SUPPORTS:` or `AGAINST:`, with the timestamps, machines, measurements and SDA
values behind it. The final line is the operator instruction. Overlapping detections (a fault during a storm, a
spoof during a storm) are linked as one **operational event**, while each cause is reported separately, as the
guide asks.

### How the labelled data was used

- To discover each mechanism's signature: the per-class plots are in `out/report_dev.html`.
- To fit the S4→loss response curve. It uses only uniform-loss minutes and no labels.
- To choose thresholds, then verify with the official scorer on the dev window (`python -m straightrows evaluate`).
- As a regression test: `tests/test_dataset.py` requires 12/12, onsets ≤5 min, and the overlapping T2 fault
  and spoof reported as two detections.

## Assurance and rigor

All numbers below come from `python -m straightrows robustness` (`out/robustness.json`), using the official
scorer on the development window.

| Configuration | Dev diagnostic /40 | Correct episodes | False positives |
|---|---|---|---|
| **Full system** | 40.00 | 12/12 | 0 |
| Official starter baseline | 20.15 | 7/12 | 5 |
| Ablation: farm telemetry only (no S4 in the regional model) | 36.61 | 11/12 | 2 |
| + Gaussian noise: C/N0 σ=1 dB, XTE σ=0 m | 40.00 | 12/12 | 0 |
| + Gaussian noise: C/N0 σ=2 dB, XTE σ=0 m | 40.00 | 12/12 | 0 |
| + Gaussian noise: C/N0 σ=3 dB, XTE σ=0 m | 40.00 | 12/12 | 0 |
| + Gaussian noise: C/N0 σ=0 dB, XTE σ=0.05 m | 40.00 | 12/12 | 0 |
| + Gaussian noise: C/N0 σ=0 dB, XTE σ=0.1 m | 40.00 | 12/12 | 0 |
| + Gaussian noise: C/N0 σ=2 dB, XTE σ=0.1 m | 40.00 | 12/12 | 0 |
| S4 shifted -30 min | 40.00 | 12/12 | 0 |
| S4 shifted -15 min | 40.00 | 12/12 | 0 |
| S4 shifted +15 min | 40.00 | 12/12 | 0 |
| S4 shifted +30 min | 40.00 | 12/12 | 0 |
| all RFI reports removed | 40.00 | 12/12 | 0 |

Noise is drawn independently for mean and minimum C/N0, so the mean-minus-min gap that drives the multipath
and antenna-fault detectors is perturbed too (σ=3 dB on each gives about 4.2 dB on the gap).

**Threshold sensitivity.** Each of the 28 thresholds in `constants.py` was scaled independently by ×0.7, ×0.85, ×1.15 and ×1.3 (112 runs). 27 of 28 keep 40/40 at every setting.
 The exceptions:

| Threshold | Factor | Dev diagnostic | Correct | False positives |
|---|---|---|---|---|
| `drop_gap_db` | ×1.3 | 37.08 | 11/12 | 0 |

**RFI dependence.** Removing every RFI report leaves all classifications unchanged. Dev jamming confidence stays at 0.97, 0.97 and 0.88: the telemetry checks alone saturate the 0.97 cap, and the report only shifts the log-odds underneath it. TRN-05 never had a report. RFI reports are corroboration, not a dependency.

**Not covered by the sweep.** Secondary constants are written inline next to the logic that uses them: the
4 dB S4 acceptance band, the ±30 min S4 timing band, gap-bridging durations, the 0.02 peer link-loss cutoff, and
the confidence-check cutoffs in `evidence.py`. They are not part of the sweep. Of these, the S4 timing band is
stress-tested by the ±30 min S4 shift rows above.

**Antenna-fault margin.** The one threshold with limited headroom is `drop_gap_db` (15 dB). At 19.5 dB, the antenna fault's dropouts (weakest satellite 20–30 dB below the mean) start to fall under the bar and TRN-09 is missed. Lowering it is safe.

**Failure analysis.**

- **Starter baseline (7/12):** it misses TRN-05 (jamming inside the storm: its Kp ≥ 5 gate calls it space
  weather), TRN-08 (spoofing during the storm), TRN-10 (correction-link fault: its per-minute link share hovers around the baseline's 0.7 cutoff) and
  both multipath labels, and adds 5 false positives.
- **Farm-only ablation (11/12):** without S4, TRN-05's uniform 23 dB loss is absorbed into the "regional"
  level. It is missed, and only its two edges leak out as short false jamming fragments. This is the case the
  S4 fusion exists for.
- **Spoofing threshold:** before the storm-adaptive term was added, lowering the drift threshold by 30 %
  produced 19 false spoofing alerts, all inside the Kp 9 storm where every machine's XTE is about 1 m. The
  adaptive threshold removed them.

## Source audit: scoring the data feeds

Every RFI report is checked against the fleet's own telemetry over the full 72 hours (`python -m straightrows audit`):

| Report | Verdict | Evidence |
|---|---|---|
| RFI-001 | Corroborated | jamming 2026-05-04 16:30Z; emitter estimate 0.53 km from report centre, inside the ellipse |
| RFI-002 | Corroborated | jamming 2026-05-04 21:00Z; estimate 0.57 km from centre, inside |
| RFI-003 | Corroborated | jamming 2026-05-06 02:12Z; estimate 1.77 km from centre, inside |
| RFI-004 | Discounted | uniform loss explained by S4: the storm, not an emitter (the report itself says "may be storm-related") |
| (none) | Missed by feed | jamming 2026-05-05 19:59Z and 2026-05-06 18:30Z, each geolocated from telemetry alone |

**Feed trust: precision 0.75 (3 of 4 reports corroborated), recall 0.60 (3 of 5 detected jamming events reported).**
Run continuously, the same bookkeeping would give every provider a reliability score earned against independent
evidence.

## Every tractor is a sensor

The audit also geolocates each jamming episode again from every subset of the affected tractors. A **single
tractor**, just driving its rows, lands within **149 m** of the full-fleet estimate in all 14 tractor-episodes, and
inside the RFI report's ellipse in 8 of 8 reported cases. The row pattern alone supplies enough geometry. That
suggests every GNSS-guided machine already in the field is an RF-interference sensor that needs no new hardware.
These are synthetic data, generated with clean free-space propagation; with real-world noise, each additional
machine would become an independent cross-check rather than a redundant one.

## Multipath reporting choice

Every north-headland turn (about 162 per machine per working day) shows the same ~17 s reflection. Only one
pass per day is the scored "representative", and the official rules exempt recurrent multipath reports of ≤3
minutes from false-positive penalties. The submission therefore reports **each verified pass** as a separate
single-tractor `CONTINUE_FALSE_ALARM` with tight bounds: 486 in the evaluation window, 493 detections in total,
under the 500 limit. Operationally, these would be collapsed into a single "known reflector at the north
headland" notice, as the HTML report does.

## Limitations and failure modes

- **In-sample tuning.** Only 12 labelled episodes exist, with 1–3 per class. The robustness study measures margin,
  but a different simulator configuration (gradual jammer ramp, two spoofed machines at once, a storm without S4
  coverage) could break assumptions.
- **S4 response is fitted on one storm.** An S4 level never seen in development is linearly interpolated or
  extrapolated from neighbouring knots.
- **Spoofing relies on the simulated guidance monitor.** Reported position does not move in this data, and a real
  receiver does not know its own cross-track error. A real deployment needs an independent reference: IMU, wheel
  odometry, vision on row markers, or a second antenna.
- **Neighbour comparison assumes most machines are healthy.** If all three suffered identical equipment faults
  at once, the fault would look regional.
- **Geolocation assumes one stationary emitter and free-space loss.** A moving or directional jammer would fit
  poorly; the RMS is reported so a poor fit is visible.
- **Retrospective processing.** Several steps look ahead: centred smoothing (30 s), the multipath baseline
  (2 min after the turn), fault-rate and fleet-noise windows (~5 min), spoofing ramp fit and end search (4–10 min),
  the regional level (15 min) and the S4 acceptance band (±30 min). An online version would need causal windows
  and would trade onset accuracy for latency; the guide allows retrospective processing.

## Additional data that would improve attribution

- **Per-satellite C/N0, elevation and azimuth.** These would separate multipath (low-elevation, one azimuth) from
  antenna faults (all satellites), and spoofing (suspiciously equal C/N0 across satellites) from genuine signals.
- **Multi-frequency observables (L1/L2/L5).** Scintillation and jamming affect bands differently; most jammers
  target L1.
- **AGC level and noise floor** from the receiver: the most direct jamming indicator.
- **IMU or wheel odometry:** an independent position reference that turns spoofing detection from "suspicious
  drift" into "proven inconsistency".
- **Local ionosonde, ROTI or TEC maps** at finer cadence than 15 min, and the RTK base station's own observables:
  a degraded base points to the region, a healthy base points to the rover.
- **RFI reports with ellipse orientation and timestamps of the event**, not just of the report.

## Repository layout

```
straightrows/          the package (python -m straightrows ...)
  constants.py         classes, actions, windows and the 28 tuned detection thresholds
  data.py              feed loading
  signals.py           conditioning, S4 response fit, regional/excess decomposition
  detectors.py         one detector per mechanism
  localize.py          interference geolocation
  context.py           SDA lookups and RFI matching
  evidence.py          checks, confidence, evidence text, operational events
  pipeline.py          end-to-end analysis, submission writer, dev evaluation
  report.py            self-contained HTML operator report
  audit.py             RFI-feed trust score and single-tractor geolocation study
  robustness.py        assurance study
  official_scorer.py   official diagnostic scorer (vendored, unmodified)
  official_validator.py official submission validator (vendored, unmodified)
benchmarks/starter_baseline.py   official starter baseline (vendored, for comparison only)
scripts/get_data.py    dataset download + checksum
tests/                 synthetic unit tests and dataset regression tests
docs/DEMO_SCRIPT.md    five-minute demo plan and Q&A preparation
submission.json        the submitted results
submission.provenance.json  code and input hashes, thresholds, S4 response behind submission.json
.github/workflows/     CI: clean-machine download, tests, exact regeneration of submission.json
```

All challenge data is synthetic and unclassified. Code is released under the Apache License 2.0; vendored files
come from `Bluestaq/ndsu-hackathon-submissions` under the same license.
