"""Identifiers, windows and tuned thresholds.

Every threshold below was chosen by inspecting the 48 h development window
(hours 0-48, 12 labelled episodes) and is exercised by tests/test_dev_eval.py.
The sensitivity study in straightrows/robustness.py perturbs each of them.
"""
import pandas as pd

TRACTORS = ("T1", "T2", "T3")
ACTION = {
    "SPACE_WEATHER": "HALT_OR_DEGRADED_MODE",
    "JAMMING": "HALT_AND_REPORT",
    "SPOOFING": "HALT_CROSSCHECK_AND_REPORT",
    "EQUIPMENT_FAULT": "SERVICE_MACHINE_OTHERS_CONTINUE",
    "MULTIPATH": "CONTINUE_FALSE_ALARM",
}
CLASSES = tuple(ACTION)

DEV_START = pd.Timestamp("2026-05-04T00:00:00Z")
DEV_END = pd.Timestamp("2026-05-06T00:00:00Z")
EVAL_START = pd.Timestamp("2026-05-06T00:00:00Z")
EVAL_END = pd.Timestamp("2026-05-07T00:00:00Z")

# Receiver nominal mean C/N0 in clear sky (median of unlabelled working hours).
NOMINAL_CN0 = 45.0

THRESHOLDS = {
    # --- shared signal conditioning
    "smooth_s": 61,                 # centred rolling-median window (s); preserves step edges
    "working_speed_mps": 1.0,       # a tractor is "working" above this speed
    # --- regional (space-weather) common mode
    "uniform_spread_db": 1.0,       # fleet C/N0 spread below this = spatially uniform
    "regional_window_s": 1801,      # rolling median of the uniform level (s)
    "sw_depression_db": 3.0,        # regional C/N0 depression that degrades guidance
    "sw_min_minutes": 20,           # storms are long; shorter dips are not regional
    "s4_corroborate": 0.25,         # S4 at or above this corroborates scintillation
    "kp_storm": 5.0,                # Kp >= 5 is a G1+ geomagnetic storm
    # --- jamming: excess loss on top of the regional level
    "jam_excess_db": 3.0,
    "jam_min_minutes": 3,
    "jam_bridge_s": 120,
    "jam_member_db": 4.5,           # tractor counted as affected above this mean excess
    "jam_member_rtk": 0.85,         # ...or if its RTK_FIXED share falls below this
    # --- spoofing: guidance error without signal-health collapse
    "spoof_resid_m": 0.5,
    "spoof_end_m": 0.3,
    "spoof_min_minutes": 3,
    "spoof_max_excess_db": 2.0,     # the tractor's own C/N0 must look healthy
    # --- equipment faults: one machine differs from its neighbours
    "link_window_s": 301,
    "link_bad_frac": 0.10,          # >=10 % correction-link loss on one tractor
    "link_min_minutes": 5,
    "drop_gap_db": 15.0,            # mean-minus-min C/N0 gap that marks an antenna dropout
    "drop_window_s": 301,
    "drop_rate": 0.10,              # dropout share above peers that marks a faulty antenna
    "fault_min_minutes": 5,
    # --- multipath: brief reflections at the north headland obstruction
    "mp_pre_s": 8,
    "mp_post_s": 9,
    "mp_gap_rise_db": 4.0,
    "mp_headland_margin_deg": 0.00005,
}
