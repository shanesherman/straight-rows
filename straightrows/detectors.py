"""One detector per physical mechanism.

Each detector returns Candidate objects: an interval, the affected tractors,
and the raw measurements that later become checkable evidence. Detectors are
deliberately independent so causes can overlap (a fault during a storm, a
spoof during a storm, jamming during a storm).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .constants import TRACTORS

SECOND = pd.Timedelta(seconds=1)


@dataclass
class Candidate:
    cls: str
    start: pd.Timestamp
    end: pd.Timestamp
    tractors: list
    metrics: dict = field(default_factory=dict)

    @property
    def minutes(self):
        return (self.end - self.start).total_seconds() / 60


def intervals(mask: pd.Series, bridge_s=0, min_s=0):
    """Contiguous True runs of a 1 Hz boolean series as [start, end) pairs."""
    values = mask.fillna(False).to_numpy(dtype=bool)
    if not values.any():
        return []
    edges = np.diff(np.concatenate([[0], values.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    runs = []
    for s, e in zip(starts, ends):
        if runs and s - runs[-1][1] <= bridge_s:
            runs[-1][1] = e
        else:
            runs.append([s, e])
    index = mask.index
    out = []
    for s, e in runs:
        if e - s >= min_s:
            out.append((index[s], index[e - 1] + SECOND))
    return out


def _window(frame, start, end):
    return frame.loc[start:end - SECOND]


# --------------------------------------------------------------------------- space weather
def space_weather(sig, ds, th):
    any_working = sig.working.any(axis=1)
    depressed = sig.regional > th["sw_depression_db"]
    out = []
    for start, end in intervals(depressed & any_working, bridge_s=600, min_s=th["sw_min_minutes"] * 60):
        work = _window(sig.working, start, end).mean()
        tractors = [t for t in TRACTORS if work[t] >= 0.5]
        began = sig.regional.loc[:start - SECOND]
        began = began[began <= th["sw_depression_db"]]
        degradation_onset = began.index[-1] + SECOND if len(began) else start
        out.append(Candidate("SPACE_WEATHER", start, end, tractors, {
            "regional_db": float(_window(sig.regional, start, end).mean()),
            "s4_expected_db": float(_window(sig.s4_expected, start, end).mean()),
            "spread_db": float((_window(sig.cn0, start, end).max(axis=1) - _window(sig.cn0, start, end).min(axis=1)).median()),
            "excess_db": {t: float(_window(sig.excess[t], start, end).mean()) for t in tractors},
            "degradation_onset": degradation_onset,
        }))
    return out


# --------------------------------------------------------------------------- jamming
def jamming(sig, ds, th):
    flags = (sig.excess > th["jam_excess_db"]) & sig.working
    out = []
    for start, end in intervals(flags.any(axis=1), bridge_s=th["jam_bridge_s"], min_s=th["jam_min_minutes"] * 60):
        excess = _window(sig.excess, start, end).mean()
        rtk_during = _window(sig.rtk, start, end).mean()
        rtk_before = _window(sig.rtk, start - pd.Timedelta("30min"), start).mean()
        tractors = [
            t for t in TRACTORS
            if excess[t] >= th["jam_member_db"]
            or (rtk_during[t] < th["jam_member_rtk"] and rtk_before[t] >= 0.95)
        ]
        if not tractors:
            tractors = [str(excess.idxmax())]
        first = _window(sig.excess, start, start + pd.Timedelta("60s")).mean()
        prior = _window(sig.excess, start - pd.Timedelta("120s"), start - pd.Timedelta("30s")).mean()
        out.append(Candidate("JAMMING", start, end, tractors, {
            "excess_db": {t: float(excess[t]) for t in TRACTORS},
            "step_db": {t: float(first[t] - prior[t]) for t in TRACTORS},
            "rtk_during": {t: float(rtk_during[t]) for t in TRACTORS},
            "rtk_before": {t: float(rtk_before[t]) for t in TRACTORS},
            "regional_db": float(_window(sig.regional, start, end).mean()),
            "spread_db": float(excess.max() - excess.min()),
            "xte_m": {t: float(_window(sig.xte[t], start, end).mean()) for t in TRACTORS},
        }))
    return out


# --------------------------------------------------------------------------- spoofing
def _ramp_onset(resid: pd.Series, first: pd.Timestamp):
    """Back-extrapolate a linear guidance-error ramp to where it left zero."""
    seg = resid.loc[first - pd.Timedelta("10min"):first + pd.Timedelta("4min")]
    fit = seg[(seg > 0.08) & (seg < 3.0)]
    fit = fit.loc[first - pd.Timedelta("6min"):]
    if len(fit) < 60:
        return first, float("nan")
    x = (fit.index - first).total_seconds().to_numpy()
    slope, intercept = np.polyfit(x, fit.to_numpy(), 1)
    if slope <= 0:
        return first, float("nan")
    onset = first + pd.Timedelta(seconds=float(-intercept / slope))
    onset = min(max(onset, first - pd.Timedelta("10min")), first)
    return onset.floor("s"), float(slope * 60)


def spoofing(sig, ds, th):
    out = []
    # Storms raise every machine's guidance noise; demand more drift when the fleet is noisy.
    fleet_noise = sig.xte.median(axis=1).rolling(601, center=True, min_periods=1).median()
    for t in TRACTORS:
        resid = sig.xte_resid[t]
        healthy = sig.excess[t] < th["spoof_max_excess_db"]
        flag = (resid > th["spoof_resid_m"] + fleet_noise) & sig.working[t] & healthy
        for start, end in intervals(flag, bridge_s=60, min_s=th["spoof_min_minutes"] * 60):
            onset, rate = _ramp_onset(resid, start)
            tail = resid.loc[end:end + pd.Timedelta("10min")]
            done = tail[tail <= th["spoof_end_m"]]
            finish = done.index[0] if len(done) else end
            others = [o for o in TRACTORS if o != t]
            window = slice(onset, finish - SECOND)
            out.append(Candidate("SPOOFING", onset, finish, [t], {
                "max_resid_m": float(resid.loc[window].max()),
                "ramp_m_per_min": rate,
                "own_excess_db": float(sig.excess[t].loc[window].mean()),
                "cn0_vs_peers_db": float((sig.cn0[t] - sig.cn0[others].mean(axis=1)).loc[window].mean()),
                "rtk_share": float(sig.rtk[t].loc[window].mean()),
                "peer_rtk_share": float(sig.rtk[others].loc[window].mean().mean()),
                "peer_xte_m": float(sig.xte[others].loc[window].mean().mean()),
                "regional_db": float(sig.regional.loc[window].mean()),
            }))
    return out


# --------------------------------------------------------------------------- equipment faults
def _refine(raw: pd.Series, start, end, pad="5min"):
    """Snap a smoothed interval to its first and last raw event."""
    pad = pd.Timedelta(pad)
    events = raw.loc[start - pad:end + pad]
    events = events[events > 0]
    if events.empty:
        return start, end
    return events.index[0], events.index[-1] + SECOND


def equipment(sig, ds, th, multipath_mask=None):
    out = []
    win_l = th["link_window_s"]
    link_rate = sig.link_bad.rolling(win_l, center=True, min_periods=1).mean()
    drops = (sig.gap > th["drop_gap_db"]).astype(float)
    if multipath_mask is not None:
        drops = drops.where(~multipath_mask, 0.0)
    drop_rate = drops.rolling(th["drop_window_s"], center=True, min_periods=1).mean()
    for t in TRACTORS:
        others = [o for o in TRACTORS if o != t]
        # correction-link fault: this machine loses corrections, neighbours do not
        flag = (link_rate[t] > th["link_bad_frac"]) & (link_rate[others].max(axis=1) < 0.02) & sig.working[t]
        for start, end in intervals(flag, bridge_s=300, min_s=th["link_min_minutes"] * 60):
            s, e = _refine(sig.link_bad[t], start, end)
            w = slice(s, e - SECOND)
            out.append(Candidate("EQUIPMENT_FAULT", s, e, [t], {
                "kind": "correction_link",
                "link_loss_share": float(sig.link_bad[t].loc[w].mean()),
                "peer_link_loss_share": float(sig.link_bad[others].loc[w].mean().mean()),
                "rtk_share": float(sig.rtk[t].loc[w].mean()),
                "peer_rtk_share": float(sig.rtk[others].loc[w].mean().mean()),
                "xte_m": float(sig.xte[t].loc[w].mean()),
                "peer_xte_m": float(sig.xte[others].loc[w].mean().mean()),
                "excess_db": float(sig.excess[t].loc[w].mean()),
            }))
        # antenna/cable fault: intermittent deep dropouts on one machine only
        flag = ((drop_rate[t] - drop_rate[others].max(axis=1)) > th["drop_rate"]) & sig.working[t]
        for start, end in intervals(flag, bridge_s=300, min_s=th["fault_min_minutes"] * 60):
            s, e = _refine(drops[t], start, end, pad="3min")
            w = slice(s, e - SECOND)
            sats = ds.wide["sats"]
            out.append(Candidate("EQUIPMENT_FAULT", s, e, [t], {
                "kind": "antenna",
                "dropout_share": float(drops[t].loc[w].mean()),
                "peer_dropout_share": float(drops[others].loc[w].mean().mean()),
                "min_cn0_p10": float(ds.wide["min_cn0"][t].loc[w].quantile(0.1)),
                "peer_min_cn0_p10": float(ds.wide["min_cn0"][others].loc[w].quantile(0.1).mean()),
                "sats": float(sats[t].loc[w].mean()),
                "peer_sats": float(sats[others].loc[w].mean().mean()),
                "excess_db": float(sig.excess[t].loc[w].mean()),
            }))
    return out


# --------------------------------------------------------------------------- multipath
def headland_turns(ds, th):
    """Seconds at which each working tractor reverses at the north headland."""
    lat, speed = ds.wide["lat"], ds.wide["speed"]
    north = float(lat.max().max())
    turns = {}
    for t in TRACTORS:
        near = (lat[t] > north - th["mp_headland_margin_deg"]) & (speed[t] > th["working_speed_mps"])
        times = []
        for start, end in intervals(near):
            seg = lat[t].loc[start:end - SECOND]
            times.append(seg.idxmax())
        turns[t] = times
    return turns, north


def multipath(sig, ds, th):
    turns, north = headland_turns(ds, th)
    pre, post = pd.Timedelta(seconds=th["mp_pre_s"]), pd.Timedelta(seconds=th["mp_post_s"])
    out, mask = [], pd.DataFrame(False, index=sig.index, columns=list(TRACTORS))
    for t in TRACTORS:
        gap = sig.gap[t]
        for k, turn in enumerate(turns[t]):
            s, e = turn - pre, turn + post
            inside = gap.loc[s:e - SECOND]
            before = gap.loc[turn - pd.Timedelta("120s"):turn - pd.Timedelta("30s")]
            after = gap.loc[turn + pd.Timedelta("30s"):turn + pd.Timedelta("120s")]
            base = pd.concat([before, after]).median()
            rise = float(inside.median() - base)
            mask.loc[s:e - SECOND, t] = True
            if rise < th["mp_gap_rise_db"]:
                continue
            xte = ds.wide["xte"][t]
            out.append(Candidate("MULTIPATH", s, e, [t], {
                "gap_rise_db": rise,
                "min_cn0": float(ds.wide["min_cn0"][t].loc[s:e - SECOND].min()),
                "base_min_cn0": float(ds.wide["min_cn0"][t].loc[turn - pd.Timedelta("120s"):turn - pd.Timedelta("30s")].median()),
                "xte_peak_m": float(xte.loc[s:e - SECOND].max()),
                "lat": float(ds.wide["lat"][t].loc[turn]),
                "lon": float(ds.wide["lon"][t].loc[turn]),
                "turn": turn, "north_lat": north,
                "fix": float(sig.rtk[t].loc[s:e - SECOND].mean()),
            }))
    return out, mask
