"""Turn detector candidates into decisions: checks, confidence, evidence text.

Each class has a short list of explicit checks. A check either supports the
attribution (adds its weight to the log-odds) or argues against it (subtracts
its penalty). Corroboration that is known to be incomplete, such as RFI reports,
carries a small penalty when absent. Confidence is the logistic of the summed
log-odds, clipped to [0.05, 0.97] so no answer is ever presented as certain.
"""
import math
from dataclasses import dataclass, field

import pandas as pd

from . import context as ctx
from .localize import localize
from .constants import ACTION, THRESHOLDS, TRACTORS


@dataclass
class Check:
    name: str
    passed: bool
    weight: float
    penalty: float
    text: str


@dataclass
class Detection:
    cls: str
    start: pd.Timestamp
    end: pd.Timestamp
    tractors: list
    checks: list
    guidance: str
    summary: str
    metrics: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    prior: float = 0.0

    @property
    def action(self):
        return ACTION[self.cls]

    @property
    def confidence(self):
        logit = self.prior + sum(c.weight if c.passed else -c.penalty for c in self.checks)
        return round(min(0.97, max(0.05, 1 / (1 + math.exp(-logit)))), 2)

    def evidence(self):
        lines = [self.summary]
        lines += [("SUPPORTS: " if c.passed else "AGAINST: ") + c.text for c in self.checks]
        lines += self.notes
        lines.append("OPERATOR: " + self.guidance)
        return lines

    def to_json(self):
        return {
            "start": self.start.isoformat(), "end": self.end.isoformat(),
            "tractors": list(self.tractors), "class": self.cls, "action": self.action,
            "confidence": self.confidence, "evidence": self.evidence(),
        }


def hm(ts):
    return ts.strftime("%H:%M:%SZ") if ts.second else ts.strftime("%H:%MZ")


def stamp(ts):
    return ts.strftime("%Y-%m-%d ") + hm(ts)


def span(c):
    return f"{stamp(c.start)} to {hm(c.end)}"


def names(tractors):
    return ",".join(tractors)


# --------------------------------------------------------------------------- per class
def space_weather(c, ds, sig):
    m, sda = c.metrics, ctx.space_context(ds, c.start, c.end)
    max_excess = max((abs(v) for v in m["excess_db"].values()), default=0.0)
    rtk = sig.rtk.loc[c.start:c.end, c.tractors].mean().mean()
    xte = sig.xte.loc[c.start:c.end, c.tractors].mean().mean()
    advisories = ctx.advisories_in_effect(ds, c.start, c.end, "SPACE_WEATHER")
    rfi = ctx.rfi_nearby(ds, c.start, c.end)
    checks = [
        Check("uniform", m["spread_db"] < 1.0, 1.0, 1.0,
              f"C/N0 fell together on {names(c.tractors)} to {45 - m['regional_db']:.1f} dB-Hz "
              f"({m['regional_db']:.1f} dB below nominal 45); median fleet spread {m['spread_db']:.2f} dB, "
              "so the loss is regional, not machine- or location-specific"),
        Check("s4_explains", abs(m["regional_db"] - m["s4_expected_db"]) < 3.0, 1.2, 1.0,
              f"Regional S4 {sda['s4_at_onset']:.2f} at {hm(sda['s4_at_onset_ts'])} (mean {sda['s4_mean']:.2f} during) "
              f"predicts {m['s4_expected_db']:.1f} dB loss from the dev-fitted S4 response; observed {m['regional_db']:.1f} dB"),
        Check("kp", sda["kp_max_6h"] >= THRESHOLDS["kp_storm"], 0.6, 0.3,
              f"Kp peaked at {sda['kp_max_6h']:.1f} at {hm(sda['kp_max_ts'])} within 6 h before/during "
              f"(G{min(5, max(0, int(sda['kp_max_6h']) - 4))} storm level); Dst minimum {sda['dst_min']:.0f} nT"),
        Check("no_local_excess", max_excess < 2.0, 0.6, 0.6,
              f"Every tractor stayed within {max_excess:.1f} dB of the regional level, ruling out a local jammer "
              "or a single faulty antenna as the main cause"),
        Check("impact", rtk < 0.5, 0.3, 0.3,
              f"Guidance impact: RTK_FIXED share {rtk:.0%}, mean cross-track error {xte:.2f} m"),
    ]
    if advisories:
        a = advisories[-1]
        checks.append(Check("advisory", True, 0.3, 0.0, f"{a['id']} ({stamp(a['ts'])}): {a['text']}"))
    notes = []
    if c.metrics["degradation_onset"] < c.start:
        notes.append(f"NOTE: regional degradation began at {hm(c.metrics['degradation_onset'])} while machines were "
                     f"parked; the scored guidance episode starts when work resumed at {hm(c.start)}.")
    for r in rfi:
        notes.append(f"NOTE: {r['id']} ({stamp(r['ts'])}, confidence {r['confidence']}) is nearby in time; "
                     "a uniform, S4-consistent loss is better explained by scintillation than by a local emitter.")
    return Detection(
        "SPACE_WEATHER", c.start, c.end, c.tractors, checks, prior=0.2,
        summary=f"SPACE_WEATHER on {names(c.tractors)}, {span(c)}: regional scintillation degraded GNSS for the whole fleet.",
        guidance=("Halt precision work or drop to degraded mode (wider pass-to-pass tolerance, no RTK-dependent "
                  "seeding/spraying) for all machines; resume when C/N0 recovers above 42 dB-Hz and S4 falls below 0.3."),
        metrics={**m, **sda}, notes=notes,
    )


def jamming(c, ds, sig):
    m, sda = c.metrics, ctx.space_context(ds, c.start, c.end)
    affected = {t: m["excess_db"][t] for t in c.tractors}
    top = max(affected.values())
    step = max(m["step_db"][t] for t in c.tractors)
    storm = m["regional_db"] > 3.0
    matches = ctx.rfi_matches(ds, c.start, c.end, c.tractors)
    excluded = [t for t in TRACTORS if t not in c.tractors]
    checks = [
        Check("excess", top >= 5.0, 1.2, 1.0,
              "C/N0 loss beyond what regional S4 explains: "
              + ", ".join(f"{t} {v:.1f} dB" for t, v in affected.items())),
        Check("abrupt", step >= 5.0, 0.8, 0.3,
              f"Abrupt onset at {hm(c.start)}: up to {step:.1f} dB extra loss within 60 s "
              "(scintillation builds with S4; a switched-on emitter does not)"),
    ]
    if storm:
        checks.append(Check("space_quiet", False, 0.6, 0.4,
                            f"A storm is in progress (S4 {sda['s4_at_onset']:.2f}, regional loss {m['regional_db']:.1f} dB), "
                            f"but the loss exceeds the storm level by {top:.1f} dB; reported as a separate concurrent cause"))
    else:
        checks.append(Check("space_quiet", sda["s4_at_onset"] < THRESHOLDS["s4_corroborate"] and sda["kp_at_onset"] < THRESHOLDS["kp_storm"], 0.6, 0.4,
                            f"Space weather quiet: S4 {sda['s4_at_onset']:.2f} at {hm(sda['s4_at_onset_ts'])}, "
                            f"Kp {sda['kp_at_onset']:.1f}; scintillation cannot explain the loss"))
    if m["spread_db"] >= 2.0:
        checks.append(Check("differential", True, 0.4, 0.0,
                            f"Loss differs by {m['spread_db']:.1f} dB between machines, consistent with distance "
                            "to a local emitter"))
    if matches:
        r = max(matches, key=lambda r: r["confidence"])
        checks.append(Check("rfi", True, 1.0, 0.0,
                            f"{r['id']} at {hm(r['ts'])} ({r['delay_min']:.0f} min after onset, within the 10-30 min "
                            f"reporting delay): L1 interference, report confidence {r['confidence']}, ellipse centre "
                            f"{r['distance_km']:.1f} km from the fleet (semi-major {r['semi_major_km']:.1f} km"
                            f"{', covers the fleet' if r['covers_fleet'] else ''})"))
    else:
        checks.append(Check("rfi", False, 1.0, 0.2,
                            "No RFI report matches; reports are incomplete, so attribution rests on the telemetry"))
    loc = localize(ds, sig, c.start, c.end, c.tractors)
    where = ""
    if loc and loc["rms_db"] < 1.5 and not loc["at_search_edge"]:
        where = f" (estimated source {loc['lat']:.4f}, {loc['lon']:.4f} +/- {loc['uncertainty_km'] * 1000:.0f} m)"
        agree = ""
        if matches:
            r = max(matches, key=lambda r: r["confidence"])
            rep = next(x for x in ds.rfi if x["id"] == r["id"])
            d = ctx.distance_km(loc["lat"], loc["lon"], rep["centerLat"], rep["centerLon"])
            inside = d <= rep["semiMajor_km"]
            agree = (f"; independently, it lies {d:.1f} km from the {r['id']} ellipse centre"
                     f"{' (inside the ellipse)' if inside else ' (outside the ellipse)'}")
        checks.append(Check("geolocated", True, 0.6, 0.0,
                            f"Loss varies with each machine's range as it drives its rows: a free-space point-emitter "
                            f"model (20 dB/decade) fits {loc['samples']} samples with {loc['rms_db']:.2f} dB RMS, placing "
                            f"the source at {loc['lat']:.4f}, {loc['lon']:.4f} (+/- {loc['uncertainty_km'] * 1000:.0f} m), "
                            f"{loc['range_km']:.1f} km at bearing {loc['bearing_deg']:.0f} deg from the field centre{agree}"))
    notes = [
        "IMPACT: " + ", ".join(
            f"{t} RTK_FIXED {m['rtk_before'][t]:.0%}->{m['rtk_during'][t]:.0%}, XTE {m['xte_m'][t]:.2f} m"
            for t in c.tractors)
    ]
    for t in excluded:
        notes.append(f"NOTE: {t} lost only {m['excess_db'][t]:.1f} dB and kept RTK_FIXED "
                     f"{m['rtk_during'][t]:.0%}, so it is not listed as affected.")
    return Detection(
        "JAMMING", c.start, c.end, c.tractors, checks, prior=0.0,
        summary=f"JAMMING on {names(c.tractors)}, {span(c)}: RF interference beyond what space weather explains.",
        guidance=(f"Halt GNSS-guided work on {names(c.tractors)}; report the interference{where} with times to the "
                  "regional RFI monitor/authorities; resume after C/N0 is back to nominal for 10 minutes."),
        metrics={**m, **sda, "rfi": matches, "localization": loc}, notes=notes,
    )


def spoofing(c, ds, sig):
    m, sda = c.metrics, ctx.space_context(ds, c.start, c.end)
    t = c.tractors[0]
    rate = m["ramp_m_per_min"]
    rate_txt = f"at {rate:.2f} m/min " if rate == rate else ""
    checks = [
        Check("drift", m["max_resid_m"] >= 1.0, 1.2, 1.0,
              f"Guidance-monitor cross-track error on {t} drifted {rate_txt}from {hm(c.start)} to "
              f"{m['max_resid_m']:.1f} m above its neighbours (peers averaged {m['peer_xte_m']:.2f} m)"),
        Check("healthy_signal", abs(m["cn0_vs_peers_db"]) < 1.5 and m["own_excess_db"] < 1.5, 1.0, 1.0,
              f"No signal-health collapse: {t} C/N0 within {abs(m['cn0_vs_peers_db']):.1f} dB of its neighbours "
              f"(rules out jamming and antenna faults)"),
        Check("confidently_wrong", m["rtk_share"] >= m["peer_rtk_share"] - 0.1, 0.6, 0.3,
              f"Receiver kept reporting fix quality as good as its neighbours (RTK_FIXED {m['rtk_share']:.0%} vs "
              f"{m['peer_rtk_share']:.0%}) while drifting off-line: it trusts a false solution"),
        Check("sustained", c.minutes >= 10, 0.3, 0.3,
              f"Lasted {c.minutes:.0f} min, far longer than a headland reflection"),
    ]
    if m["regional_db"] > 3.0:
        checks.append(Check("storm_free", False, 0.0, 0.3,
                            f"Concurrent storm (regional loss {m['regional_db']:.1f} dB, S4 {sda['s4_at_onset']:.2f}) "
                            "raises everyone's XTE; the drift is specific to this machine, so reported separately"))
    return Detection(
        "SPOOFING", c.start, c.end, c.tractors, checks, prior=0.0,
        summary=f"SPOOFING on {t}, {span(c)}: position/guidance drifted without the signal loss jamming would cause.",
        guidance=(f"Halt {t}; cross-check its position against an independent reference (inertial/odometry, "
                  "RTK base baseline, visible row marks) and report suspected spoofing; other machines may continue "
                  "while their cross-track error stays nominal."),
        metrics={**m, **sda},
    )


def equipment(c, ds, sig):
    m, sda = c.metrics, ctx.space_context(ds, c.start, c.end)
    t = c.tractors[0]
    if m["kind"] == "correction_link":
        checks = [
            Check("link", m["link_loss_share"] >= 0.1, 1.2, 1.0,
                  f"RTK correction link on {t} dropped {m['link_loss_share']:.0%} of the time from {hm(c.start)}; "
                  f"neighbours {m['peer_link_loss_share']:.0%}"),
            Check("machine_specific", m["peer_rtk_share"] >= 0.9, 0.8, 0.8,
                  f"{t} RTK_FIXED share {m['rtk_share']:.0%} vs neighbours {m['peer_rtk_share']:.0%}; "
                  f"XTE {m['xte_m']:.2f} m vs {m['peer_xte_m']:.2f} m"),
            Check("signal_ok", abs(m["excess_db"]) < 1.5, 0.6, 0.6,
                  f"Satellite signal strength on {t} normal ({m['excess_db']:+.1f} dB vs regional level): "
                  "not interference or space weather"),
        ]
        what, fix = "correction-link fault", "check the RTK radio/modem, cabling and base-station subscription"
    else:
        checks = [
            Check("dropouts", m["dropout_share"] >= 0.1, 1.2, 1.0,
                  f"Intermittent deep dropouts on {t}: weakest-satellite C/N0 fell >15 dB below the mean in "
                  f"{m['dropout_share']:.0%} of seconds (neighbours {m['peer_dropout_share']:.0%}); 10th-percentile "
                  f"minimum C/N0 {m['min_cn0_p10']:.1f} vs {m['peer_min_cn0_p10']:.1f} dB-Hz"),
            Check("machine_specific", m["peer_dropout_share"] < 0.05, 0.8, 0.8,
                  f"Neighbouring machines in the same field were unaffected; satellites used {m['sats']:.1f} "
                  f"vs {m['peer_sats']:.1f}"),
            Check("signal_ok", abs(m["excess_db"]) < 2.0, 0.6, 0.6,
                  f"Mean C/N0 on {t} only {m['excess_db']:+.1f} dB from the regional level: not jamming or "
                  "scintillation, which lower every satellite"),
        ]
        what, fix = "antenna/cable fault", "inspect the GNSS antenna, connector and cable"
    if sda["s4_mean"] >= THRESHOLDS["s4_corroborate"]:
        checks.append(Check("storm_free", False, 0.0, 0.3,
                            f"Concurrent scintillation (S4 {sda['s4_mean']:.2f}); fault judged relative to neighbours"))
    return Detection(
        "EQUIPMENT_FAULT", c.start, c.end, c.tractors, checks, prior=0.2,
        summary=f"EQUIPMENT_FAULT ({what}) on {t}, {span(c)}: only this machine is degraded.",
        guidance=f"Service {t}: {fix}. Other machines continue normally.",
        metrics={**m, **sda},
    )


def multipath(c, ds, sig, recurrences):
    m = c.metrics
    t = c.tractors[0]
    north_m = (m["north_lat"] - m["lat"]) * ctx.KM_PER_DEG_LAT * 1000
    checks = [
        Check("headland", north_m < 10, 0.8, 0.8,
              f"{t} was turning at the north headland ({m['lat']:.5f}, {m['lon']:.5f}; {north_m:.0f} m from the "
              f"field edge) at {hm(m['turn'])}"),
        Check("brief", c.minutes <= 3, 0.5, 0.5,
              f"Weakest-satellite C/N0 dipped {m['gap_rise_db']:.1f} dB (to {m['min_cn0']:.1f} dB-Hz) for "
              f"{(c.end - c.start).total_seconds():.0f} s only"),
        Check("recurring", recurrences >= 3, 0.6, 0.3,
              f"The same signature recurs at this headland on {recurrences} turns by {t} in the analysed window"),
        Check("harmless", m["xte_peak_m"] < 0.5, 0.4, 0.4,
              f"Guidance barely moved: peak XTE {m['xte_peak_m']:.2f} m"),
    ]
    return Detection(
        "MULTIPATH", c.start, c.end, c.tractors, checks, prior=0.0,
        summary=f"MULTIPATH on {t}, {stamp(c.start)} to {hm(c.end)}: brief reflection at the north headland obstruction.",
        guidance="Continue: false alarm from a known reflector. Consider suppressing alerts in this headland zone.",
        metrics=m,
    )


def build(candidates, ds, sig):
    mp = [c for c in candidates if c.cls == "MULTIPATH"]
    per_tractor = {t: sum(1 for c in mp if c.tractors == [t]) for t in TRACTORS}
    out = []
    for c in candidates:
        if c.cls == "MULTIPATH":
            out.append(multipath(c, ds, sig, per_tractor[c.tractors[0]]))
        else:
            out.append({"SPACE_WEATHER": space_weather, "JAMMING": jamming, "SPOOFING": spoofing,
                        "EQUIPMENT_FAULT": equipment}[c.cls](c, ds, sig))
    link_operational_events(out)
    return out


def link_operational_events(detections):
    """Group overlapping non-multipath detections into operational events."""
    major = sorted((d for d in detections if d.cls != "MULTIPATH"), key=lambda d: d.start)
    groups = []
    for d in major:
        if groups and d.start < max(x.end for x in groups[-1]):
            groups[-1].append(d)
        else:
            groups.append([d])
    for i, group in enumerate(groups, 1):
        if len(group) < 2:
            continue
        for d in group:
            others = "; ".join(f"{o.cls} on {names(o.tractors)} {hm(o.start)}-{hm(o.end)}" for o in group if o is not d)
            d.notes.append(f"OPERATIONAL EVENT OE-{i:02d}: concurrent with {others}. Distinct causes are reported "
                           "separately; apply the most restrictive action to each affected machine.")
