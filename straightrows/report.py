"""Self-contained HTML operator report: timeline, per-episode evidence and charts."""
import base64
import html
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .constants import NOMINAL_CN0, TRACTORS  # noqa: E402

TRACTOR_COLOR = {"T1": "#2a78d6", "T2": "#eb6834", "T3": "#1baf7a"}
CLASS_COLOR = {"SPACE_WEATHER": "#6250d6", "JAMMING": "#e34948", "SPOOFING": "#eda100",
               "EQUIPMENT_FAULT": "#008300", "MULTIPATH": "#e87ba4"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "legend.frameon": False,
})


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _fmt_axis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz="UTC"))


def timeline(ds, sig, decisions, start, end):
    fig, axes = plt.subplots(3, 1, figsize=(11, 6.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 1.6, 0.9]})
    ax = axes[0]
    rows = {t: i for i, t in enumerate(reversed(TRACTORS))}
    for d in decisions:
        for t in d.tractors:
            y = rows[t]
            if d.cls == "MULTIPATH":
                ax.plot([d.start, d.start], [y - 0.42, y - 0.28], color=CLASS_COLOR[d.cls], lw=0.8)
                continue
            ax.barh(y, d.end - d.start, left=d.start, height=0.5, color=CLASS_COLOR[d.cls],
                    edgecolor="#fcfcfb", linewidth=2)
    ax.set_yticks(list(rows.values()), list(rows.keys()))
    ax.set_ylim(-0.6, 2.6)
    ax.set_title("Detections by tractor (ticks under each row = per-pass headland multipath)", loc="left", color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CLASS_COLOR.values()]
    ax.legend(handles, list(CLASS_COLOR), ncol=5, loc="upper left", bbox_to_anchor=(0, -0.08), fontsize=8)
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    cn0 = ds.wide["cn0"].loc[start:end].resample("2min").mean()
    for t in TRACTORS:
        ax.plot(cn0.index, cn0[t], color=TRACTOR_COLOR[t], label=t)
    reg = (NOMINAL_CN0 - sig.regional.loc[start:end]).resample("2min").mean()
    ax.plot(reg.index, reg, color=MUTED, lw=1.2, ls="--", label="S4-consistent regional level")
    ax.set_ylabel("mean C/N0 (dB-Hz)")
    ax.legend(ncol=4, loc="lower left", fontsize=8)
    ax.set_title("Fleet signal strength", loc="left", color=INK)

    ax = axes[2]
    s4 = ds.s4.loc[start - pd.Timedelta("15min"):end]
    ax.step(s4.index, s4, where="post", color="#6250d6", label="S4 (regional scintillation)")
    ax.axhline(0.25, color=MUTED, lw=0.8, ls=":")
    ax.set_ylabel("S4")
    kp = ds.kp.loc[start - pd.Timedelta("3h"):end]
    for ts, v in kp.items():
        if ts >= start:
            ax.annotate(f"Kp {v:.1f}", (ts, 0.85), xycoords=("data", "axes fraction"),
                        fontsize=7, color=MUTED)
    ax.set_title("Space weather context (Kp annotated every 3 h)", loc="left", color=INK)
    _fmt_axis(ax)
    ax.set_xlim(start, end)
    fig.tight_layout()
    return _png(fig)


def episode_figure(ds, sig, d):
    pad = pd.Timedelta("30min") if d.cls != "MULTIPATH" else pd.Timedelta("90s")
    a, b = d.start - pad, d.end + pad
    freq = "30s" if d.cls != "MULTIPATH" else "1s"
    third = d.cls == "EQUIPMENT_FAULT"
    fig, axes = plt.subplots(3 if third else 2, 1, figsize=(10, 5.6 if third else 4.2), sharex=True)
    ax = axes[0]
    channel = "min_cn0" if d.cls == "MULTIPATH" else "cn0"
    series = ds.wide[channel].loc[a:b].resample(freq).mean()
    for t in TRACTORS:
        lw = 2.0 if t in d.tractors else 1.0
        ax.plot(series.index, series[t], color=TRACTOR_COLOR[t], lw=lw, label=t + (" (affected)" if t in d.tractors else ""))
    if d.cls != "MULTIPATH":
        reg = (NOMINAL_CN0 - sig.regional.loc[a:b]).resample(freq).mean()
        ax.plot(reg.index, reg, color=MUTED, ls="--", lw=1.2, label="regional level")
    ax.set_ylabel("min C/N0" if d.cls == "MULTIPATH" else "mean C/N0 (dB-Hz)")
    ax.legend(ncol=4, fontsize=8, loc="lower left")
    ax = axes[1]
    xte = ds.wide["xte"].loc[a:b].resample(freq).mean()
    for t in TRACTORS:
        ax.plot(xte.index, xte[t], color=TRACTOR_COLOR[t], lw=2.0 if t in d.tractors else 1.0)
    ax.set_ylabel("cross-track error (m)")
    if third:
        ax = axes[2]
        if d.metrics.get("kind") == "correction_link":
            v = sig.link_bad.loc[a:b].resample("1min").mean()
            ax.set_ylabel("link loss share / min")
        else:
            v = (sig.gap.loc[a:b] > 15).resample("1min").mean()
            ax.set_ylabel("dropout share / min")
        for t in TRACTORS:
            ax.plot(v.index, v[t], color=TRACTOR_COLOR[t], lw=2.0 if t in d.tractors else 1.0)
    for ax in axes:
        ax.axvspan(d.start, d.end, color=CLASS_COLOR[d.cls], alpha=0.12, lw=0)
    _fmt_axis(axes[-1])
    if d.cls == "MULTIPATH":
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz="UTC"))
    axes[0].set_title(f"{d.cls} {', '.join(d.tractors)}  {d.start:%H:%M:%S}-{d.end:%H:%M:%S}Z (shaded)",
                      loc="left", color=INK)
    fig.tight_layout()
    return _png(fig)


CSS = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;--line:#e4e3df;--card:#ffffff;--ok:#0b6b2e;--warn:#8a5a00}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1a1a19;--ink:#f0efec;--muted:#c3c2b7;--line:#3a3936;--card:#232321;--ok:#7bd88f;--warn:#f0c060}}
:root[data-theme="dark"]{--bg:#1a1a19;--ink:#f0efec;--muted:#c3c2b7;--line:#3a3936;--card:#232321;--ok:#7bd88f;--warn:#f0c060}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1120px;margin:0 auto;padding:24px 16px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:32px 0 8px}
.sub{color:var(--muted)}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:16px 0}
.chip{display:inline-block;padding:1px 8px;border-radius:999px;color:#fff;font-size:12px;font-weight:600}
img{max-width:100%;border-radius:6px;background:#fcfcfb}
ul.ev{padding-left:0;list-style:none}ul.ev li{margin:4px 0}
.s{color:var(--ok)}.a{color:var(--warn)}.op{border-left:4px solid var(--ink);padding:6px 10px;margin-top:8px}
details summary{cursor:pointer;color:var(--muted)}
.wrap{overflow-x:auto}
"""


def _evidence_html(d):
    items = []
    for line in d.evidence()[1:]:
        if line.startswith("SUPPORTS: "):
            items.append(f'<li><span class="s">&#10003; supports</span> {html.escape(line[10:])}</li>')
        elif line.startswith("AGAINST: "):
            items.append(f'<li><span class="a">&#9888; against</span> {html.escape(line[9:])}</li>')
        elif line.startswith("OPERATOR: "):
            continue
        else:
            items.append(f"<li>{html.escape(line)}</li>")
    return '<ul class="ev">' + "".join(items) + "</ul>"


def _chip(cls):
    return f'<span class="chip" style="background:{CLASS_COLOR[cls]}">{cls}</span>'


def write(path, ds, sig, decisions, start, end, window):
    major = [d for d in decisions if d.cls != "MULTIPATH"]
    mp = [d for d in decisions if d.cls == "MULTIPATH"]
    rows = "".join(
        f"<tr><td>E{i:02d}</td><td>{d.start:%H:%M:%S}&ndash;{d.end:%H:%M:%S}Z</td><td>{', '.join(d.tractors)}</td>"
        f"<td>{_chip(d.cls)}</td><td><code>{d.action}</code></td><td>{d.confidence:.2f}</td></tr>"
        for i, d in enumerate(major, 1))
    cards = "".join(
        f'<div class="card" id="E{i:02d}"><h3>E{i:02d} &middot; {_chip(d.cls)} &middot; {", ".join(d.tractors)} '
        f'&middot; confidence {d.confidence:.2f}</h3><p>{html.escape(d.summary)}</p>'
        f'<img alt="{d.cls} episode chart" src="data:image/png;base64,{episode_figure(ds, sig, d)}">'
        f"{_evidence_html(d)}<div class=\"op\"><b>Recommended action <code>{d.action}</code>.</b> "
        f"{html.escape(d.guidance)}</div></div>"
        for i, d in enumerate(major, 1))
    mp_counts = {t: sum(1 for d in mp if d.tractors == [t]) for t in TRACTORS}
    mp_example = ""
    if mp:
        ex = mp[len(mp) // 2]
        mp_example = (f'<img alt="multipath example" src="data:image/png;base64,{episode_figure(ds, sig, ex)}">'
                      f"{_evidence_html(ex)}")
    mp_rows = "".join(f"<tr><td>{d.start:%H:%M:%S}&ndash;{d.end:%H:%M:%S}Z</td><td>{d.tractors[0]}</td>"
                      f"<td>{d.metrics['gap_rise_db']:.1f} dB</td><td>{d.metrics['xte_peak_m']:.2f} m</td>"
                      f"<td>{d.confidence:.2f}</td></tr>" for d in mp)
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Straight Rows Report</title>
<style>{CSS}</style></head><body><main>
<h1>Straight Rows &middot; fleet guidance report</h1>
<p class="sub">Window {start:%Y-%m-%d %H:%M}Z to {end:%Y-%m-%d %H:%M}Z ({window}) &middot; three tractors near Casselton, ND &middot;
{len(major)} episodes + {len(mp)} brief headland multipath passes. All data synthetic and unclassified.</p>
<h2>Episodes and recommended actions</h2><div class="wrap"><table>
<tr><th>ID</th><th>Time (UTC)</th><th>Tractors</th><th>Cause</th><th>Action</th><th>Confidence</th></tr>{rows}</table></div>
<h2>Timeline</h2><img alt="timeline of detections, C/N0 and space weather" src="data:image/png;base64,{timeline(ds, sig, decisions, start, end)}">
<h2>Episode evidence</h2>{cards}
<h2>Recurring headland multipath</h2>
<div class="card"><p>Every turn at the north headland produces a ~17 s reflection: the weakest satellite drops several dB while
mean C/N0, RTK fix and guidance stay healthy. Passes reported: {", ".join(f"{t} {n}" for t, n in mp_counts.items())}.
Each is reported separately as a &le;3 min <code>CONTINUE_FALSE_ALARM</code>; the official rules exempt recurrent
short multipath reports from false-positive penalties.</p>{mp_example}
<details><summary>All {len(mp)} multipath passes</summary><div class="wrap"><table>
<tr><th>Time</th><th>Tractor</th><th>Min C/N0 dip</th><th>Peak XTE</th><th>Confidence</th></tr>{mp_rows}</table></div></details></div>
</main></body></html>"""
    path.write_text(doc)
