"""Locate an interference source from the fleet's own C/N0 loss.

As tractors drive their rows, their distance to an emitter changes, and so does
the excess C/N0 loss. Under free-space propagation the jammer-to-noise ratio
falls 20 dB per decade of range, so for each sample

    excess_dB = A - 20 * log10(range_km)

with an unknown emitter position (x, y) and source strength A. A grid search
over a 16 km x 16 km area minimises the squared residual (A is solved in closed
form at every grid point). The fit uses only telemetry; the RFI report is then
an independent check.
"""
import math

import numpy as np
import pandas as pd

from .constants import TRACTORS

KM_LAT = 110.57


def _frame(lat0):
    k_lon = 111.32 * math.cos(math.radians(lat0))
    return k_lon


def localize(ds, sig, start, end, tractors=TRACTORS, step_km=0.05, extent_km=8.0, min_excess_db=2.0):
    lat0 = float(ds.wide["lat"].mean().mean())
    lon0 = float(ds.wide["lon"].mean().mean())
    k_lon = _frame(lat0)
    xs, ys, es = [], [], []
    a, b = start + pd.Timedelta("60s"), end - pd.Timedelta("60s")
    for t in tractors:
        e = sig.excess[t].loc[a:b].resample("20s").mean()
        lat = ds.wide["lat"][t].loc[a:b].resample("20s").mean()
        lon = ds.wide["lon"][t].loc[a:b].resample("20s").mean()
        keep = (e > min_excess_db).to_numpy()
        xs.append(((lon.to_numpy() - lon0) * k_lon)[keep])
        ys.append(((lat.to_numpy() - lat0) * KM_LAT)[keep])
        es.append(e.to_numpy()[keep])
    x, y, e = np.concatenate(xs), np.concatenate(ys), np.concatenate(es)
    if len(e) < 30:
        return None
    def search(cx, cy, half, step):
        axis = np.arange(-half, half + 1e-9, step)
        gx, gy = np.meshgrid(cx + axis, cy + axis)
        gx, gy = gx.ravel(), gy.ravel()
        f = -20.0 * np.log10(np.hypot(x[None, :] - gx[:, None], y[None, :] - gy[:, None]) + 0.01)
        resid = e[None, :] - f
        resid -= resid.mean(axis=1, keepdims=True)
        return gx, gy, np.sqrt((resid ** 2).mean(axis=1))

    gx, gy, rms = search(0.0, 0.0, extent_km, 0.2)          # coarse pass
    best = int(rms.argmin())
    gx, gy, rms = search(gx[best], gy[best], 0.8, step_km)   # fine pass around the coarse optimum
    best = int(rms.argmin())
    bx, by = float(gx[best]), float(gy[best])
    near = rms <= rms[best] + 0.1  # grid points fitting within 0.1 dB RMS of the best
    radius = float(np.max(np.hypot(gx[near] - bx, gy[near] - by)))
    edge = max(abs(bx), abs(by)) >= extent_km - step_km
    return {
        "lat": lat0 + by / KM_LAT, "lon": lon0 + bx / k_lon,
        "rms_db": float(rms[best]), "samples": int(len(e)),
        "uncertainty_km": max(radius, 0.25), "at_search_edge": bool(edge),
        "fleet_lat": lat0, "fleet_lon": lon0,
        "range_km": float(math.hypot(bx, by)),
        "bearing_deg": float((math.degrees(math.atan2(bx, by)) + 360) % 360),
    }
