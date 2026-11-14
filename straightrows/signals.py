"""Signal conditioning and the S4-informed regional C/N0 model.

The key decomposition, per tractor i and second t:

    deficit_i(t) = NOMINAL_CN0 - smoothed mean C/N0
                 = regional(t)          # spatially uniform, explained by scintillation (S4)
                 + excess_i(t)          # anything else: interference, local effects

`regional` is measured from the farm itself (seconds when all three tractors
agree within 1 dB) but only accepted where it is consistent with what the
regional S4 proxy predicts. A uniform loss that S4 cannot explain is therefore
left in `excess` and can be attributed to interference rather than the Sun.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import NOMINAL_CN0, TRACTORS


@dataclass
class Signals:
    index: pd.DatetimeIndex
    working: pd.DataFrame        # bool per tractor
    cn0: pd.DataFrame            # smoothed mean C/N0
    deficit: pd.DataFrame
    regional: pd.Series          # S4-consistent uniform loss (dB)
    s4_expected: pd.Series       # loss predicted from S4 alone (dB)
    excess: pd.DataFrame         # deficit - regional
    xte: pd.DataFrame            # smoothed cross-track error
    xte_resid: pd.DataFrame      # xte minus the cleanest neighbour
    gap: pd.DataFrame            # raw mean-minus-min C/N0 gap (dB)
    link_bad: pd.DataFrame       # raw 1 - base_link_ok
    rtk: pd.DataFrame            # raw RTK_FIXED indicator


def rolling_median(frame, window):
    return frame.rolling(window, center=True, min_periods=1).median()


def s4_on_grid(s4: pd.Series, index):
    """S4 (15-minute cadence) forward-filled onto the 1 Hz telemetry grid."""
    grid = s4.reindex(s4.index.union(index)).sort_index().ffill().reindex(index)
    return grid.fillna(0.0)


def tolerance_band(series: pd.Series, tolerance="30min"):
    """Rolling min and max within +/- tolerance.

    The guide warns S4 lags Kp and should be compared with actual farm impact
    rather than assumed to share its timing, so acceptance uses a time band.
    """
    steps = int(pd.Timedelta(tolerance).total_seconds()) * 2 + 1
    roll = series.rolling(steps, center=True, min_periods=1)
    return roll.min(), roll.max()


@dataclass
class S4Response:
    """Monotone piecewise-linear map from S4 to expected uniform C/N0 loss (dB)."""
    knots_s4: np.ndarray
    knots_db: np.ndarray

    def __call__(self, s4):
        return np.interp(np.asarray(s4, dtype=float), self.knots_s4, self.knots_db)

    def describe(self):
        return ", ".join(f"S4 {s:.2f} -> {d:.1f} dB" for s, d in zip(self.knots_s4, self.knots_db))


def fit_s4_response(deficit_uniform: pd.Series, s4_grid: pd.Series, bins=(0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)):
    """Fit expected uniform loss vs S4 from development data (isotonic, binned medians).

    Uses minutes where all tractors agree (spatially uniform loss), which is
    exactly what regional scintillation should produce.
    """
    frame = pd.DataFrame({"d": deficit_uniform, "s4": s4_grid}).dropna()
    frame = frame.resample("1min").median().dropna()
    centers, levels = [0.0], [0.0]
    for lo, hi in zip(bins[:-1], bins[1:]):
        chunk = frame[(frame.s4 >= lo) & (frame.s4 < hi)]
        if len(chunk) >= 10:
            centers.append(float(chunk.s4.median()))
            levels.append(float(max(0.0, chunk.d.median())))
    centers, levels = np.array(centers), np.maximum.accumulate(np.array(levels))
    order = np.argsort(centers)
    centers, levels = centers[order], levels[order]
    centers = np.append(centers, 1.5)
    levels = np.append(levels, levels[-1])
    return S4Response(centers, levels)


def uniform_level(deficit: pd.DataFrame, th):
    spread = deficit.max(axis=1) - deficit.min(axis=1)
    return deficit.median(axis=1).where(spread < th["uniform_spread_db"])


def build(ds, th, response: S4Response | None = None, fit_window=None, use_sda=True) -> Signals:
    """use_sda=False is the farm-only ablation: any uniform loss counts as regional."""
    w = ds.wide
    index = w["cn0"].index
    working = w["speed"] > th["working_speed_mps"]
    cn0 = rolling_median(w["cn0"], th["smooth_s"])
    deficit = NOMINAL_CN0 - cn0
    s4_grid = s4_on_grid(ds.s4, index)
    uniform = uniform_level(deficit, th)
    if response is None:
        mask = slice(None) if fit_window is None else slice(*fit_window)
        response = fit_s4_response(uniform.loc[mask], s4_grid.loc[mask])
    expected = pd.Series(response(s4_grid.values), index=index)
    # Accept a uniform loss as regional only when S4 can explain it: within 4 dB
    # of the loss S4 predicts at some moment in the surrounding +/- 30 minutes.
    low, high = tolerance_band(expected)
    accepted = uniform.where((uniform >= low - 4.0) & (uniform <= high + 4.0)) if use_sda else uniform
    regional = accepted.rolling(th["regional_window_s"], center=True, min_periods=60).median()
    regional = regional.interpolate(method="time", limit_area="inside")
    regional = regional.fillna(expected if use_sda else 0.0).clip(lower=0.0)
    excess = deficit.sub(regional, axis=0)

    xte = rolling_median(w["xte"], 31)
    resid = pd.DataFrame(index=index, columns=list(TRACTORS), dtype=float)
    for t in TRACTORS:
        others = [o for o in TRACTORS if o != t]
        resid[t] = xte[t] - xte[others].min(axis=1)

    signals = Signals(
        index=index, working=working, cn0=cn0, deficit=deficit, regional=regional,
        s4_expected=expected, excess=excess, xte=xte, xte_resid=resid,
        gap=w["cn0"] - w["min_cn0"], link_bad=1.0 - w["link"], rtk=w["rtk"],
    )
    signals.response = response
    return signals
