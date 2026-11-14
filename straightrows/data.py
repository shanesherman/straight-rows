"""Load the farm telemetry and SDA feeds into tidy pandas objects."""
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .constants import TRACTORS


@dataclass
class Dataset:
    tele: pd.DataFrame            # long format, one row per tractor-second
    wide: dict                    # channel -> DataFrame(index=ts, columns=T1..T3)
    kp: pd.Series
    dst: pd.Series
    s4: pd.Series
    advisories: list
    rfi: list
    labels: list = field(default_factory=list)


def _series(rows, key):
    frame = pd.DataFrame(rows)
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    return frame.set_index("ts")[key].sort_index()


def load(data_dir) -> Dataset:
    data = Path(data_dir)
    tele = pd.read_csv(data / "farm/tractor_telemetry.csv.gz")
    tele["ts"] = pd.to_datetime(tele["ts"], utc=True)
    tele = tele.sort_values(["tractor_id", "ts"]).reset_index(drop=True)
    tele["rtk_fixed"] = (tele["fix_state"] == "RTK_FIXED").astype(float)
    tele["base_link_ok"] = tele["base_link_ok"].astype(float)
    channels = {
        "cn0": "mean_cn0_dbhz", "min_cn0": "min_cn0_dbhz", "xte": "cross_track_error_m",
        "speed": "speed_mps", "link": "base_link_ok", "rtk": "rtk_fixed",
        "sats": "num_sats", "lat": "lat", "lon": "lon",
    }
    wide = {
        name: tele.pivot(index="ts", columns="tractor_id", values=col)[list(TRACTORS)]
        for name, col in channels.items()
    }
    sgi = json.loads((data / "sda/sgi_indices.json").read_text())
    s4 = json.loads((data / "sda/scintillation_series.json").read_text())
    adv = json.loads((data / "sda/constellation_advisories.json").read_text())["advisories"]
    rfi = json.loads((data / "sda/rfi_reports.json").read_text())["reports"]
    labels_path = data / "labels/train_episodes.json"
    labels = json.loads(labels_path.read_text())["episodes"] if labels_path.exists() else []
    for item in adv + rfi:
        item["ts"] = pd.Timestamp(item["ts"])
    return Dataset(
        tele=tele, wide=wide,
        kp=_series(sgi["kp"], "kp"), dst=_series(sgi["dst"], "dst_nT"),
        s4=_series(s4["series"], "s4"),
        advisories=sorted(adv, key=lambda a: a["ts"]),
        rfi=sorted(rfi, key=lambda r: r["ts"]),
        labels=labels,
    )
