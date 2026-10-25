"""CH-BatteryGen 采样（脚本 05/06 共用）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from evfd.data.load_chbatterygen import FAULT_TYPES, load_segment


def pick_vehicles(root: str, chem: str, fault: str) -> list[str]:
    d = Path(root) / chem / fault
    vins = sorted({p.parent.name for p in d.glob("vin_*/*.csv")})
    vins.sort(key=lambda s: int(s.split("_")[1]))
    return vins


def select_segment_files(root: str, chem: str, fault: str, vin: str) -> list[Path]:
    """每车取 charge_1 + discharge_1 + charge_2（缺则顺延）。"""
    d = Path(root) / chem / fault / vin
    out = []
    for pat in ["vin_*_charge_1.csv", "vin_*_discharge_1.csv", "vin_*_charge_2.csv"]:
        found = sorted(d.glob(pat))
        if found:
            out.append(found[0])
    return out


def sample_segments(
    root: str,
    normal_vehicles_per_chem: int = 40,
    segments_per_vehicle: int = 3,
    max_rows: int = 400,
    seed: int = 42,
    chems: tuple[str, ...] = ("NCM", "LFP"),
):
    """返回 (samples, design_normals, ch_map)。

    samples: [(chem, fault, vin, segname, df)]；正常车按 seed 对半分为设计/测试。
    """
    from evfd.config import load_config

    ch_map = load_config()["ch_classes"]
    rng = np.random.default_rng(seed)
    samples: list[tuple[str, str, str, str, "object"]] = []
    design_normals: dict[str, list[str]] = {}
    for chem in chems:
        nv = pick_vehicles(root, chem, "normal")
        idx = rng.permutation(len(nv))
        design_normals[chem] = [nv[i] for i in idx[: normal_vehicles_per_chem // 2]]
        test_normals = [nv[i] for i in idx[normal_vehicles_per_chem // 2: normal_vehicles_per_chem]]
        for vin in design_normals[chem] + test_normals + sum(
            (pick_vehicles(root, chem, f) for f in FAULT_TYPES[1:]), []
        ):
            fault = "normal"
            for f in FAULT_TYPES[1:]:
                if (Path(root) / chem / f / vin).exists():
                    fault = f
                    break
            for p in select_segment_files(root, chem, fault, vin)[:segments_per_vehicle]:
                df = load_segment(p, max_rows)
                if len(df) >= 20:
                    samples.append((chem, fault, vin, p.name, df))
    return samples, design_normals, ch_map
