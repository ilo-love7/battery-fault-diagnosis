"""CH-BatteryGen 加载与车端包级特征（跨数据集验证用）。

CSV 字段：TIME, CHARGE_STATUS, SUM_VOLTAGE, SUM_CURRENT, SOC,
MAX_CELL_VOLT, MIN_CELL_VOLT, MAX_TEMP, MIN_TEMP, VOLT_1..VOLT_M。
按论文精神做 segment 级样本：每段抽成统计特征向量，车级标签。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FAULT_TYPES = ["normal", "self_discharge", "high_resistance", "low_capacity"]


@dataclass
class SegmentSample:
    chemistry: str          # NCM / LFP
    fault: str              # normal / self_discharge / high_resistance / low_capacity
    vin: str
    segment: str            # 文件名
    mode: str               # charge / discharge
    df: pd.DataFrame        # 原始段数据（最多 max_rows 行）


def list_segments(root: str, chemistry: str, fault: str, vin: str) -> list[Path]:
    d = Path(root) / chemistry / fault / vin
    return sorted(d.glob("*.csv"))


def load_segment(path: Path, max_rows: int = 400) -> pd.DataFrame:
    df = pd.read_csv(path)
    if max_rows and len(df) > max_rows:
        # 均匀下采样，保留整段轮廓（10s 采样下 400 步 ≈ 66 分钟）
        idx = np.linspace(0, len(df) - 1, max_rows).astype(int)
        df = df.iloc[idx].reset_index(drop=True)
    return df


def iter_vehicle_segments(
    root: str, chemistry: str, fault: str, vin: str,
    segments_per_vehicle: int = 3, max_rows: int = 400,
) -> list[SegmentSample]:
    out: list[SegmentSample] = []
    for p in list_segments(root, chemistry, fault, vin)[:segments_per_vehicle]:
        mode = "discharge" if "discharge" in p.stem.lower() else "charge"
        df = load_segment(p, max_rows)
        if len(df) < 20:
            continue
        out.append(SegmentSample(chemistry, fault, vin, p.name, mode, df))
    return out


def cell_voltage_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("VOLT_")]


def add_pack_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """追加包级因果告警指标列（spread / dev_min / d_temp3），流式可得。"""
    out = df.copy()
    vcols = cell_voltage_cols(df)
    V = df[vcols].values
    out["spread"] = out["MAX_CELL_VOLT"] - out["MIN_CELL_VOLT"]
    out["dev_min"] = np.median(V, axis=1) - out["MIN_CELL_VOLT"].values
    out["d_temp3"] = out["MAX_TEMP"].diff(3).fillna(0.0)
    return out


def segment_features(df: pd.DataFrame) -> dict[str, float]:
    """段级统计特征（对应论文 12 维统计特征的思想，扩展为 18 维）。

    分两类：
    - 包级场：SOC/电流/总电压/温度统计；
    - 一致性场：单体电压离散度、极差、离群深度（自放电/低容量的关键信号）。
    """
    vcols = cell_voltage_cols(df)
    V = df[vcols].values                      # [T, M]
    median_v = np.median(V, axis=1)           # 每时刻包内中位数
    min_v = V.min(axis=1)
    max_v = V.max(axis=1)
    spread = max_v - min_v
    dev_min = median_v - min_v                # 最偏低单体深度（自放电特征）
    dev_max = max_v - median_v
    cur = df["SUM_CURRENT"].values
    t = np.arange(len(df))
    # 单体时序方差（自放电单体会随时间漂移，方差偏大）
    cell_var = V.var(axis=0)
    # 端电压斜率（每 100 步）
    def _slope(y):
        if len(y) < 3 or np.allclose(y, y[0]):
            return 0.0
        return float(np.polyfit(t, y, 1)[0] * 100)
    f: dict[str, float] = {
        # 包级
        "soc_mean": float(df["SOC"].mean()),
        "soc_std": float(df["SOC"].std()),
        "soc_min": float(df["SOC"].min()),
        "current_mean": float(np.abs(cur).mean()),
        "current_std": float(cur.std()),
        "sumvolt_mean": float(df["SUM_VOLTAGE"].mean()),
        "sumvolt_std": float(df["SUM_VOLTAGE"].std()),
        "temp_max": float(df["MAX_TEMP"].max()),
        "temp_min": float(df["MIN_TEMP"].min()),
        "temp_spread": float((df["MAX_TEMP"] - df["MIN_TEMP"]).mean()),
        "dt_max_abs": float(np.abs(np.diff(df["MAX_TEMP"].values)).max()) if len(df) > 1 else 0.0,
        # 一致性场
        "cell_spread_mean": float(spread.mean()),
        "cell_spread_max": float(spread.max()),
        "cell_spread_std": float(spread.std()),
        "dev_min_mean": float(dev_min.mean()),
        "dev_min_max": float(dev_min.max()),
        "dev_min_p90": float(np.quantile(dev_min, 0.9)),
        "dev_max_mean": float(dev_max.mean()),
        "cell_var_max": float(cell_var.max()) if V.shape[1] else 0.0,
        "minvolt_slope": _slope(min_v),
        "maxvolt_slope": _slope(max_v),
        "minvolt_min": float(min_v.min()),
    }
    return f
