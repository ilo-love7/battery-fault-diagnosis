"""Mendeley《EV Lithium Ion Battery Fault Diagnosis Dataset》加载。

数据形态：每个 xlsx 是一条连续时间序列（行序即时间序），
三分类版 1152 行 = 健康 424 → 过放电 364 → 短路 364 三个连续区块。
列：SoC(%), Temperature(K), Voltage(V), Label(0/1/2)。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CLASS_NAMES = {0: "healthy", 1: "overdischarge", 2: "short_circuit"}
AMBIENT_TEMP_K = 298.15


@dataclass
class MendeleyData:
    df: pd.DataFrame          # 含列 [soc, temp, volt, label, step_in_block]
    blocks: list[tuple[int, int, int]]  # (start, end_exclusive, label) 连续区块
    source: str               # "multi" | "simple"

    @property
    def n_classes(self) -> int:
        return int(self.df["label"].nunique())


def load_mendeley(xlsx_path: str, source: str = "multi") -> MendeleyData:
    """读取 xlsx 并规范化列名、识别连续区块。"""
    raw = pd.read_excel(xlsx_path)
    raw.columns = [c.strip().lower() for c in raw.columns]
    # 两版列名统一：soc/temperature/voltage/label
    df = pd.DataFrame({
        "soc": raw["soc"].astype(float),
        "temp": raw["temperature"].astype(float),
        "volt": raw["voltage"].astype(float),
        "label": raw["label"].astype(int),
    })
    lab = df["label"].values
    change = np.r_[0, np.where(lab[1:] != lab[:-1])[0] + 1]
    ends = np.r_[change[1:], len(lab)]
    blocks = [(int(a), int(b), int(lab[a])) for a, b in zip(change, ends)]
    df["step_in_block"] = np.concatenate(
        [np.arange(b - a) for a, b, _ in blocks]
    )
    return MendeleyData(df=df, blocks=blocks, source=source)


def load_multi(xlsx_path: str) -> MendeleyData:
    return load_mendeley(xlsx_path, source="multi")


def load_simple(xlsx_path: str) -> MendeleyData:
    d = load_mendeley(xlsx_path, source="simple")
    d.df["label"] = d.df["label"].map({0: 0, 1: 1})  # 二分类：0 健康 / 1 不健康
    d.blocks = [(a, b, int(d.df["label"].iloc[a])) for a, b, _ in d.blocks]
    return d
