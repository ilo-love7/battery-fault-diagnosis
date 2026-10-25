"""特征工程（全部因果/只用当前与过去，模拟 BMS 在线流式计算）。

注意：不按标签分组滚动——在线 BMS 不知道未来标签，滚动统计只能沿
时间流计算；跨区块边界的过渡行保留"故障初期仍带健康历史"的真实形态。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

AMBIENT_TEMP_K = 298.15

# 点级特征列（经典模型用）
POINT_FEATURES = [
    "soc", "temp", "volt",                # 原始 3 维
    "t_above_amb",                        # 高于环境温度
    "d_soc", "d_temp", "d_volt",          # 1 步差分（首行补 0）
    "volt_rollmean5", "temp_rollmean5",   # 5 步滚动均值
    "volt_rollstd5", "temp_rollstd5",     # 5 步滚动波动
]

RAW_COLS = ["soc", "temp", "volt"]


def add_point_features(df: pd.DataFrame) -> pd.DataFrame:
    """沿整条轨迹追加因果特征（只用当前与过去的窗口）。"""
    out = df.copy()
    out["t_above_amb"] = out["temp"] - AMBIENT_TEMP_K
    for col in ("soc", "temp", "volt"):
        out[f"d_{col}"] = out[col].diff().fillna(0.0)
        out[f"{col}_rollmean5"] = out[col].rolling(5, min_periods=1).mean()
        out[f"{col}_rollstd5"] = out[col].rolling(5, min_periods=1).std().fillna(0.0)
    return out


def fit_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return mu, sd


def apply_standardizer(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd
