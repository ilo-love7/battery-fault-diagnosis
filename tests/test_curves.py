"""曲线表示模块（残差热图 / SOC 对齐形状）正确性测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evfd.data.curves import (
    SCALE_V,
    _bilinear_resize,
    residual_heatmap,
    soc_aligned_shape,
)


def _mk_charge_df(n=120, M=10, fault_cell=None, fault_offset=0.0, early_lag=0.0):
    t = np.arange(n)
    soc = np.linspace(40, 85, n)
    df = pd.DataFrame({
        "TIME": t * 10, "CHARGE_STATUS": 1, "SUM_VOLTAGE": 300 + 0.1 * t,
        "SUM_CURRENT": -50.0, "SOC": soc, "MAX_CELL_VOLT": 3.7, "MIN_CELL_VOLT": 3.6,
        "MAX_TEMP": 30.0, "MIN_TEMP": 25.0,
    })
    rng = np.random.default_rng(0)
    for c in range(1, M + 1):
        v = 3.5 + 0.003 * t / 10 + rng.normal(0, 0.0005, n)
        if fault_cell is not None and c == fault_cell:
            v = v - fault_offset - early_lag * (1 - t / n)   # 静态偏低 + 前期滞后
        df[f"VOLT_{c}"] = v
    return df


def test_bilinear_resize_identity():
    img = np.random.default_rng(1).normal(0, 1, (16, 12)).astype(np.float32)
    out = _bilinear_resize(img, 16, 12)
    assert np.allclose(out, img, atol=1e-5)


def test_bilinear_resize_shape_and_range():
    img = np.clip(np.random.default_rng(2).normal(0, 0.3, (37, 91)), -1, 1).astype(np.float32)
    out = _bilinear_resize(img, 128, 96)
    assert out.shape == (128, 96) and out.dtype == np.float32
    assert out.min() >= img.min() - 0.05 and out.max() <= img.max() + 0.05


def test_residual_heatmap_fixed_scale_and_stripe():
    df = _mk_charge_df(fault_cell=5, fault_offset=0.02)     # 20mV 偏低 → 热图 -0.4
    H = residual_heatmap(df)
    assert H.shape == (128, 96)
    col = H[:, int(4.5 / 10 * 96): int(5.5 / 10 * 96)]       # 单体 5 对应列带
    other = np.delete(H, np.arange(int(4.5 / 10 * 96), int(5.5 / 10 * 96)), axis=1)
    assert col.mean() < -0.2 and abs(other.mean()) < 0.1     # 条带暗、其余近零
    assert abs(SCALE_V - 0.05) < 1e-9                        # 固定物理尺度不被段内自适应


def test_shape_features_capture_slow_early_rise():
    normal = soc_aligned_shape(_mk_charge_df())
    sd = soc_aligned_shape(_mk_charge_df(fault_cell=3, fault_offset=0.01, early_lag=0.02))
    # 自放电（持续偏低型，论文 Fig.2b 机理）：低进度处最深单体电压更低、离散度更大
    assert sd["minv_p10"] < normal["minv_p10"]
    assert sd["spread_p10"] > normal["spread_p10"]
    assert list(normal.keys()) == list(sd.keys())
