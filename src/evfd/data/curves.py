"""全曲线形状表示（攻自放电：形状特征而非统计偏移，报告 §6.3）。

两种互补表示：
1. residual_heatmap —— 单体电压残差热图 [T×M]：逐时刻减包中位数（去共模 SOC
   趋势），故障单体呈亮/暗条纹（论文 Fig.2/3 的机理可视化）。用**固定物理尺度**
   （50 mV 满量程）而非段内自适应缩放——后者会把故障条纹归一化掉。
2. soc_aligned_shape —— SOC 相对进度对齐的曲线描述子：min-cell / 离散度 / 中位数
   三条曲线在固定进度点的取值 + 早期爬升斜率（自放电"前期爬升慢"的直接度量）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from evfd.data.load_chbatterygen import cell_voltage_cols

SCALE_V = 0.05          # 残差热图满量程：±50 mV → ±1（固定物理尺度，跨段可比）
HEAT_T, HEAT_M = 128, 96


def residual_heatmap(df: pd.DataFrame, target_t: int = HEAT_T,
                     target_m: int = HEAT_M) -> np.ndarray:
    """返回 [target_t, target_m] float32，值域 [-1,1]。"""
    V = df[cell_voltage_cols(df)].values.astype(np.float32)      # [T, M]
    R = V - np.median(V, axis=1, keepdims=True)
    H = np.clip(R / SCALE_V, -1.0, 1.0)
    return _bilinear_resize(H, target_t, target_m)


def _bilinear_resize(img: np.ndarray, th: int, tw: int) -> np.ndarray:
    """简单双线性缩放（无 cv2/scipy 依赖）。"""
    T, M = img.shape
    ti = np.linspace(0, T - 1, th)
    mi = np.linspace(0, M - 1, tw)
    i0 = np.floor(ti).astype(int); i1 = np.minimum(i0 + 1, T - 1)
    m0 = np.floor(mi).astype(int); m1 = np.minimum(m0 + 1, M - 1)
    a = img[np.ix_(i0, m0)]      # [th, tw] 左上
    b = img[np.ix_(i1, m0)]      # 左下
    c = img[np.ix_(i0, m1)]      # 右上
    d = img[np.ix_(i1, m1)]      # 右下
    wi = (ti - i0)[:, None]      # [th, 1] 行权重
    wm = (mi - m0)[None, :]      # [1, tw] 列权重
    return (a * (1 - wi) * (1 - wm) + b * wi * (1 - wm)
            + c * (1 - wi) * wm + d * wi * wm).astype(np.float32)


SHAPE_GRID = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
SHAPE_FEATURES = (
    [f"minv_p{int(p*100)}" for p in SHAPE_GRID]
    + [f"spread_p{int(p*100)}" for p in SHAPE_GRID]
    + [f"medv_p{int(p*100)}" for p in SHAPE_GRID]
    + ["minv_early_slope", "spread_early_slope", "medv_full_slope"]
)


def soc_aligned_shape(df: pd.DataFrame) -> dict[str, float]:
    """SOC 相对进度（0-1）对齐的三条曲线 + 斜率特征（充电段用）。"""
    soc = df["SOC"].values.astype(float)
    V = df[cell_voltage_cols(df)].values.astype(float)
    min_v, max_v = V.min(1), V.max(1)
    spread = max_v - min_v
    med = np.median(V, axis=1)
    rng = soc.max() - soc.min()
    p = (soc - soc.min()) / rng if rng > 1e-6 else np.zeros_like(soc)
    order = np.argsort(p)
    ps = p[order]
    # SOC 进程需单调化（充电段基本单调；保留首遇点）
    up = np.r_[True, np.diff(ps) > 0]
    ps_u = ps[up]
    feats: dict[str, float] = {}
    curves = {}
    for name, y in [("minv", min_v), ("spread", spread), ("medv", med)]:
        ys = y[order][up]
        # 进一步去重（相同进度取均值）
        pu, inv = np.unique(ps_u, return_inverse=True)
        yu = np.zeros(len(pu))
        cnt = np.zeros(len(pu))
        np.add.at(yu, inv, ys)
        np.add.at(cnt, inv, 1.0)
        yu /= np.maximum(cnt, 1)
        curves[name] = (pu, yu)
        yy = np.interp(SHAPE_GRID, pu, yu)
        feats.update({f"{name}_p{int(g*100)}": float(v) for g, v in zip(SHAPE_GRID, yy)})
    # 早期爬升斜率：进度 0.1→0.4
    for name in ("minv", "spread"):
        pu, yu = curves[name]
        y10 = np.interp(0.1, pu, yu)
        y40 = np.interp(0.4, pu, yu)
        feats[f"{name}_early_slope"] = float((y40 - y10) / 0.3)
    pu, yu = curves["medv"]
    feats["medv_full_slope"] = float((yu[-1] - yu[0]) / max(pu[-1] - pu[0], 1e-6))
    return feats
