"""输入分布漂移监控（PSI + 支撑覆盖度）——树模型外推失效的落地防线。

背景（报告 §4.2）：XGBoost 在 P2 时序划分下过放电召回跌至 0.677，
根因是测试段输入分布超出训练支撑。车端部署时模型本身无法感知外推，
必须在输入侧挂漂移监控：PSI 超限即告警/回退到规则层。
"""

from __future__ import annotations

import numpy as np


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10,
        eps: float = 1e-6) -> float:
    """单变量 PSI（Population Stability Index）。

    分箱以 expected 的分位数为基准；PSI = Σ (a_e - p_e)·ln(a_e/p_e)。
    经验阈值：<0.1 稳定；0.1–0.25 关注；>0.25 显著漂移。
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")
    if np.ptp(expected) == 0:                # 常数列：分箱无意义，直接比较
        return 0.0 if np.allclose(actual, expected[0], atol=1e-12) else float("inf")
    qs = np.quantile(expected, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    qs = np.unique(qs)
    if len(qs) < 3:                          # 退化分箱
        return 0.0
    p = np.histogram(expected, bins=qs)[0] / len(expected)
    a = np.histogram(actual, bins=qs)[0] / len(actual)
    p = np.clip(p, eps, None)
    a = np.clip(a, eps, None)
    return float(np.sum((a - p) * np.log(a / p)))


def drift_report(X_train: np.ndarray, X_test: np.ndarray,
                 feature_names: list[str] | None = None) -> dict:
    """逐特征 PSI + 支撑覆盖度（测试样本落在训练 [min,max] 外的比例）。"""
    names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]
    out = {}
    for j, name in enumerate(names):
        tr, te = X_train[:, j], X_test[:, j]
        lo, hi = float(np.min(tr)), float(np.max(tr))
        out[name] = {
            "psi": psi(tr, te),
            "out_of_support": float(np.mean((te < lo) | (te > hi))),
        }
    out["_summary"] = {
        "max_psi_feature": max(
            (k for k in out if k != "_summary"), key=lambda k: out[k]["psi"]
        ),
        "max_psi": max(out[k]["psi"] for k in out if k != "_summary"),
        "mean_psi": float(np.mean([out[k]["psi"] for k in out if k != "_summary"])),
    }
    return out
