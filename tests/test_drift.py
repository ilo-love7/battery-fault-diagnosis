"""漂移监控（PSI）正确性测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evfd.evaluate.drift import drift_report, psi


def test_psi_identical_distribution_near_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    assert psi(x, x) < 0.01
    assert psi(x, rng.normal(0, 1, 5000)) < 0.05


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    y = rng.normal(3, 1, 5000)          # 均值漂移 3σ
    assert psi(x, y) > 1.0


def test_psi_constant_column():
    x = np.full(100, 3.5)
    assert psi(x, x) == 0.0
    assert psi(x, np.full(100, 4.0)) == float("inf")


def test_drift_report_summary_and_support():
    rng = np.random.default_rng(1)
    tr = rng.normal(0, 1, (1000, 2))
    te = rng.normal(0, 1, (1000, 2))
    te[:, 0] += 5.0                      # 特征 0 漂移 + 大部分出支撑
    rep = drift_report(tr, te, ["a", "b"])
    assert rep["_summary"]["max_psi_feature"] == "a"
    assert rep["a"]["out_of_support"] > 0.8   # 5σ 漂移、n=1000 训练支撑 ±3.7σ
    assert rep["b"]["out_of_support"] < 0.02
