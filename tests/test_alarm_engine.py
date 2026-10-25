"""告警引擎与特征工程的正确性测试。

运行：python -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evfd.alarm.thresholds import design_thresholds, run_alarms
from evfd.data.features import add_point_features


def _mk_df(n=50):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "temp": 300 + rng.normal(0, 1, n).cumsum() * 0.01,
        "volt": 3.5 + rng.normal(0, 1, n).cumsum() * 0.001,
        "soc": np.linspace(100, 50, n),
        "label": 0,
    })


def test_persistence_requires_consecutive_crossings():
    df = _mk_df()
    df["sig"] = 0.0
    df.loc[30:34, "sig"] = 5.0     # 5 步越限 → 应触发（持久 3 步后）
    df.loc[45, "sig"] = 5.0        # 单步越限 → 不应触发
    ts = design_thresholds(
        df.iloc[:20], [{"name": "R", "column": "sig", "direction": "high"}],
        quantile=0.9,
    )
    res, events = run_alarms(df, ts, persistence=3, cooldown=10)
    assert res["R_trig"].iloc[32:35].all()
    assert not res["R_trig"].iloc[45]
    assert len(events["R"]) == 1 and events["R"][0].start == 32  # 持久 3 步后触发


def test_direction_low_threshold():
    df = _mk_df()
    df["v"] = np.linspace(4.0, 3.0, len(df))
    ts = design_thresholds(
        df.iloc[:20], [{"name": "LOW", "column": "v", "direction": "low"}],
        quantile=0.99,
    )
    # direction=low：阈值 = 定标段的 1% 分位（低尾），应落在定标段量程下沿附近
    assert 3.5 < ts.rules[0].threshold < 3.7
    res, events = run_alarms(df, ts, persistence=3, cooldown=10)
    assert len(events["LOW"]) >= 1          # 后段电压低于定标低尾 → 触发
    assert events["LOW"][0].start >= 20     # 且在定标段之外才触发


def test_hard_rule_independent_of_quantile():
    df = _mk_df()
    df["v"] = 3.5
    ts = design_thresholds(
        df, [{"name": "X", "column": "v", "direction": "high"}],
        quantile=0.99, hard_rules=[
            {"name": "H", "column": "v", "direction": "low", "threshold": 0.5},
        ],
    )
    assert ts.rules[1].kind == "hard" and ts.rules[1].threshold == 0.5


def test_point_features_are_causal():
    df = _mk_df(30)
    out = add_point_features(df)
    # 改变未来行不应影响过去行的滚动特征
    df2 = df.copy()
    df2.loc[25:, "volt"] = 99.0
    out2 = add_point_features(df2)
    assert np.allclose(
        out["volt_rollmean5"].iloc[:25], out2["volt_rollmean5"].iloc[:25]
    )
    assert np.allclose(out["d_volt"].iloc[:25], out2["d_volt"].iloc[:25])


def test_event_cooldown_merging():
    n = 60
    df = pd.DataFrame({"sig": np.concatenate([
        np.zeros(10), np.ones(3), np.zeros(4), np.ones(3), np.zeros(40)
    ]), "label": 0})
    df["volt"] = 3.5
    df["temp"] = 300.0
    df["soc"] = 50.0
    ts = design_thresholds(
        df.iloc[:5], [{"name": "R", "column": "sig", "direction": "high"}],
        quantile=0.5,
    )
    res, events = run_alarms(df, ts, persistence=3, cooldown=10)
    # 两个触发段间隔 4 步 < cooldown=10 → 合并为一个事件
    assert len(events["R"]) == 1
