"""B1 → A1 桥接的正确性测试。

重点验证三件事：
1. 规格能读、校验能过、阈值与 B1 交付物一致（-1.425 mV）；
2. **Arrhenius 归一自洽**：25 ℃ 实测的 -0.443 mV 归一到 45 ℃ 必须等于 -1.781 mV
   （这正是 B1 用两点定 Ea/R 的定义，归一公式与它必须互逆）；
3. 没有静置段的数据上**不产生特征列、不注册规则**，而不是塞一列 NaN。

运行：python -m pytest tests/test_b1_bridge.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evfd.alarm.b1_bridge import (
    add_feature,
    applicable,
    arrhenius_to_ref,
    check_spec,
    extra_hard_rules,
    find_rest_windows,
    hard_rules,
    load_spec,
    ocv_drift_45c_mv,
)
from evfd.alarm.thresholds import design_thresholds, run_alarms

SPEC_PATH = ROOT / "configs" / "b1_precursor_spec.json"


# ---------------------------------------------------------------- 夹具
def _mk_rest_series(
    t_rest_c: float = 45.0,
    dv_mv: float = -1.781,
    days: float = 7.0,
    n: int = 169,          # 7 天 × 24 h，每小时 1 点
    v0: float = 4.0,
    noise_v: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """合成一条纯静置轨迹：温度恒定、SOC 不变、电压线性漂移 dv_mv。"""
    rng = np.random.default_rng(seed)
    t_h = np.linspace(0.0, days * 24.0, n)
    frac = (t_h - t_h[0]) / (t_h[-1] - t_h[0])
    volt = v0 + dv_mv / 1000.0 * frac
    if noise_v:
        volt = volt + rng.normal(0, noise_v, n)
    return pd.DataFrame({
        "t_h": t_h,
        "volt": volt,
        "temp": np.full(n, t_rest_c + 273.15),
        "soc": np.full(n, 60.0),
    })


@pytest.fixture(scope="module")
def spec() -> dict:
    return load_spec(SPEC_PATH)


# ---------------------------------------------------------------- 规格
def test_spec_loads_and_passes_checks(spec):
    assert check_spec(spec) == []
    assert spec["reference_temp_C"] == 45.0
    assert spec["rest_days"] == 7.0
    # B1 v1.6 的两点活化能
    assert spec["ea_over_R_K"] == pytest.approx(6599.9, rel=1e-4)


def test_hard_rules_match_b1_delivery(spec):
    rules = hard_rules(spec)
    assert len(rules) == 1
    r = rules[0]
    assert r["name"] == "ALM_B1_OCV_DRIFT"
    assert r["column"] == "ocv_drift_45c_mv"
    assert r["direction"] == "low"          # 漂移越负越糟
    assert r["threshold"] == pytest.approx(-1.425)   # 标称 -1.781 的 80%
    # 硬规则格式必须能被 design_thresholds 直接消费
    ts = design_thresholds(pd.DataFrame({"ocv_drift_45c_mv": [0.0]}), [], hard_rules=rules)
    assert ts.to_dict()["ALM_B1_OCV_DRIFT"]["kind"] == "hard"


def test_check_spec_flags_wrong_reference_temp(spec):
    bad = dict(spec, reference_temp_C=25.0)
    issues = check_spec(bad)
    assert any("参考温度" in s for s in issues)


def test_load_spec_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="to_a1.py"):
        load_spec(tmp_path / "nope.json")


# ---------------------------------------------------------------- 归一
def test_arrhenius_is_identity_at_reference_temp(spec):
    ea = spec["ea_over_R_K"]
    assert arrhenius_to_ref(-1.781, 318.15, ea) == pytest.approx(-1.781, rel=1e-12)


def test_arrhenius_maps_25c_to_45c_equivalent(spec):
    """B1 的两点定标必须与归一公式互逆：25 ℃ 的 -0.443 mV ⇒ 45 ℃ 的 -1.781 mV。"""
    ea = spec["ea_over_R_K"]
    got = arrhenius_to_ref(-0.443, 298.15, ea)
    assert got == pytest.approx(-1.781, rel=2e-3), f"归一结果 {got:.4f} 与标称 -1.781 不符"


def test_colder_rest_needs_larger_amplification(spec):
    ea = spec["ea_over_R_K"]
    f25 = arrhenius_to_ref(1.0, 298.15, ea)
    f45 = arrhenius_to_ref(1.0, 318.15, ea)
    assert f25 > f45 == pytest.approx(1.0)


# ---------------------------------------------------------------- 静置段
def test_find_rest_window_on_clean_series(spec):
    df = _mk_rest_series()
    wins = find_rest_windows(df, spec["rest_days"])
    assert len(wins) == 1
    assert wins[0] == (0, len(df) - 1)


def test_short_series_has_no_window(spec):
    df = _mk_rest_series(days=3.0, n=73)
    assert find_rest_windows(df, spec["rest_days"]) == []


def test_soc_change_breaks_window(spec):
    df = _mk_rest_series()
    df.loc[df.index[80:], "soc"] = 55.0      # 中途掉 5% SOC > 2% 容差
    wins = find_rest_windows(df, spec["rest_days"])
    assert all(b - a < 80 for a, b in wins)  # 不会跨过 SOC 突变点


def test_temp_excursion_breaks_window(spec):
    df = _mk_rest_series()
    df.loc[df.index[80], "temp"] += 5.0      # 单点温度跳 5 K > 1 K 容差
    wins = find_rest_windows(df, spec["rest_days"])
    assert all(b - a < 80 for a, b in wins)


# ---------------------------------------------------------------- 特征
def test_feature_equals_nominal_at_45c(spec):
    df = _mk_rest_series(t_rest_c=45.0, dv_mv=-1.781)
    i0, i1 = 0, len(df) - 1
    dv45, dv_raw, t_mean = ocv_drift_45c_mv(df, spec, i0, i1)
    assert dv_raw == pytest.approx(-1.781, abs=1e-9)
    assert t_mean == pytest.approx(318.15)
    assert dv45 == pytest.approx(-1.781, rel=1e-12)


def test_feature_equals_nominal_at_25c_after_normalisation(spec):
    df = _mk_rest_series(t_rest_c=25.0, dv_mv=-0.443)
    dv45, dv_raw, _ = ocv_drift_45c_mv(df, spec, 0, len(df) - 1)
    assert dv_raw == pytest.approx(-0.443, abs=1e-9)
    assert dv45 == pytest.approx(-1.781, rel=2e-3)


def test_feature_column_is_causal_and_ffilled(spec):
    df = _mk_rest_series()
    out = add_feature(df, spec)
    col = out["ocv_drift_45c_mv"]
    assert np.isnan(col.iloc[0])               # 静置段结束前未标定 → NaN
    assert col.iloc[-1] == pytest.approx(-1.781, rel=1e-12)


def test_max_age_days_expires_stale_value(spec):
    df = _mk_rest_series(days=7.0, n=169)
    # 拼接 30 天"行驶"（SOC 变化，温度变化），值应过期
    tail = pd.DataFrame({
        "t_h": df["t_h"].iloc[-1] + np.arange(1, 721, dtype=float),
        "volt": np.full(720, 3.9),
        "temp": np.full(720, 303.15),
        "soc": np.linspace(60, 40, 720),
    })
    both = pd.concat([df, tail], ignore_index=True)
    fresh = add_feature(both, spec, max_age_days=15.0)
    assert np.isnan(fresh["ocv_drift_45c_mv"].iloc[-1])


# ---------------------------------------------------------------- 守卫
def test_applicable_false_without_time_column(spec):
    df = pd.DataFrame({"volt": [4.0, 3.99], "temp": [300, 300], "soc": [60, 60]})
    ok, why = applicable(df, spec)
    assert ok is False
    assert "缺时间列" in why


def test_applicable_false_when_span_too_short(spec):
    df = _mk_rest_series(days=0.05, n=10)
    ok, why = applicable(df, spec)
    assert ok is False
    assert "时间跨度" in why


def test_extra_hard_rules_returns_empty_when_disabled(spec, capsys):
    acfg = {"b1_precursors": {"enabled": False, "spec_path": str(SPEC_PATH)}}
    assert extra_hard_rules(_mk_rest_series(), acfg) == []
    assert "未启用" in capsys.readouterr().out


def test_extra_hard_rules_skips_inapplicable_data(spec, capsys):
    """规格有效但数据没有静置段 → 空列表 + 明确原因，绝不产生 NaN 列。"""
    acfg = {"b1_precursors": {"enabled": True, "spec_path": str(SPEC_PATH)}}
    df = pd.DataFrame({"volt": [4.0, 3.99], "temp": [300, 300], "soc": [60, 60]})
    assert extra_hard_rules(df, acfg) == []
    assert "不适用" in capsys.readouterr().out


def test_extra_hard_rules_activates_on_rest_data(spec, capsys):
    acfg = {"b1_precursors": {"enabled": True, "spec_path": str(SPEC_PATH)}}
    rules = extra_hard_rules(_mk_rest_series(), acfg)
    assert len(rules) == 1 and rules[0]["threshold"] == pytest.approx(-1.425)
    assert "接入 1 条" in capsys.readouterr().out


# ---------------------------------------------------------------- 端到端
def test_rule_fires_on_lithium_plating_like_drift(spec):
    """把特征列喂给真实告警引擎：-1.781 mV 应越过 -1.425 阈值并触发。"""
    df = add_feature(_mk_rest_series(), spec)
    # 末尾补 5 步已标定状态，满足 persistence=3
    tail = df.iloc[[-1] * 5].copy()
    tail["t_h"] = df["t_h"].iloc[-1] + np.arange(1, 6, dtype=float)
    df = pd.concat([df, tail], ignore_index=True)

    ts = design_thresholds(
        pd.DataFrame({"ocv_drift_45c_mv": [0.0]}), [], hard_rules=hard_rules(spec)
    )
    res, events = run_alarms(df, ts, persistence=3, cooldown=10)
    assert np.isnan(res["ocv_drift_45c_mv"].iloc[0])       # 静置前无估计
    assert events["ALM_B1_OCV_DRIFT"], "越过 -1.425 mV 却没触发告警"
    assert events["ALM_B1_OCV_DRIFT"][0].peak == pytest.approx(-1.781, rel=1e-12)


def test_healthy_drift_does_not_fire(spec):
    """健康状态（-0.443 mV，45 ℃ 下不放大）不应触发。"""
    df = add_feature(_mk_rest_series(t_rest_c=45.0, dv_mv=-0.443), spec)
    tail = df.iloc[[-1] * 5].copy()
    tail["t_h"] = df["t_h"].iloc[-1] + np.arange(1, 6, dtype=float)
    df = pd.concat([df, tail], ignore_index=True)

    ts = design_thresholds(
        pd.DataFrame({"ocv_drift_45c_mv": [0.0]}), [], hard_rules=hard_rules(spec)
    )
    _, events = run_alarms(df, ts, persistence=3, cooldown=10)
    assert events["ALM_B1_OCV_DRIFT"] == []
