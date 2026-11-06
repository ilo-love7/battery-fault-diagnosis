"""B1 析锂前兆 → A1 告警规则的桥接。

设计边界（重要）：
- **只读 B1 的规格 JSON**，不在 A1 里重算 B1 的电化学模型。规格是 B1 的交付物
  （`b1-aging-attribution/results/a1/precursor_spec.json`），本模块把它变成
  `design_thresholds(hard_rules=...)` 能吃的字典。
- **只接 hard 规则**。B1 给的是"物理兜底阈值 = 标称值的 80%"，不是健康段分位数；
  分位数定标仍由 A1 在车端健康数据上做（`design_thresholds` 的 quantile 分支）。
- **静置段是硬前提**。`ocv_drift_45c_mv` 的定义是"静置 7 天的开路电压漂移"，
  没有 ≥7 天静置段的数据集上它根本不存在。本模块用 `applicable()` 显式守卫：
  不满足就返回 False + 原因，调用方跳过该规则，**绝不产生 NaN 列去喂告警引擎**
  （NaN 在 `_crossed` 里恒为 False，看似安全，但会让 `seg_vals.min()` 之类的
  统计悄悄变成 NaN，属于会埋雷的写法）。

A1 现有两个数据集（Mendeley 1152 行、CH-BatteryGen 每段 ≤400 行 / 10 s 采样）
都没有 7 天静置段，所以本模块在现有评估里**默认不启用**（config 的
`alarm.b1_precursors.enabled: false`）。接入路径通了，等有带静置段的数据
（例如实车停车段、A2 的静置标定数据）再打开即可。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

T_REF_C = 45.0                 # spec 的参考温度
T_REF_K = T_REF_C + 273.15

# spec 必须带的字段（缺一个就说明 B1 的版本和本模块对不上）
SPEC_REQUIRED = (
    "source", "ea_over_R_K", "reference_temp_C", "rest_days", "precursors",
)

# 静置段判定默认阈值（对应 spec 的 preconditions 前两条）
SOC_TOL_PCT = 2.0
TEMP_TOL_K = 1.0
CURRENT_TOL_C = 0.02           # 0.02C 以下视作静置（第三条 preconditions）


# --------------------------------------------------------------------------
# 规格读取与校验
# --------------------------------------------------------------------------
def load_spec(path: str | Path) -> dict:
    """读 B1 的 precursor_spec.json，并做字段校验。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"找不到 B1 前兆规格：{p}\n"
            "  生成方式：cd b1-aging-attribution && python src/to_a1.py"
        )
    spec = json.loads(p.read_text(encoding="utf-8"))
    missing = [k for k in SPEC_REQUIRED if k not in spec]
    if missing:
        raise ValueError(f"B1 规格缺字段 {missing}，与当前 A1 桥接版本不兼容：{p}")
    if not spec["precursors"]:
        raise ValueError(f"B1 规格里 precursors 为空：{p}")
    return spec


def check_spec(spec: dict) -> list[str]:
    """返回规格的问题清单（空列表 = 通过）。只报问题，不改数据。"""
    issues: list[str] = []
    if abs(float(spec["reference_temp_C"]) - T_REF_C) > 1e-6:
        issues.append(
            f"参考温度 {spec['reference_temp_C']} ℃ ≠ 桥接约定的 {T_REF_C} ℃，"
            "Arrhenius 归一的分母会错"
        )
    if float(spec["ea_over_R_K"]) <= 0:
        issues.append("ea_over_R_K 必须为正")
    for pr in spec["precursors"]:
        for key in ("name", "column", "direction", "kind", "threshold"):
            if key not in pr:
                issues.append(f"{pr.get('name', '<无名>')} 缺 {key}")
        if pr.get("direction") not in ("high", "low"):
            issues.append(f"{pr.get('name')} 的 direction 必须是 high/low")
        if pr.get("kind") not in ("hard", "quantile"):
            issues.append(f"{pr.get('name')} 的 kind 必须是 hard/quantile")
    return issues


def hard_rules(spec: dict) -> list[dict]:
    """只取 kind=="hard" 的前兆，转成 design_thresholds 的 hard_rules 格式。"""
    return [
        {
            "name": pr["name"],
            "column": pr["column"],
            "direction": pr["direction"],
            "threshold": float(pr["threshold"]),
        }
        for pr in spec["precursors"]
        if pr.get("kind") == "hard"
    ]


# --------------------------------------------------------------------------
# 静置段识别
# --------------------------------------------------------------------------
def find_rest_windows(
    df: pd.DataFrame,
    rest_days: float,
    time_col: str = "t_h",
    volt_col: str = "volt",
    temp_col: str = "temp",
    soc_col: str = "soc",
    curr_col: str | None = None,
    soc_tol_pct: float = SOC_TOL_PCT,
    temp_tol_k: float = TEMP_TOL_K,
    current_tol_c: float = CURRENT_TOL_C,
) -> list[tuple[int, int]]:
    """找出所有满足 spec preconditions 的静置段，返回 [(i0, i1)]（含端点）。

    判定（对应 spec 的 3 条 preconditions）：
      1. 时长 ≥ rest_days；
      2. 段内 SOC 极差 ≤ soc_tol_pct，且无电流事件（有 curr_col 时 |I| ≤ current_tol_c）；
      3. 段内温度极差 ≤ temp_tol_k。
    """
    need = [time_col, volt_col, temp_col]
    lack = [c for c in need if c not in df.columns]
    if lack:
        raise KeyError(f"静置段识别缺列 {lack}；现有列：{list(df.columns)}")

    t = df[time_col].to_numpy(dtype=float)
    temp = df[temp_col].to_numpy(dtype=float)
    if np.isnan(t).any():
        raise ValueError(f"{time_col} 含 NaN，无法判定静置时长")
    if not np.all(np.diff(t) >= 0):
        raise ValueError(f"{time_col} 必须单调不减（当前不满足，先按时间排序）")

    soc = df[soc_col].to_numpy(dtype=float) if soc_col in df.columns else None
    curr = (
        df[curr_col].to_numpy(dtype=float)
        if curr_col is not None and curr_col in df.columns
        else None
    )

    n = len(df)
    out: list[tuple[int, int]] = []
    i0 = 0
    while i0 < n:
        i1 = i0
        # 贪心向右扩：每扩一步检查三条约束是否仍成立
        while i1 + 1 < n:
            j = i1 + 1
            soc_rng = (
                float(np.nanmax(soc[i0:j + 1]) - np.nanmin(soc[i0:j + 1]))
                if soc is not None else 0.0
            )
            if soc_rng > soc_tol_pct:
                break
            t_rng = float(np.nanmax(temp[i0:j + 1]) - np.nanmin(temp[i0:j + 1]))
            if t_rng > temp_tol_k:
                break
            if curr is not None and np.nanmax(np.abs(curr[i0:j + 1])) > current_tol_c:
                break
            i1 = j
        if t[i1] - t[i0] >= rest_days * 24.0 - 1e-9:
            out.append((i0, i1))
            i0 = i1 + 1
        else:
            i0 += 1
    return out


def applicable(
    df: pd.DataFrame,
    spec: dict,
    time_col: str = "t_h",
    **kw,
) -> tuple[bool, str]:
    """能不能在 df 上算 B1 前兆？返回 (可/否, 原因)。原因直接可打印。"""
    if time_col not in df.columns:
        return False, (
            f"缺时间列 `{time_col}`：B1 前兆的定义是「静置 {spec['rest_days']:g} 天的"
            "OCV 漂移」，没有绝对时间就无法判定静置时长"
        )
    if "volt" not in df.columns:
        return False, "缺 `volt` 列"
    try:
        wins = find_rest_windows(df, float(spec["rest_days"]), time_col=time_col, **kw)
    except (KeyError, ValueError) as exc:
        return False, f"静置段识别失败：{exc}"
    if not wins:
        span_h = float(df[time_col].iloc[-1] - df[time_col].iloc[0])
        return False, (
            f"没有满足前置条件的静置段（需 ≥{spec['rest_days']:g} 天、SOC 极差 "
            f"≤{SOC_TOL_PCT:g}%、温度极差 ≤{TEMP_TOL_K:g} K）；"
            f"本数据时间跨度仅 {span_h:.1f} h"
        )
    return True, f"找到 {len(wins)} 个合格静置段"


# --------------------------------------------------------------------------
# 特征计算
# --------------------------------------------------------------------------
def arrhenius_to_ref(dv_mv: float, t_rest_k: float, ea_over_r: float) -> float:
    """把实测温度下的漂移归一到 45 ℃。

    dV45 = dV(T) * exp(-Ea/R * (1/T_ref - 1/T))
    T = T_ref 时系数为 1；T < T_ref（更冷）时系数 > 1，把冷态的慢漂移放大到 45 ℃ 等效值。
    """
    if t_rest_k <= 0:
        raise ValueError(f"温度必须是开尔文正值，收到 {t_rest_k}")
    return float(dv_mv) * float(
        np.exp(-float(ea_over_r) * (1.0 / T_REF_K - 1.0 / float(t_rest_k)))
    )


def ocv_drift_45c_mv(
    df: pd.DataFrame,
    spec: dict,
    i0: int,
    i1: int,
    time_col: str = "t_h",
    volt_col: str = "volt",
    temp_col: str = "temp",
) -> tuple[float, float, float]:
    """单个静置段的 (归一漂移 mV, 原始漂移 mV, 段平均温度 K)。"""
    v0 = float(df[volt_col].iloc[i0])
    v1 = float(df[volt_col].iloc[i1])
    dv_mv = 1000.0 * (v1 - v0)
    t_mean = float(df[temp_col].iloc[i0:i1 + 1].mean())
    dv45 = arrhenius_to_ref(dv_mv, t_mean, float(spec["ea_over_R_K"]))
    return dv45, dv_mv, t_mean


def add_feature(
    df: pd.DataFrame,
    spec: dict,
    time_col: str = "t_h",
    max_age_days: float | None = None,
    **kw,
) -> pd.DataFrame:
    """追加 `ocv_drift_45c_mv` 列：每个静置段结束点写入该段的归一漂移值。

    因果性：值只在静置段**结束后**才可得（BMS 不可能在静置中途知道 7 天后的电压），
    所以只在 i1 处赋值，然后前向填充——语义是「最近一次静置标定得到的析锂前兆估计」，
    与 A1「只用当前与过去」的原则一致。

    max_age_days：估计值的有效期（天）。超过后置 NaN（不参与告警）。None = 永久有效。
    """
    ok, why = applicable(df, spec, time_col=time_col, **kw)
    if not ok:
        raise ValueError(f"该数据不支持 B1 前兆计算：{why}")

    out = df.copy()
    col = "ocv_drift_45c_mv"
    out[col] = np.nan
    wins = find_rest_windows(out, float(spec["rest_days"]), time_col=time_col, **kw)
    for i0, i1 in wins:
        dv45, _, _ = ocv_drift_45c_mv(
            out, spec, i0, i1, time_col=time_col,
            volt_col=kw.get("volt_col", "volt"), temp_col=kw.get("temp_col", "temp"),
        )
        out.loc[out.index[i1], col] = dv45

    out[col] = out[col].ffill()

    if max_age_days is not None:
        # 记录每个位置"最近一次标定"的时间戳，超过有效期就清空
        t = out[time_col].to_numpy(dtype=float)
        stamp = np.full(len(out), np.nan)
        for i0, i1 in wins:
            stamp[i1] = t[i1]
        stamp_s = pd.Series(stamp).ffill().to_numpy()
        stale = (t - stamp_s) > max_age_days * 24.0
        out.loc[stale, col] = np.nan
    return out


# --------------------------------------------------------------------------
# 一键接入
# --------------------------------------------------------------------------
def extra_hard_rules(
    df: pd.DataFrame,
    acfg: dict,
    time_col: str = "t_h",
    verbose: bool = True,
) -> list[dict]:
    """从 A1 配置的 `alarm.b1_precursors` 段取规则；不适用时返回 []。

    acfg 是 config.yaml 的 `alarm` 段。返回的规则可直接拼给
    `design_thresholds(..., hard_rules=make_hard(acfg) + extra_hard_rules(...))`。
    """
    bcfg = acfg.get("b1_precursors") or {}
    if not bcfg.get("enabled", False):
        if verbose:
            print("  [B1] 未启用（alarm.b1_precursors.enabled=false），跳过析锂前兆规则")
        return []

    spec_path = bcfg.get("spec_path")
    if not spec_path:
        raise ValueError("alarm.b1_precursors.enabled=true 但没给 spec_path")
    spec_path = Path(spec_path)
    if not spec_path.is_absolute():
        # 与 config.py 一致：相对路径以仓库根为基准
        from evfd.config import ROOT
        spec_path = (ROOT / spec_path).resolve()

    spec = load_spec(spec_path)
    issues = check_spec(spec)
    if issues:
        raise ValueError("B1 规格校验失败：\n  - " + "\n  - ".join(issues))

    ok, why = applicable(df, spec, time_col=time_col)
    if not ok:
        if verbose:
            print(f"  [B1] 规格有效但本数据不适用，跳过：{why}")
        return []

    rules = hard_rules(spec)
    if verbose:
        print(f"  [B1] 接入 {len(rules)} 条硬规则（{why}）：")
        for r in rules:
            print(f"        {r['name']}: {r['column']} {r['direction']} "
                  f"{r['threshold']:+.3f} ({spec['source']})")
    return rules
