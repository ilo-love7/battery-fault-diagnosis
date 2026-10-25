"""告警引擎与阈值设计（项目 A1 核心交付物之一）。

设计原则（车端 BMS 可实现性）：
- 所有指标只依赖当前与过去（因果、流式）；
- 阈值在「健康数据分位数」下定标，目标 = 控制健康段步级误报率；
- 越限需连续 persistence 步才触发（抗单点抖动），触发后 cooldown 步内不重复计事件；
- 物理硬规则（电压塌陷、SoC<0）独立于分位数，作兜底。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Rule:
    name: str
    column: str          # 指标列名（df 中已存在）
    direction: str       # "high" 指标越大越糟 / "low" 越小越糟
    threshold: float
    kind: str = "quantile"   # quantile | hard


@dataclass
class AlarmEvent:
    rule: str
    start: int
    end: int             # 含
    peak: float


@dataclass
class ThresholdSet:
    rules: list[Rule] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            r.name: {
                "column": r.column, "direction": r.direction,
                "threshold": float(r.threshold), "kind": r.kind,
            }
            for r in self.rules
        }


def design_thresholds(
    healthy: pd.DataFrame,
    specs: list[dict],            # [{name, column, direction}]
    quantile: float = 0.995,
    hard_rules: list[dict] | None = None,   # [{name, column, direction, threshold}]
) -> ThresholdSet:
    """健康段分位数定标 + 物理硬规则。

    direction="high"：阈值 = 健康指标 q 分位（越限=超过它）；
    direction="low" ：阈值 = 健康指标 1-q 分位（越限=低于它）。
    """
    rules: list[Rule] = []
    for s in specs:
        col = healthy[s["column"]].dropna().values
        if s["direction"] == "high":
            thr = float(np.quantile(col, quantile))
        else:
            thr = float(np.quantile(col, 1.0 - quantile))
        rules.append(Rule(s["name"], s["column"], s["direction"], thr, "quantile"))
    for h in hard_rules or []:
        rules.append(
            Rule(h["name"], h["column"], h["direction"], float(h["threshold"]), "hard")
        )
    return ThresholdSet(rules)


def _crossed(values: np.ndarray, rule: Rule) -> np.ndarray:
    if rule.direction == "high":
        return values > rule.threshold
    return values < rule.threshold


def run_alarms(
    df: pd.DataFrame,
    ts: ThresholdSet,
    persistence: int = 3,
    cooldown: int = 10,
) -> tuple[pd.DataFrame, dict[str, list[AlarmEvent]]]:
    """沿行序流式跑全部规则。

    返回 (逐步结果 df, {rule: [AlarmEvent]})。
    逐步结果 df 增加 {rule}_cross（当步越限）、{rule}_trig（持续 persistence 步触发）、
    alarm_any（任一规则触发）。
    """
    out = df.copy()
    events: dict[str, list[AlarmEvent]] = {}
    n = len(out)
    any_trig = np.zeros(n, dtype=bool)
    for rule in ts.rules:
        cross = _crossed(out[rule.column].values, rule)
        # 连续 persistence 步越限才触发
        trig = np.zeros(n, dtype=bool)
        run = 0
        for i in range(n):
            run = run + 1 if cross[i] else 0
            trig[i] = run >= persistence
        # 事件化：触发段为事件；触发中断后若越限在 cooldown 内延续则并入同一事件
        evs: list[AlarmEvent] = []
        i = 0
        while i < n:
            if not trig[i]:
                i += 1
                continue
            j, k = i, i
            while k < n:
                if trig[k]:
                    j, k = k, k + 1
                elif k - j <= cooldown and cross[k]:
                    k += 1
                else:
                    break
            seg_vals = out[rule.column].values[i:j + 1]
            peak = float(seg_vals.max() if rule.direction == "high" else seg_vals.min())
            evs.append(AlarmEvent(rule.name, i, j, peak))
            i = j + cooldown + 1
        out[f"{rule.name}_cross"] = cross
        out[f"{rule.name}_trig"] = trig
        any_trig |= trig
        events[rule.name] = evs
    out["alarm_any"] = any_trig
    return out, events


def evaluate_alarm_performance(
    df: pd.DataFrame,
    events: dict[str, list[AlarmEvent]],
    blocks: list[tuple[int, int, int]],
    class_names: dict[int, str],
    label_col: str = "label",
) -> dict:
    """健康段误报 + 各故障块检出率/延迟 + 规则×类别混淆。"""
    res: dict = {"healthy": {}, "fault_blocks": [], "rule_class_matrix": {}}
    healthy_mask = (df[label_col] == 0).values
    alarm_mask = df["alarm_any"].values
    res["healthy"] = {
        "steps": int(healthy_mask.sum()),
        "step_false_alarm_rate": float(alarm_mask[healthy_mask].mean()) if healthy_mask.any() else 0.0,
        "n_events": int(sum(len(evs) for evs in events.values())),
        "events_per_rule": {r: len(evs) for r, evs in events.items()},
    }
    rules = list(events.keys())
    matrix = {r: {c: 0 for c in class_names.values()} for r in rules}
    for start, end, lab in blocks:
        name = class_names[lab]
        seg = df.iloc[start:end]
        fired = {r: [e for e in events[r] if e.start >= start and e.start < end] for r in rules}
        detected_any = any(len(f) > 0 for f in fired.values())
        lat = None
        if detected_any:
            firsts = [e.start for f in fired.values() for e in f]
            lat = int(min(firsts) - start)
        res["fault_blocks"].append({
            "class": name, "start": start, "end": end, "length": end - start,
            "detected": bool(detected_any), "latency_steps": lat,
            "rules_fired": [r for r in rules if len(fired[r]) > 0],
            "n_events_per_rule": {r: len(fired[r]) for r in rules},
        })
        for r in rules:
            if fired[r]:
                matrix[r][name] += 1
    res["rule_class_matrix"] = matrix
    return res
