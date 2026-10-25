"""04 告警阈值设计：健康包络分位数定标 + 物理硬规则 → 误报/检出/延迟。

主方案：全健康块定标（BMS 包络标定惯例；健康块内误报为 in-sample，如实报告），
故障块上评估检出率与延迟（out-of-sample）。
敏感性：前 60% 定标 + 后 40% 验证 —— 展示"单轨迹欠覆盖"问题（健康深放电
瞬态 430K 不在前 60% 包络内 → 后段误报），引出跨库定标的必要性。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evfd.alarm.thresholds import (
    design_thresholds,
    evaluate_alarm_performance,
    run_alarms,
)
from evfd.config import load_config
from evfd.data.load_mendeley import CLASS_NAMES, load_multi
from evfd.evaluate.metrics import save_metrics

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

SPECS = [
    {"name": "ALM_OT", "column": "temp", "direction": "high"},     # 过温
    {"name": "ALM_TR", "column": "d_temp3", "direction": "high"},  # 温升速率
    {"name": "ALM_UV", "column": "volt", "direction": "low"},      # 欠压预警
    {"name": "ALM_VD", "column": "d_volt3", "direction": "low"},   # 电压骤降
    {"name": "ALM_SOC", "column": "soc", "direction": "low"},      # SoC 过低
]


def make_hard(acfg: dict) -> list[dict]:
    return [
        {"name": "ALM_ISC", "column": "volt", "direction": "low",
         "threshold": acfg["short_voltage_v"]},      # 电压塌陷 → 内短路
        {"name": "ALM_OD", "column": "soc", "direction": "low",
         "threshold": acfg["soc_overdischarge"]},    # SoC<0 → 过放电
    ]


def main() -> None:
    cfg = load_config()
    acfg = cfg["alarm"]
    mdir = Path(cfg["paths"]["metrics_dir"])
    fdir = Path(cfg["paths"]["figures_dir"])
    for d in (mdir, fdir):
        d.mkdir(parents=True, exist_ok=True)

    md = load_multi(cfg["paths"]["mendeley_multi"])
    df = md.df.copy()
    df["d_temp3"] = df["temp"].diff(3).fillna(0.0)
    df["d_volt3"] = df["volt"].diff(3).fillna(0.0)

    h_start, h_end, _ = md.blocks[0]
    healthy = df.iloc[h_start:h_end]

    results = {}
    ts_main = None
    events_main = None
    df_res_main = None

    for variant in ("full_envelope", "calib60"):
        if variant == "full_envelope":
            calib = healthy
        else:
            calib = healthy.iloc[: int(len(healthy) * 0.6)]
        ts = design_thresholds(
            calib, SPECS, quantile=acfg["healthy_quantile"], hard_rules=make_hard(acfg)
        )
        df_res, events = run_alarms(
            df, ts, persistence=acfg["persistence"], cooldown=acfg["cooldown"]
        )
        # 检出评估只对故障块（健康块是定标块，其越限计入误报统计）
        perf = evaluate_alarm_performance(df_res, events, md.blocks[1:], CLASS_NAMES)
        # 健康段误报：
        if variant == "full_envelope":
            mask = np.zeros(len(df_res), dtype=bool)
            mask[h_start:h_end] = True
            note = "in-sample（全健康块定标，同一块统计误报）"
            eval_steps = int(mask.sum())
            fa_window = (h_start, h_end)
        else:
            calib_end = h_start + int((h_end - h_start) * 0.6)
            mask = np.zeros(len(df_res), dtype=bool)
            mask[calib_end:h_end] = True
            note = "out-of-sample（前60%定标，其后40%统计误报，含深放电热瞬态）"
            eval_steps = int(mask.sum())
            fa_window = (calib_end, h_end)
        perf["healthy"]["step_false_alarm_rate"] = float(df_res["alarm_any"].values[mask].mean())
        perf["healthy"]["eval_steps"] = eval_steps
        perf["healthy"]["note"] = note
        perf["healthy"]["healthy_events_in_block"] = int(sum(
            1 for evs in events.values() for e in evs
            if fa_window[0] <= e.start < fa_window[1]
        ))
        perf["thresholds"] = ts.to_dict()
        results[variant] = perf
        if variant == "full_envelope":
            ts_main, events_main, df_res_main = ts, events, df_res

    print("== 主方案：全健康块包络定标 ==")
    for name, info in ts_main.to_dict().items():
        print(f"  {name}: {info['column']} {info['direction']} "
              f"{info['threshold']:.4f} ({info['kind']})")
    hm = results["full_envelope"]["healthy"]
    print(f"  健康块误报率={hm['step_false_alarm_rate']:.4f} "
          f"({hm['note']}; 健康块内事件 {hm['healthy_events_in_block']} 个)")
    for blk in results["full_envelope"]["fault_blocks"]:
        print(f"  {blk['class']}: 检出={blk['detected']} "
              f"延迟={blk['latency_steps']} 步 规则={blk['rules_fired']}")

    print("\n== 敏感性：前60%定标 ==")
    hm2 = results["calib60"]["healthy"]
    print(f"  定标段后40%内事件数={hm2['healthy_events_in_block']} "
          f"误报率={hm2['step_false_alarm_rate']:.4f} （单轨迹欠覆盖演示）")

    out = {
        "main": results["full_envelope"],
        "sensitivity_calib60": results["calib60"],
        "settings": {
            "persistence": acfg["persistence"], "cooldown": acfg["cooldown"],
            "healthy_quantile": acfg["healthy_quantile"],
            "sampling_note": "行序=时间步（数据未给绝对时间）",
        },
    }
    save_metrics(out, mdir / "alarm_design.json")

    # ---- 图：告警时间线（主方案） ----
    colors = {0: "#2a9d8f", 1: "#e9c46a", 2: "#e76f51"}
    rule_colors = {
        "ALM_OT": "#d62828", "ALM_TR": "#f77f00", "ALM_UV": "#4361ee",
        "ALM_VD": "#7209b7", "ALM_SOC": "#0a9396", "ALM_ISC": "#9d0208",
        "ALM_OD": "#bc4749",
    }
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for ax, col, ylab in zip(
        axes, ("volt", "temp", "soc"), ("电压 (V)", "温度 (K)", "SoC (%)")
    ):
        for a, b, lab in md.blocks:
            ax.plot(df.index[a:b], df[col].iloc[a:b], color=colors[lab], lw=1.0)
        ax.set_ylabel(ylab)
    for r, evs in events_main.items():
        for e in evs:
            for ax in axes:
                ax.axvspan(e.start, e.end, color=rule_colors.get(r, "gray"), alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.4) for c in rule_colors.values()]
    axes[0].legend(handles, rule_colors.keys(), fontsize=7, ncol=4, loc="upper left")
    axes[0].set_title("告警事件时间线（主方案：全健康块包络定标；色带=告警事件）")
    axes[-1].set_xlabel("行序（时间步）")
    fig.tight_layout()
    fig.savefig(fdir / "fig04_告警时间线.png", dpi=160)
    plt.close(fig)

    # ---- 图：阈值与三类轨迹（含阈值线） ----
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    thr_map = ts_main.to_dict()
    for ax, col, ylab in zip(
        axes, ("volt", "temp", "soc"), ("电压 (V)", "温度 (K)", "SoC (%)")
    ):
        for a, b, lab in md.blocks:
            ax.plot(df.index[a:b], df[col].iloc[a:b], color=colors[lab], lw=1.0)
        ax.set_ylabel(ylab)
    for rname in ("ALM_UV", "ALM_ISC"):
        if rname in thr_map:
            axes[0].axhline(thr_map[rname]["threshold"], ls="--", lw=1,
                            color=rule_colors[rname], label=rname)
    for rname in ("ALM_OT", "ALM_TR"):
        if rname in thr_map:
            axes[1].axhline(thr_map[rname]["threshold"], ls="--", lw=1,
                            color=rule_colors[rname], label=rname)
    for rname in ("ALM_SOC", "ALM_OD"):
        if rname in thr_map:
            axes[2].axhline(thr_map[rname]["threshold"], ls="--", lw=1,
                            color=rule_colors[rname], label=rname)
    for ax in axes:
        ax.legend(fontsize=7, ncol=3, loc="upper left")
    axes[0].set_title("告警阈值 vs 三类轨迹（绿=健康 / 黄=过放电 / 红=短路）")
    axes[-1].set_xlabel("行序（时间步）")
    fig.tight_layout()
    fig.savefig(fdir / "fig03_阈值线.png", dpi=160)
    plt.close(fig)

    print("\n告警设计完成。")


if __name__ == "__main__":
    main()
