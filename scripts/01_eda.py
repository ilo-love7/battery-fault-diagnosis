"""01 EDA：Mendeley 三分类数据结构/量程/分布确认，产出图与摘要。"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evfd.config import load_config
from evfd.data.load_mendeley import CLASS_NAMES, load_multi, load_simple
from evfd.evaluate.metrics import save_metrics

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    cfg = load_config()
    fig_dir = Path(cfg["paths"]["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    md = load_multi(cfg["paths"]["mendeley_multi"])
    df = md.df
    print("== 三分类版结构 ==")
    print("blocks:", [(a, b, CLASS_NAMES[l]) for a, b, l in md.blocks])
    print(df.groupby("label").agg(["min", "max", "mean", "std"]).round(3).to_string())
    print("缺失值:", int(df.isna().sum().sum()))

    ms = load_simple(cfg["paths"]["mendeley_simple"])
    print("\n== 二分类版结构 ==")
    print("blocks:", [(a, b, l) for a, b, l in ms.blocks])

    summary = {
        "multi": {
            "rows": len(df),
            "blocks": [
                {"start": a, "end": b, "class": CLASS_NAMES[l]} for a, b, l in md.blocks
            ],
            "per_class_stats": {
                CLASS_NAMES[int(l)]: {
                    c: {
                        "min": float(g[c].min()), "max": float(g[c].max()),
                        "mean": float(g[c].mean()), "std": float(g[c].std()),
                    }
                    for c in ("soc", "temp", "volt")
                }
                for l, g in df.groupby("label")
            },
        },
        "simple": {
            "rows": len(ms.df),
            "blocks": [{"start": a, "end": b, "label": l} for a, b, l in ms.blocks],
        },
    }
    save_metrics(summary, Path(cfg["paths"]["metrics_dir"]) / "eda_mendeley.json")

    # 图 1：三类轨迹
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    colors = {0: "#2a9d8f", 1: "#e9c46a", 2: "#e76f51"}
    for a, b, lab in md.blocks:
        seg = df.iloc[a:b]
        for ax, col, ylab in zip(
            axes, ("volt", "temp", "soc"), ("电压 (V)", "温度 (K)", "SoC (%)")
        ):
            ax.plot(seg["step_in_block"] + a, seg[col], color=colors[lab], lw=1.2)
            ax.set_ylabel(ylab)
    for ax in axes:
        for a, b, lab in md.blocks:
            ax.axvspan(a, b, color=colors[lab], alpha=0.08)
    axes[0].set_title("Mendeley 三分类轨迹：绿=健康 0-423 / 黄=过放电 424-787 / 红=短路 788-1151")
    axes[-1].set_xlabel("行序（时间步）")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig01_三类轨迹.png", dpi=160)
    plt.close(fig)

    # 图 2：类内分布
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, col, title in zip(
        axes, ("volt", "temp", "soc"), ("电压 V", "温度 K", "SoC %")
    ):
        for lab, cname in CLASS_NAMES.items():
            ax.hist(df[df.label == lab][col], bins=40, alpha=0.55,
                    label=cname, color=colors[lab])
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig02_类分布.png", dpi=160)
    plt.close(fig)

    print("\nEDA 完成：图与摘要已写入 results/")


if __name__ == "__main__":
    main()
