"""07 故障明细深挖（v1.3）：单体定位 + 故障分级 + 严重度分层检出。

数据源：faults.details.-.LFP.batteries.xlsx（100 辆 LFP 故障车：
vid / Fault type / Severity / Fault cell id(1-based) / Fault Value）。

三个任务：
A) 故障单体定位：按物理机理先验定方向——自放电→最深单体（median−V 最大）、
   高内阻/低容量→最高单体（V−median 最大，充电平台抬高/提前截止）；
   评 top-1 / top-3 命中率（方向先验来自论文 Fig.2，属先验而非调参）。
B) 故障分级（对齐论文 Task 2）：车级特征 → 轻/中/重 3 类，5 折分层 CV 的 XGB。
C) 严重度分层检出：v1.1 两级告警车级 flag × 严重度交叉表。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evfd.config import load_config
from evfd.data.chsample import select_segment_files
from evfd.data.load_chbatterygen import add_pack_indicators, load_segment
from evfd.evaluate.metrics import save_metrics
from evfd.models.classical import train_xgb

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
SEVERITY_MAP = {"Mild": 0, "Moderate": 1, "Severe": 2}


def cell_dev_profile(root: str, vin: str, files: list[Path],
                     mode: str = "discharge", window=(0.1, 0.7)) -> np.ndarray:
    """单体偏差画像 [M]：放电/充电段内，SOC 相对进度窗口上的 (V_cell − 包中位数) 均值。

    window 避开段首尾（截止/收敛段全体趋同，稀释故障偏差）。
    """
    profs = []
    for p in files:
        if ("discharge" in p.name.lower()) != (mode == "discharge"):
            continue
        df = load_segment(p, 400)
        soc = df["SOC"].values.astype(float)
        rng = soc.max() - soc.min()
        prog = (soc - soc.min()) / rng if rng > 1e-6 else np.linspace(0, 1, len(soc))
        mask = (prog >= window[0]) & (prog <= window[1])
        if mask.sum() < 10:
            mask = np.ones(len(df), dtype=bool)
        V = df[[c for c in df.columns if c.startswith("VOLT_")]].values.astype(float)[mask]
        profs.append((V - np.median(V, axis=1, keepdims=True)).mean(axis=0))
    return np.mean(profs, axis=0) if profs else None


def eval_localization(det: pd.DataFrame, root: str, mode: str,
                      direction: str, base: int, window=(0.1, 0.7)) -> pd.DataFrame:
    """方向×编号制式下的定位评测。

    编号约定（诊断矩阵实证）：xlsx 的 Fault cell id 为 **0-based**（VOLT_N 为
    1-based），即 0-based 索引 = cell id；base=0 即直接以 cell id 作 0-based 索引。
    """
    rows = []
    for _, r in det.iterrows():
        if r["cell"] == 0:
            continue
        files = []
        for ft in ("self_discharge", "high_resistance", "low_capacity"):
            cand = select_segment_files(root, "LFP", ft, r["vin"])
            if cand:
                files = cand
                break
        prof = cell_dev_profile(root, r["vin"], files, mode=mode, window=window)
        if prof is None:
            continue
        true_cell = int(r["cell"]) + base           # 转 0-based 索引
        if true_cell >= len(prof) or true_cell < 0:
            continue
        order = (np.argsort(prof) if direction == "deepest" else np.argsort(-prof))
        rows.append({"vin": r["vin"], "ftype": r["ftype"], "sev": r["sev"],
                     "M": len(prof), "true0": true_cell,
                     "top1": int(order[0] == true_cell),
                     "top3": int(true_cell in order[:3].tolist())})
    return pd.DataFrame(rows)


def eval_bidirectional(det: pd.DataFrame, root: str, base: int = 0) -> pd.DataFrame:
    """双向极值打分：score_c = 放电最深深度 + 充电最高抬升（故障单体双向偏离包中位数）。"""
    rows = []
    for _, r in det.iterrows():
        if r["cell"] == 0:
            continue
        files = []
        for ft in ("self_discharge", "high_resistance", "low_capacity"):
            cand = select_segment_files(root, "LFP", ft, r["vin"])
            if cand:
                files = cand
                break
        dprof = cell_dev_profile(root, r["vin"], files, mode="discharge")
        cprof = cell_dev_profile(root, r["vin"], files, mode="charge")
        if dprof is None and cprof is None:
            continue
        n = max(len(dprof) if dprof is not None else 0,
                len(cprof) if cprof is not None else 0)
        deep = -dprof if dprof is not None else np.zeros(n)
        up = cprof if cprof is not None else np.zeros(n)
        deep = np.pad(deep, (0, n - len(deep)))
        up = np.pad(up, (0, n - len(up)))
        score = deep + up
        true_cell = int(r["cell"]) + base
        if true_cell >= n or true_cell < 0:
            continue
        order = np.argsort(-score)
        rows.append({"vin": r["vin"], "ftype": r["ftype"], "sev": r["sev"],
                     "M": n, "true0": true_cell,
                     "top1": int(order[0] == true_cell),
                     "top3": int(true_cell in order[:3].tolist())})
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    seed = cfg["splits"]["random_state"]
    mdir = Path(cfg["paths"]["metrics_dir"])
    fdir = Path(cfg["paths"]["figures_dir"])
    for d in (mdir, fdir):
        d.mkdir(parents=True, exist_ok=True)
    root = cfg["paths"]["chbatterygen_root"]
    xlsx = cfg["paths"]["chbatterygen_faults_xlsx"]

    det = pd.read_excel(xlsx)
    det = det.rename(columns={
        "vid": "vin", "Fault type": "ftype", "Severity": "sev",
        "Fault cell id": "cell", "Fault Value": "fval"})
    det["sev_id"] = det["sev"].map(SEVERITY_MAP)
    print(f"明细 {len(det)} 行（{det.ftype.value_counts().to_dict()}）")

    # ---------- A) 单体定位：主先验 = 放电段最深单体（论文 Fig.2 e-h）+ 0-based 编号
    # 约定（诊断矩阵实证：xlsx 的 Fault cell id 为 0-based，VOLT_N 为 1-based） ----------
    configs = [
        ("discharge_deepest_base0", "discharge", "deepest", 0, (0.1, 0.7)),   # 主先验
        ("discharge_deepest_base1", "discharge", "deepest", 1, (0.1, 0.7)),
        ("charge_deepest_base0", "charge", "deepest", 0, (0.1, 0.7)),
        ("discharge_highest_base0", "discharge", "highest", 0, (0.1, 0.7)),
        ("charge_highest_base0", "charge", "highest", 0, (0.1, 0.7)),
        # 诊断：全窗口含放电末端（低容量"提前截止"的偏差集中区）
        ("discharge_deepest_full_base0", "discharge", "deepest", 0, (0.0, 1.0)),
    ]
    diag = {}
    loc = None
    for name, mode, direction, base, window in configs:
        df_loc = eval_localization(det, root, mode, direction, base, window=window)
        diag[name] = {
            "n": int(len(df_loc)),
            "top1": float(df_loc.top1.mean()) if len(df_loc) else None,
            "top3": float(df_loc.top3.mean()) if len(df_loc) else None,
        }
        if name == "discharge_deepest_base0":
            loc = df_loc
    # 双向极值打分变体：故障单体充电偏高 + 放电偏低，双向偏离取和
    bi = eval_bidirectional(det, root, base=0)
    diag["bidirectional_base0"] = {
        "n": int(len(bi)), "top1": float(bi.top1.mean()), "top3": float(bi.top3.mean())}
    save_metrics(diag, mdir / "fault_localization_diag.json")
    print("\n== A) 定位诊断矩阵（方向 × 模式 × 编号制式；cell id 为 0-based） ==")
    for k, v in diag.items():
        print(f"  {k}: top1={v['top1']:.1%} top3={v['top3']:.1%} (n={v['n']})")

    loc.to_csv(mdir / "fault_localization.csv", index=False)
    loc_summary = {}
    for ft, g in loc.groupby("ftype"):
        loc_summary[ft] = {
            "n": int(len(g)), "top1": float(g.top1.mean()), "top3": float(g.top3.mean()),
            "by_severity": {s: {"n": int(len(gg)), "top1": float(gg.top1.mean()),
                                "top3": float(gg.top3.mean())}
                            for s, gg in g.groupby("sev")},
        }
    loc_summary["overall"] = {"n": int(len(loc)), "top1": float(loc.top1.mean()),
                              "top3": float(loc.top3.mean())}
    save_metrics(loc_summary, mdir / "fault_localization.json")
    print("-- 主先验（放电段最深单体，0-based 编号）分类型：")
    for ft, s in loc_summary.items():
        if ft == "overall":
            continue
        print(f"  {ft}: top1={s['top1']:.1%} top3={s['top3']:.1%} (n={s['n']})")
    print(f"  overall: top1={loc_summary['overall']['top1']:.1%} "
          f"top3={loc_summary['overall']['top3']:.1%}")

    # ---------- B) 故障分级（轻/中/重，5 折分层 CV） ----------
    vf = pd.read_csv(mdir / "chbatterygen_vehicle_features.csv")
    lfp_fault = vf[(vf.chem == "LFP") & (vf.fault != "normal")].copy()
    mg = lfp_fault.merge(det[["vin", "sev", "sev_id", "ftype", "fval"]],
                         on="vin", how="inner", suffixes=("", "_det"))
    print(f"\n分级样本: {len(mg)}（对齐失败 {len(lfp_fault) - len(mg)}）")
    feat_cols = [c for c in vf.columns
                 if c not in ("chem", "vin", "fault", "is_design", "label")]
    X = mg[feat_cols].values
    y = mg.sev_id.values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=int)
    for tr, te in skf.split(X, y):
        clf = train_xgb(X[tr], y[tr], seed=seed)
        oof[te] = clf.predict(X[te])
    acc = float((oof == y).mean())
    wf1 = float(f1_score(y, oof, average="weighted", zero_division=0))
    conf = pd.crosstab(pd.Series(y, name="真"), pd.Series(oof, name="预测"))
    majority = max(np.bincount(y)) / len(y)
    grading = {
        "model": "xgb_vehicle_features", "cv": "5fold_stratified_oof",
        "n": int(len(y)), "accuracy": acc, "weighted_f1": wf1,
        "majority_baseline": float(majority),
        "confusion_rows_true_012": conf.values.tolist(),
        "per_type": {},
    }
    for ft, g in mg.groupby("ftype"):
        m = g.index.values
        grading["per_type"][ft] = {
            "n": int(len(g)), "accuracy": float((oof[m] == y[m]).mean())}
    save_metrics(grading, mdir / "fault_grading.json")
    print(f"== B) 故障分级（XGB 5 折 OOF）: acc={acc:.3f} wF1={wf1:.3f} "
          f"(多数类基线 {majority:.3f}) ==")
    print(conf.to_string())
    for ft, s in grading["per_type"].items():
        print(f"  {ft}: acc={s['accuracy']:.3f} (n={s['n']})")

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ConfusionMatrixDisplay.from_predictions(
        y, oof, labels=[0, 1, 2], display_labels=["轻 Mild", "中 Moderate", "重 Severe"],
        cmap="Blues", colorbar=False, ax=ax)
    ax.set_title("故障分级 · XGB 车级特征（5 折 OOF，LFP）", fontsize=10)
    fig.tight_layout()
    fig.savefig(fdir / "fig09_分级混淆矩阵.png", dpi=160)
    plt.close(fig)

    # ---------- C) 严重度分层检出（v1.1 告警 flag × 严重度） ----------
    al = pd.read_csv(mdir / "chbatterygen_alarm_segments.csv")
    al = al[(al.chem == "LFP") & (al.fault != "normal")
            & (al.variant == "V1_full_normal_pool")]
    veh_alarm = al.groupby("vin").agg(alarm=("alarm", "max")).reset_index()
    sev_det = veh_alarm.merge(det[["vin", "ftype", "sev"]], on="vin", how="inner")
    strat = {}
    for (ft, sv), g in sev_det.groupby(["ftype", "sev"]):
        strat.setdefault(ft, {})[sv] = {
            "n": int(len(g)), "alarm_rate": float(g.alarm.mean())}
    save_metrics(strat, mdir / "fault_severity_detection.json")
    print("\n== C) 严重度分层检出（V1 两级告警，车级） ==")
    for ft, d_ in strat.items():
        items = ", ".join(f"{s} {v['alarm_rate']:.0%}(n={v['n']})"
                          for s, v in d_.items())
        print(f"  {ft}: {items}")

    # ---------- 图：定位准确率条形 ----------
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    types = [t for t in loc_summary if t != "overall"]
    x = np.arange(len(types))
    w = 0.35
    t1 = [loc_summary[t]["top1"] for t in types]
    t3 = [loc_summary[t]["top3"] for t in types]
    ax.bar(x - w / 2, t1, w, label="Top-1", color="#2a6f97")
    ax.bar(x + w / 2, t3, w, label="Top-3", color="#a3cef1")
    ax.set_xticks(x, [t.replace(" ", "\n") for t in types])
    ax.set_ylabel("命中率")
    ax.set_ylim(0, 1.05)
    for i, (a, b) in enumerate(zip(t1, t3)):
        ax.text(i - w / 2, a + 0.02, f"{a:.0%}", ha="center", fontsize=9)
        ax.text(i + w / 2, b + 0.02, f"{b:.0%}", ha="center", fontsize=9)
    ax.axhline(1 / 27, ls=":", c="gray", lw=1)
    ax.text(len(types) - 0.45, 1 / 27 + 0.02, "随机(27串)", fontsize=8, color="gray")
    ax.set_title("故障单体定位（跨充电段偏差画像，方向先验）", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(fdir / "fig10_单体定位.png", dpi=160)
    plt.close(fig)

    print("\n故障明细深挖完成。")


if __name__ == "__main__":
    main()
