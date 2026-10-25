"""诊断小工具（A1 交付物：可直接扩展为车端诊断工具）。

输入一段时序数据，自动识别格式，输出：
- Mendeley 三特征流（SoC/T/V）：逐点诊断（XGBoost）+ 流式告警事件 + 最终结论
- CH-BatteryGen 包级流（MAX/MIN_CELL_VOLT, VOLT_N…）：两级告警 + 段级四类诊断

用法：
  python tools/diagnostic_tool.py --input <csv|xlsx> [--chem NCM] [--mode charge] [--out out.csv]
  python tools/diagnostic_tool.py --demo      # 内置两个示例：Mendeley 短路段 + CH 故障车段
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evfd.alarm.thresholds import design_thresholds, run_alarms
from evfd.config import load_config
from evfd.data.features import POINT_FEATURES, add_point_features
from evfd.data.load_chbatterygen import segment_features
from evfd.models.classical import load_model

CLASS_NAMES = {0: "健康", 1: "过放电故障", 2: "短路故障"}


def load_alarm_thresholds(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------- Mendeley 三特征流 ----------------

def diagnose_mendeley_stream(df: pd.DataFrame, cfg: dict) -> dict:
    """df 需含列 soc/temp/volt（大小写不敏感），行序=时间步。"""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"temperature": "temp"})
    df["d_temp3"] = df["temp"].diff(3).fillna(0.0)
    df["d_volt3"] = df["volt"].diff(3).fillna(0.0)

    # 1) 流式告警（阈值来自 04 脚本设计的健康包络）
    alarm_info = load_alarm_thresholds(
        Path(cfg["paths"]["metrics_dir"]) / "alarm_design.json"
    )
    thr = alarm_info["main"]["thresholds"]
    rules = []
    for name, info in thr.items():
        rules.append(type("R", (), {
            "name": name, "column": info["column"], "direction": info["direction"],
            "threshold": info["threshold"], "kind": info["kind"],
        })())
    ts = type("TS", (), {"rules": rules})()
    res, events = run_alarms(
        df, ts, persistence=cfg["alarm"]["persistence"],
        cooldown=cfg["alarm"]["cooldown"],
    )

    # 2) XGBoost 逐点诊断
    feat = add_point_features(res)
    model = load_model(Path(cfg["paths"]["models_dir"]) / "xgb_multi.joblib")
    pred = model.predict(feat[POINT_FEATURES].values)
    proba = model.predict_proba(feat[POINT_FEATURES].values)

    # 3) 汇总
    alarm_events = [
        {"rule": r, "start": int(e.start), "end": int(e.end), "peak": float(e.peak)}
        for r, evs in events.items() for e in evs
    ]
    tail = pred[-max(1, len(pred) // 10):]          # 末 10% 步的主导判别
    values, counts = np.unique(pred, return_counts=True)
    final = {
        "末段主导判别": CLASS_NAMES.get(int(np.bincount(tail).argmax()), str(np.bincount(tail).argmax())),
        "全程判别占比": {CLASS_NAMES.get(int(v), str(v)): float(c / len(pred))
                        for v, c in zip(values, counts)},
        "告警事件数": len(alarm_events),
        "触发过的告警": sorted({e["rule"] for e in alarm_events}),
    }
    return {"df": res.assign(pred=pred, prob_max=proba.max(axis=1)),
            "events": alarm_events, "final": final}


# ---------------- CH-BatteryGen 包级流 ----------------

def diagnose_pack_stream(df: pd.DataFrame, chem: str, mode: str, cfg: dict) -> dict:
    info = load_alarm_thresholds(
        Path(cfg["paths"]["metrics_dir"]) / "chbatterygen_alarm.json"
    )
    from evfd.data.load_chbatterygen import add_pack_indicators

    ind = add_pack_indicators(df)
    key = f"{chem}_{mode}"
    stats = segment_features(ind)

    # Tier 2 段级包络
    t2_thr = info["thresholds_t2"].get(key)
    t2_fired = []
    if t2_thr:
        for col, i in t2_thr.items():
            if i["direction"] == "high" and stats[col] > i["threshold"]:
                t2_fired.append(col)
            if i["direction"] == "low" and stats[col] < i["threshold"]:
                t2_fired.append(col)

    # Tier 1 步级告警
    t1_events, t1_rules = [], []
    t1_thr = info["thresholds_t1"].get(key)
    if t1_thr:
        rules = [type("R", (), {
            "name": n, "column": i["column"], "direction": i["direction"],
            "threshold": i["threshold"], "kind": i["kind"],
        })() for n, i in t1_thr.items()]
        res, events = run_alarms(ind, type("TS", (), {"rules": rules})(),
                                 persistence=info["settings"]["t1_persistence"],
                                 cooldown=info["settings"]["t1_cooldown"])
        t1_events = [
            {"rule": r, "start": int(e.start), "end": int(e.end), "peak": float(e.peak)}
            for r, evs in events.items() for e in evs
        ]
        t1_rules = sorted({e["rule"] for e in t1_events})

    # 段级四类诊断（XGB 统计特征 + 曲线热图融合 CNN（v1.2 最优段级模型），概率取均值）
    feats = dict(stats)
    feats["mode_charge"] = 1.0 if mode == "charge" else 0.0
    feats["chem_ncm"] = 1.0 if chem == "NCM" else 0.0
    feats["n_cells"] = float(sum(1 for c in df.columns if c.startswith("VOLT_")))
    model = load_model(Path(cfg["paths"]["models_dir"]) / "xgb_chbatterygen.joblib")
    with open(Path(cfg["paths"]["metrics_dir"]) / "chbatterygen_feature_names.json",
              encoding="utf-8") as f:
        feat_names = json.load(f)["feature_names"]
    X = pd.DataFrame([feats])[feat_names]
    proba = model.predict_proba(X.values)[0]
    names = ["normal", "self_discharge", "high_resistance", "low_capacity"]

    cnn_proba = None
    if mode == "charge":
        try:
            import torch

            from evfd.data.curves import SHAPE_FEATURES, residual_heatmap, soc_aligned_shape
            from evfd.models.deep import CurveFusionNet

            net = CurveFusionNet(num_dim=len(SHAPE_FEATURES) + len(feat_names) - 3,
                                 n_classes=4)
            net.load_state_dict(
                torch.load(Path(cfg["paths"]["models_dir"]) / "curve_fusion_net.pt",
                           map_location="cpu"))
            net.eval()
            with open(Path(cfg["paths"]["metrics_dir"]) / "curve_fusion_meta.json",
                      encoding="utf-8") as f:
                cmeta = json.load(f)
            num = dict(stats)
            num.update(soc_aligned_shape(df))
            num["mode_charge"] = 1.0
            num["chem_ncm"] = 1.0 if chem == "NCM" else 0.0
            num["n_cells"] = feats["n_cells"]
            zv = (np.array([num[c] for c in cmeta["num_feature_names"]])
                  - np.array(cmeta["num_mu"])) / np.array(cmeta["num_sd"])
            with torch.no_grad():
                cp = torch.softmax(net(
                    torch.from_numpy(residual_heatmap(df))[None, None],
                    torch.from_numpy(zv.astype(np.float32))[None]), dim=1)[0].numpy()
            cnn_proba = {n: float(p) for n, p in zip(names, cp)}
            proba = (proba + cp) / 2.0                 # 简单late-fusion 均值
        except Exception as e:                         # 模型缺失/内存不足时降级
            print(f"  (曲线 CNN 推理跳过: {type(e).__name__})")

    final = {
        "段级诊断": {n: float(p) for n, p in zip(names, proba)},
        "Tier1 步级告警": t1_rules,
        "Tier2 一致性异常": t2_fired,
    }
    if cnn_proba is not None:
        final["曲线CNN单独判别"] = cnn_proba
    return {"final": final, "t1_events": t1_events, "stats": stats}


def diagnose_vehicle_stream(files: list[Path], chem: str, cfg: dict) -> dict:
    """多段车级诊断：段级告警 + 跨段特征 + 车级四类分类。"""
    from evfd.data.load_chbatterygen import (
        add_pack_indicators,
        segment_features,
        vehicle_cross_features,
    )

    seg_recs = []
    seg_reports = []
    for p in files:
        df = pd.read_csv(p)
        mode = "discharge" if "discharge" in p.name.lower() else "charge"
        ind = add_pack_indicators(df)
        stats = segment_features(ind)
        seg_recs.append({"ind": ind, "stats": stats, "mode": mode})
        seg_reports.append((p.name, mode, stats))

    vf = vehicle_cross_features(seg_recs)
    vf["chem_ncm"] = 1.0 if chem == "NCM" else 0.0
    vf["n_cells_mean"] = float(np.mean([
        sum(1 for c in pd.read_csv(p, nrows=0).columns if c.startswith("VOLT_"))
        for p in files
    ]))
    model = load_model(Path(cfg["paths"]["models_dir"]) / "xgb_chbatterygen_vehicle.joblib")
    with open(Path(cfg["paths"]["metrics_dir"]) / "chbatterygen_vehicle_feature_names.json",
              encoding="utf-8") as f:
        feat_names = json.load(f)["feature_names"]
    X = pd.DataFrame([vf])[feat_names]
    proba = model.predict_proba(X.values)[0]
    names = ["normal", "self_discharge", "high_resistance", "low_capacity"]
    return {
        "车级诊断": {n: float(p) for n, p in zip(names, proba)},
        "段数": len(files),
        "各段概要": [
            {"segment": n, "mode": m, "cell_spread_max": s["cell_spread_max"],
             "dev_min_p90": s["dev_min_p90"]}
            for n, m, s in seg_reports
        ],
    }


# ---------------- CLI ----------------

def detect_and_load(path: Path) -> tuple[pd.DataFrame, str]:
    if path.suffix.lower() == ".xlsx":
        raw = pd.read_excel(path)
        raw.columns = [c.strip().lower() for c in raw.columns]
        df = raw.rename(columns={"temperature": "temp"})[["soc", "temp", "volt", "label"]]
        return df, "mendeley"
    df = pd.read_csv(path)
    if any(c.startswith("VOLT_") for c in df.columns):
        return df, "pack"
    lower = {c.lower() for c in df.columns}
    if {"soc", "temp", "volt"} <= lower or {"soc", "temperature", "voltage"} <= lower:
        return df, "mendeley"
    raise SystemExit(f"无法识别输入格式：{list(df.columns)[:6]}…")


def print_mendeley_report(out: dict) -> None:
    print("\n========== Mendeley 三特征流诊断 ==========")
    print(f"步数: {len(out['df'])}")
    for k, v in out["final"].items():
        print(f"  {k}: {v}")
    if out["events"]:
        print("  告警事件:")
        for e in out["events"][:12]:
            print(f"    [{e['rule']}] 步 {e['start']}–{e['end']} (峰值 {e['peak']:.3f})")
        if len(out["events"]) > 12:
            print(f"    … 共 {len(out['events'])} 个事件")
    print("  判别依据: XGBoost 逐点分类 + 健康包络阈值流式告警（阈值见 results/metrics/alarm_design.json）")


def print_pack_report(out: dict, chem: str, mode: str) -> None:
    print(f"\n========== 包级流诊断（{chem} · {mode}） ==========")
    f = out["final"]
    diag = max(f["段级诊断"], key=f["段级诊断"].get)
    print(f"  段级诊断: {diag}  ({'，'.join(f'{k} {v:.0%}' for k, v in f['段级诊断'].items())})")
    if "曲线CNN单独判别" in f:
        cnn = f["曲线CNN单独判别"]
        print(f"  曲线CNN单独判别: {max(cnn, key=cnn.get)} "
              f"({'，'.join(f'{k} {v:.0%}' for k, v in cnn.items())})")
    print(f"  Tier1 步级告警: {f['Tier1 步级告警'] or '无'}")
    print(f"  Tier2 一致性异常: {f['Tier2 一致性异常'] or '无'}")
    if diag != "normal" and not f["Tier1 步级告警"] and not f["Tier2 一致性异常"]:
        print("  ⚠ 告警层未见异常但分类器判为故障：以告警层为准，建议增加该车的段样本复检"
              "（分类器对正常段召回约 0.6，见评测报告局限性一节）")
    if out["t1_events"]:
        for e in out["t1_events"][:8]:
            print(f"    [{e['rule']}] 步 {e['start']}–{e['end']} (峰值 {e['peak']:.3f})")
    print("  判别依据: 两级告警（阈值见 results/metrics/chbatterygen_alarm.json）+ XGBoost 统计特征 ⊕ 曲线热图融合 CNN（充电段）")


def main() -> None:
    ap = argparse.ArgumentParser(description="车端电池故障诊断小工具")
    ap.add_argument("--input", type=str, nargs="+", help="输入 csv/xlsx 路径（多个同车文件 → 车级诊断）")
    ap.add_argument("--chem", choices=["NCM", "LFP"], default=None, help="包级流的化学体系")
    ap.add_argument("--mode", choices=["charge", "discharge"], default=None, help="包级流的工况")
    ap.add_argument("--out", type=str, default=None, help="标注结果输出 csv")
    ap.add_argument("--demo", action="store_true", help="运行内置示例")
    args = ap.parse_args()
    cfg = load_config()

    if args.demo:
        # 示例 1：Mendeley 短路块（788 行起）+ 过放电块
        mp = Path(cfg["paths"]["mendeley_multi"])
        raw = pd.read_excel(mp)
        raw.columns = [c.strip().lower() for c in raw.columns]
        raw = raw.rename(columns={"temperature": "temp", "voltage": "volt"})
        for name, sl in [("健康段（前 200 步）", slice(0, 200)),
                          ("过放电块", slice(424, 788)),
                          ("短路块", slice(788, 1152))]:
            print(f"\n—— 示例输入：{name} ——")
            out = diagnose_mendeley_stream(raw.iloc[sl][["soc", "temp", "volt"]], cfg)
            print_mendeley_report(out)
        # 示例 2：CH-BatteryGen 一辆高内阻故障车 + 一辆正常车
        root = Path(cfg["paths"]["chbatterygen_root"])
        for fault, vin in [("high_resistance", "vin_41"), ("normal", "vin_101")]:
            cands = sorted((root / "LFP" / fault / vin).glob("*_charge_1.csv"))
            if not cands:
                continue
            print(f"\n—— 示例输入：LFP/{fault}/{vin} 充电段 ——")
            out = diagnose_pack_stream(pd.read_csv(cands[0]), "LFP", "charge", cfg)
            print_pack_report(out, "LFP", "charge")
        return

    if not args.input:
        ap.error("需要 --input 或 --demo")
    paths = [Path(p) for p in args.input] if isinstance(args.input, list) else [Path(args.input)]
    if len(paths) > 1:
        # 多段 → 车级诊断（同一辆车的多个段文件）
        chem = args.chem or ("NCM" if "ncm" in str(paths[0]).lower() else "LFP")
        out = diagnose_vehicle_stream(paths, chem, cfg)
        print(f"\n========== 车级诊断（{chem} · {out['段数']} 段） ==========")
        diag = max(out["车级诊断"], key=out["车级诊断"].get)
        print(f"  车级诊断: {diag}  "
              f"({'，'.join(f'{k} {v:.0%}' for k, v in out['车级诊断'].items())})")
        for s in out["各段概要"]:
            print(f"    {s['segment']} ({s['mode']}): spread_max={s['cell_spread_max']:.3f} "
                  f"dev_min_p90={s['dev_min_p90']:.3f}")
        return
    path = paths[0]
    df, kind = detect_and_load(path)
    if kind == "mendeley":
        out = diagnose_mendeley_stream(df, cfg)
        print_mendeley_report(out)
        if args.out:
            out["df"].to_csv(args.out, index=False)
            print(f"  标注结果已写出: {args.out}")
    else:
        chem = args.chem or ("NCM" if "ncm" in str(path).lower() else "LFP")
        mode = args.mode or ("discharge" if "discharge" in path.name.lower() else "charge")
        out = diagnose_pack_stream(df, chem, mode, cfg)
        print_pack_report(out, chem, mode)


if __name__ == "__main__":
    main()
