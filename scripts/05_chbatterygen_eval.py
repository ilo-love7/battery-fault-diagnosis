"""05 CH-BatteryGen 跨数据集验证（实车工况生成数据，ICLR 2026 BatteryFault）。

A) 告警规则移植：包级因果指标（过温/温升/欠压/离散度/偏低单体深度），
   在「正常车设计半区」定标，在「正常车测试半区」测误报、故障车测检出；
B) 段级四分类：18 维统计特征 + RF/XGBoost，按车分组划分（对齐论文 vehicle-level split）；
C) 图：单体电压一致性对比、车级检出率。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GroupShuffleSplit, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evfd.alarm.thresholds import design_thresholds, run_alarms
from evfd.config import load_config
from evfd.data.load_chbatterygen import (
    FAULT_TYPES,
    add_pack_indicators,
    iter_vehicle_segments,
    load_segment,
    segment_features,
)
from evfd.evaluate.metrics import classification_report_dict, save_metrics
from evfd.models.classical import save_model, train_rf, train_xgb

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def pick_vehicles(root: str, chem: str, fault: str) -> list[str]:
    d = Path(root) / chem / fault
    vins = sorted({p.parent.name for p in d.glob("vin_*/*.csv")})
    # vin_N 目录名排序按数值
    vins.sort(key=lambda s: int(s.split("_")[1]))
    return vins


def select_segment_files(root: str, chem: str, fault: str, vin: str) -> list[Path]:
    """每车取 charge_1 + discharge_1 + charge_2（缺则顺延）。"""
    d = Path(root) / chem / fault / vin
    prefs = ["vin_*_charge_1.csv", "vin_*_discharge_1.csv", "vin_*_charge_2.csv"]
    out = []
    for pat in prefs:
        found = sorted(d.glob(pat))
        if found:
            out.append(found[0])
    return out


PACK_SPECS = [
    {"name": "ALM_OT", "column": "MAX_TEMP", "direction": "high"},
    {"name": "ALM_TR", "column": "d_temp3", "direction": "high"},
    {"name": "ALM_UV", "column": "MIN_CELL_VOLT", "direction": "low"},
    {"name": "ALM_SPREAD", "column": "spread", "direction": "high"},
    {"name": "ALM_DEV", "column": "dev_min", "direction": "high"},
]


def main() -> None:
    cfg = load_config()
    ecfg = cfg["chbatterygen_eval"]
    acfg = cfg["alarm"]
    root = cfg["paths"]["chbatterygen_root"]
    mdir = Path(cfg["paths"]["metrics_dir"])
    fdir = Path(cfg["paths"]["figures_dir"])
    modir = Path(cfg["paths"]["models_dir"])
    for d in (mdir, fdir, modir):
        d.mkdir(parents=True, exist_ok=True)
    seed = cfg["splits"]["random_state"]
    n_normal = ecfg["normal_vehicles_per_chem"]
    n_seg = ecfg["segments_per_vehicle"]
    max_rows = ecfg["max_rows_per_segment"]

    rng = np.random.default_rng(seed)
    chems = ["NCM", "LFP"]
    ch_map = cfg["ch_classes"]

    # ---------- 采样 ----------
    samples = []          # (chem, fault, vin, mode, path, df)
    design_normals = {}   # (chem) -> 设计用正常车
    for chem in chems:
        nv = pick_vehicles(root, chem, "normal")
        idx = rng.permutation(len(nv))
        design_normals[chem] = [nv[i] for i in idx[: n_normal // 2]]
        test_normals = [nv[i] for i in idx[n_normal // 2: n_normal]]
        for vin in design_normals[chem] + test_normals + sum(
            (pick_vehicles(root, chem, f) for f in FAULT_TYPES[1:]), []
        ):
            fault = "normal"
            for f in FAULT_TYPES[1:]:
                if (Path(root) / chem / f / vin).exists():
                    fault = f
                    break
            for p in select_segment_files(root, chem, fault, vin)[:n_seg]:
                df = load_segment(p, max_rows)
                if len(df) >= 20:
                    samples.append((chem, fault, vin, p.name, df))
        print(f"[{chem}] 采样完成：正常设计 {len(design_normals[chem])} 车，"
              f"累计段数 {sum(1 for s in samples if s[0] == chem)}")

    # ---------- A) 告警移植：两级设计 × 两种定标方案 ----------
    # Tier 1 步级快告警（流式，剧烈特征）：温升/温升速率/欠压/骤变，持久 3 步；
    # Tier 2 段级一致性诊断（段结束后的 BMS 总结检查，慢故障）：
    #   段统计量 vs 正常段包络分位。
    # 定标方案：
    #   V1 full_normal_pool —— 全部正常车段定标（包络估计），误报 in-sample 如实报告；
    #   V2 design_test_split —— 20/20 设计/测试切分，误报 out-of-sample（部署诚实变体，
    #      预期欠覆盖 → 误报升高，与 Mendeley 敏感性分析同构，引出定标样本量教训）。
    persist, cooldown = 3, 10
    q_alarm = 0.9995
    q_seg = 0.995
    SEG_STATS_HIGH = ["cell_spread_max", "dev_min_p90", "cell_var_max", "temp_max"]
    SEG_STATS_LOW: list[str] = []

    # 预计算每段指标与统计（CSV 只解析一次）
    seg_recs = []
    for c, fault, vin, segname, df in samples:
        ind = add_pack_indicators(df)
        stats = segment_features(ind)
        seg_recs.append({
            "chem": c, "fault": fault, "vin": vin, "segment": segname,
            "mode": "discharge" if "discharge" in segname else "charge",
            "M": sum(1 for cc in df.columns if cc.startswith("VOLT_")),
            "is_design": (fault == "normal") and (vin in design_normals.get(c, [])),
            "ind": ind, "stats": stats,
        })

    alarm_rows = []
    per_variant = {}
    for variant in ("V1_full_normal_pool", "V2_design_test_split", "V3_perM_pool"):
        thresholds_t1, thresholds_t2 = {}, {}
        for chem in chems:
            for mode in ("charge", "discharge"):
                if variant == "V2_design_test_split":
                    pool_sel = [s for s in seg_recs
                                if s["chem"] == chem and s["fault"] == "normal"
                                and s["mode"] == mode and s["is_design"]]
                else:
                    pool_sel = [s for s in seg_recs
                                if s["chem"] == chem and s["fault"] == "normal"
                                and s["mode"] == mode]
                if not pool_sel:
                    continue
                # V3：按串数 M 分组定标（组内 ≥10 段才独立定标），不足回落合并池
                # —— M 逐车变化（NCM 90-96 / LFP 27-124），极差类指标分位随 M 系统性偏移
                if variant == "V3_perM_pool":
                    ts_by_m, m_notes = {}, {}
                    by_m: dict[int, list] = {}
                    for s in pool_sel:
                        by_m.setdefault(s["M"], []).append(s)
                    for m, lst in by_m.items():
                        if len(lst) >= 10:
                            ts_by_m[m] = design_thresholds(
                                pd.concat([x["ind"] for x in lst], ignore_index=True),
                                PACK_SPECS, quantile=q_alarm, hard_rules=[])
                            m_notes[m] = f"M{m}({len(lst)}段)"
                    fb_ts = design_thresholds(
                        pd.concat([x["ind"] for x in pool_sel], ignore_index=True),
                        PACK_SPECS, quantile=q_alarm, hard_rules=[])
                    thresholds_t1[f"{chem}_{mode}"] = {
                        **{f"M{m}": t.to_dict() for m, t in ts_by_m.items()},
                        "fallback": fb_ts.to_dict(), "groups": m_notes,
                    }

                    def pick_ts(m, _tbm=ts_by_m, _fb=fb_ts):
                        return _tbm.get(m, _fb)
                else:
                    pool_df = pd.concat([s["ind"] for s in pool_sel], ignore_index=True)
                    ts_pool = design_thresholds(
                        pool_df, PACK_SPECS, quantile=q_alarm, hard_rules=[])
                    thresholds_t1[f"{chem}_{mode}"] = ts_pool.to_dict()

                    def pick_ts(m, _ts=ts_pool):
                        return _ts
                # Tier 2 段级包络（定标池同 variant）
                stat_df = pd.DataFrame([s["stats"] for s in pool_sel])
                seg_thr = {}
                for col in SEG_STATS_HIGH:
                    seg_thr[col] = {"direction": "high",
                                    "threshold": float(np.quantile(stat_df[col], q_seg))}
                for col in SEG_STATS_LOW:
                    seg_thr[col] = {"direction": "low",
                                    "threshold": float(np.quantile(stat_df[col], 1 - q_seg))}
                thresholds_t2[f"{chem}_{mode}"] = seg_thr
                # 评估该化学体系+模式的全部段
                for s in seg_recs:
                    if s["chem"] != chem or s["mode"] != mode:
                        continue
                    ts = pick_ts(s["M"])
                    res, events = run_alarms(
                        s["ind"], ts, persistence=persist, cooldown=cooldown
                    )
                    fired_rules = sorted({r for r, evs in events.items() if evs})
                    t2_fired = []
                    for col, info in seg_thr.items():
                        if info["direction"] == "high" and s["stats"][col] > info["threshold"]:
                            t2_fired.append(col)
                        if info["direction"] == "low" and s["stats"][col] < info["threshold"]:
                            t2_fired.append(col)
                    alarm_rows.append({
                        "variant": variant, "chem": chem, "fault": s["fault"],
                        "vin": s["vin"], "segment": s["segment"], "mode": mode,
                        "M": s["M"], "is_design": s["is_design"],
                        "t1_alarm": bool(res["alarm_any"].any()), "t1_rules": fired_rules,
                        "t2_alarm": bool(t2_fired), "t2_stats": t2_fired,
                        "alarm": bool(res["alarm_any"].any() or t2_fired),
                        "n_steps": len(s["ind"]),
                        "alarm_step_rate": float(res["alarm_any"].mean()),
                    })
        per_variant[variant] = {"thresholds_t1": thresholds_t1,
                                "thresholds_t2": thresholds_t2}
    alarm_df = pd.DataFrame(alarm_rows)
    alarm_df.to_csv(mdir / "chbatterygen_alarm_segments.csv", index=False)

    # 车级汇总（误报口径随方案不同：V1 全部正常段 in-sample；V2 仅测试半区）
    alarm_metrics = {"variants": {}, "settings": {
        "t1_quantile": q_alarm, "t1_persistence": persist, "t1_cooldown": cooldown,
        "t2_quantile": q_seg, "t2_stats_high": SEG_STATS_HIGH,
        "t2_stats_low": SEG_STATS_LOW,
        "rules": PACK_SPECS,
    }}
    for variant in ("V1_full_normal_pool", "V2_design_test_split", "V3_perM_pool"):
        sub_all = alarm_df[alarm_df.variant == variant]
        vm = {"vehicle_level": {}, "segment_level": {}}
        for chem in chems:
            veh = sub_all[sub_all.chem == chem].groupby(["fault", "vin"]).agg(
                is_design=("is_design", "max"),
                alarm=("alarm", "max"), t1_alarm=("t1_alarm", "max"),
                t2_alarm=("t2_alarm", "max"),
            ).reset_index()
            fa_mask = (veh.fault == "normal")
            if variant == "V2_design_test_split":
                fa_mask &= (~veh.is_design.astype(bool))
            nv = veh[fa_mask]
            vm["vehicle_level"][chem] = {
                "normal_vehicles_evaluated": int(len(nv)),
                "normal_fa_vehicle": float(nv.alarm.mean()) if len(nv) else 0.0,
                "normal_fa_vehicle_t1": float(nv.t1_alarm.mean()) if len(nv) else 0.0,
                "normal_fa_vehicle_t2": float(nv.t2_alarm.mean()) if len(nv) else 0.0,
                "fault_detection_recall_vehicle": {
                    f: {
                        "combined": float(veh[veh.fault == f].alarm.mean()),
                        "t1": float(veh[veh.fault == f].t1_alarm.mean()),
                        "t2": float(veh[veh.fault == f].t2_alarm.mean()),
                    }
                    for f in FAULT_TYPES[1:] if len(veh[veh.fault == f])
                },
            }
            seg_fa = sub_all[(sub_all.chem == chem) & (sub_all.fault == "normal")]
            if variant == "V2_design_test_split":
                seg_fa = seg_fa[~seg_fa.is_design.astype(bool)]
            vm["segment_level"][chem] = {
                "normal_segments_evaluated": int(len(seg_fa)),
                "normal_fa_segment": float(seg_fa.alarm.mean()) if len(seg_fa) else 0.0,
            }
        alarm_metrics["variants"][variant] = vm

    alarm_metrics["rule_class_matrix"] = {}
    for variant in ("V1_full_normal_pool", "V2_design_test_split", "V3_perM_pool"):
        rc = {}
        for _, row in alarm_df[(alarm_df.variant == variant) & alarm_df.alarm].iterrows():
            for r in row["t1_rules"]:
                rc.setdefault(row["chem"], {}).setdefault(
                    r, {f: 0 for f in FAULT_TYPES}
                )[row["fault"]] += 1
            for r in row["t2_stats"]:
                rc.setdefault(row["chem"], {}).setdefault(
                    f"T2:{r}", {f: 0 for f in FAULT_TYPES}
                )[row["fault"]] += 1
        alarm_metrics["rule_class_matrix"][variant] = rc
    alarm_metrics["thresholds_t1"] = per_variant["V1_full_normal_pool"]["thresholds_t1"]
    alarm_metrics["thresholds_t2"] = per_variant["V1_full_normal_pool"]["thresholds_t2"]
    alarm_metrics["thresholds_t1_V3_perM"] = per_variant["V3_perM_pool"]["thresholds_t1"]
    save_metrics(alarm_metrics, mdir / "chbatterygen_alarm.json")

    print("\n== 告警移植（两级 × 三定标方案） ==")
    for variant in ("V1_full_normal_pool", "V2_design_test_split", "V3_perM_pool"):
        vm = alarm_metrics["variants"][variant]
        print(f"  [{variant}]")
        for chem in chems:
            v = vm["vehicle_level"][chem]
            s = vm["segment_level"][chem]
            print(f"    {chem}: 车级误报 {v['normal_fa_vehicle']:.1%} "
                  f"(T1 {v['normal_fa_vehicle_t1']:.0%} / T2 {v['normal_fa_vehicle_t2']:.0%}, "
                  f"评估正常车 {v['normal_vehicles_evaluated']})")
            for f, d in v["fault_detection_recall_vehicle"].items():
                print(f"       检出 {f}: 合计 {d['combined']:.0%} "
                      f"(T1 {d['t1']:.0%} / T2 {d['t2']:.0%})")
            print(f"       段级误报 {s['normal_fa_segment']:.1%} "
                  f"(正常段 {s['normal_segments_evaluated']})")

    # ---------- B) 段级四分类（车级分组划分） ----------
    feat_rows, meta = [], []
    for s in seg_recs:
        feats = dict(s["stats"])
        feats["mode_charge"] = 1.0 if s["mode"] == "charge" else 0.0
        feats["chem_ncm"] = 1.0 if s["chem"] == "NCM" else 0.0
        feats["n_cells"] = float(s["M"])
        feat_rows.append(feats)
        meta.append({
            "chem": s["chem"], "fault": s["fault"], "vin": s["vin"],
            "segment": s["segment"],
            "label": ch_map[s["fault"]], "group": f"{s['chem']}-{s['vin']}",
        })
    X = pd.DataFrame(feat_rows)
    meta_df = pd.DataFrame(meta)
    X.to_csv(mdir / "chbatterygen_segment_features.csv", index=False)
    meta_df.to_csv(mdir / "chbatterygen_segment_meta.csv", index=False)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, te_idx = next(gss.split(X, meta_df.label, groups=meta_df.group))
    Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
    ytr, yte = meta_df.label.iloc[tr_idx].values, meta_df.label.iloc[te_idx].values
    groups_te = meta_df.group.iloc[te_idx]
    # 组不重叠校验
    assert not (set(meta_df.group.iloc[tr_idx]) & set(groups_te)), "车级组泄漏！"

    feat_names = list(X.columns)
    xgb = train_xgb(Xtr.values, ytr, seed=seed)
    rf = train_rf(Xtr.values, ytr, seed)
    for name, model in [("rf", rf), ("xgb", xgb)]:
        pred = model.predict(Xte.values)
        rep = classification_report_dict(yte, pred, {v: k for k, v in ch_map.items()})
        rep["model"] = name
        rep["split"] = "vehicle_grouped_8_2"
        rep["n_segments_train"] = int(len(ytr))
        rep["n_segments_test"] = int(len(yte))
        save_metrics(rep, mdir / f"chbatterygen_classical_{name}.json")
        print(f"[CH-BatteryGen {name}] acc={rep['accuracy']:.4f} wF1={rep['weighted_f1']:.4f} "
              f"recall={rep['per_class_recall']}")
        fig, ax = plt.subplots(figsize=(5.4, 4.4))
        order = [ch_map[f] for f in FAULT_TYPES]
        ConfusionMatrixDisplay.from_predictions(
            yte, pred, labels=order,
            display_labels=FAULT_TYPES, cmap="Blues", colorbar=False, ax=ax,
        )
        ax.set_title(f"CH-BatteryGen 4类 · {name.upper()}（车级分组划分）", fontsize=10)
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(fdir / f"fig06_chbatterygen_cm_{name}.png", dpi=160)
        plt.close(fig)
    save_model(xgb, modir / "xgb_chbatterygen.joblib")
    save_metrics({"feature_names": feat_names}, mdir / "chbatterygen_feature_names.json")

    # ---------- C) 跨段持久性告警 + 车级四分类（攻自放电） ----------
    from evfd.data.load_chbatterygen import vehicle_cross_features

    veh_recs = []
    by_vehicle: dict[tuple, list] = {}
    for s in seg_recs:
        by_vehicle.setdefault((s["chem"], s["vin"]), []).append(s)
    for (chem, vin), grp in by_vehicle.items():
        vf = vehicle_cross_features(grp)
        vf["chem_ncm"] = 1.0 if chem == "NCM" else 0.0
        vf["n_cells_mean"] = float(np.mean([s["M"] for s in grp]))
        veh_recs.append({
            "chem": chem, "vin": vin, "fault": grp[0]["fault"],
            "is_design": grp[0]["is_design"], "label": ch_map[grp[0]["fault"]],
            **vf,
        })
    veh_df = pd.DataFrame(veh_recs)
    veh_df.to_csv(mdir / "chbatterygen_vehicle_features.csv", index=False)

    # 跨段持久性告警（Tier 3，车级）：最深单体跨段平均偏差 vs 正常车包络
    persist_metrics = {"settings": {"quantile": 0.995,
                                    "feature": "veh_persist_dev"}}
    for variant, sel in [("V1_full_normal_pool", veh_df.fault == "normal"),
                         ("V2_design_test_split",
                          (veh_df.fault == "normal") & veh_df.is_design.astype(bool))]:
        pm = {}
        for chem in chems:
            thr = float(np.quantile(
                veh_df[(veh_df.chem == chem) & sel]["veh_persist_dev"], 0.995))
            sub = veh_df[veh_df.chem == chem]
            flag = sub["veh_persist_dev"] > thr
            fa_mask = (sub.fault == "normal")
            if variant == "V2_design_test_split":
                fa_mask &= sub.is_design.astype(bool)
            pm[chem] = {
                "threshold_v": thr,
                "normal_fa_vehicle": float(flag[fa_mask].mean()) if fa_mask.any() else 0.0,
                "detection": {
                    f: float(flag[sub.fault == f].mean())
                    for f in FAULT_TYPES[1:] if (sub.fault == f).any()
                },
            }
        persist_metrics[variant] = pm
    save_metrics(persist_metrics, mdir / "chbatterygen_persist_alarm.json")
    print("\n== 跨段持久性告警（车级，veh_persist_dev > 正常包络 q0.995） ==")
    for variant in ("V1_full_normal_pool", "V2_design_test_split"):
        print(f"  [{variant}]")
        for chem in chems:
            pm = persist_metrics[variant][chem]
            print(f"    {chem}: 误报 {pm['normal_fa_vehicle']:.1%} | 检出 "
                  f"{({k: f'{x:.0%}' for k, x in pm['detection'].items()})}")

    # 车级四分类（一辆车一个样本，按类分层 8:2）
    vf_feat_cols = [c for c in veh_df.columns
                    if c not in ("chem", "vin", "fault", "is_design", "label")]
    Xv = veh_df[vf_feat_cols].values
    yv = veh_df["label"].values
    Xv_tr, Xv_te, yv_tr, yv_te = train_test_split(
        Xv, yv, test_size=0.2, random_state=seed, stratify=yv
    )
    for name, mdl in [("rf", train_rf(Xv_tr, yv_tr, seed)),
                      ("xgb", train_xgb(Xv_tr, yv_tr, seed=seed))]:
        pred = mdl.predict(Xv_te)
        rep = classification_report_dict(yv_te, pred, {v: k for k, v in ch_map.items()})
        rep.update({"model": name, "split": "vehicle_level_stratified_8_2",
                    "n_vehicles_train": int(len(yv_tr)),
                    "n_vehicles_test": int(len(yv_te))})
        save_metrics(rep, mdir / f"chbatterygen_vehicle_{name}.json")
        print(f"[CH 车级 {name}] acc={rep['accuracy']:.4f} wF1={rep['weighted_f1']:.4f} "
              f"recall={ {k: round(v,3) for k,v in rep['per_class_recall'].items()} }")
        if name == "xgb":
            save_model(mdl, modir / "xgb_chbatterygen_vehicle.joblib")
    save_metrics({"feature_names": vf_feat_cols},
                 mdir / "chbatterygen_vehicle_feature_names.json")

    # ---------- D) 图：一致性对比示例 ----------
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), sharey=True)
    shown = {}
    for chem, fault, vin, segname, df in samples:
        if chem != "LFP" or "discharge" in segname:
            continue
        if fault not in shown:
            shown[fault] = (vin, df)
        if len(shown) == 4:
            break
    for ax, (fault, (vin, df)) in zip(axes, shown.items()):
        vcols = [c for c in df.columns if c.startswith("VOLT_")]
        t = df["TIME"].values / 60.0
        ax.plot(t, df[vcols].values, lw=0.6, color="#999999")
        med = df[vcols].values
        ax.plot(t, np.median(med, axis=1), lw=1.6, color="#e63946", label="包中位数")
        ax.set_title(f"LFP 充电 · {fault}", fontsize=10)
        ax.set_xlabel("时间 (min)")
    axes[0].set_ylabel("单体电压 (V)")
    axes[0].legend(fontsize=8)
    fig.suptitle("正常 vs 各故障类型的单体电压一致性（灰=96 串单体，红=包中位数）", fontsize=11)
    fig.tight_layout()
    fig.savefig(fdir / "fig05_单体一致性对比.png", dpi=160)
    plt.close(fig)

    print("\nCH-BatteryGen 跨库验证完成。")


if __name__ == "__main__":
    main()
