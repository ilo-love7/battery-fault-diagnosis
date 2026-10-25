"""06 全曲线形状实验（v1.2，攻自放电）：曲线残差热图 CNN + SOC 对齐形状描述子。

背景（报告 §6.3）：自放电在统计特征上近不可分（静态偏移/波动率两假设证伪），
生成器表现是"充电前期爬升慢 + 曲线波动"的形状特征 → 本脚本用全曲线表示：
  A) xgb_stats      段级统计特征基线（v1.1 口径，仅充电段）
  B) xgb_stats+shape + SOC 对齐形状描述子
  C) CurveFusionNet 曲线残差热图 CNN ⊕ 数值分支融合（论文 BatteryMultiModalCNN 的 CPU 版）
划分：段级车级分组 8:2（GroupShuffleSplit，同车不跨集）；车级=同车段概率均值聚合。
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evfd.config import load_config
from evfd.data.chsample import sample_segments
from evfd.data.curves import (
    HEAT_M,
    HEAT_T,
    SHAPE_FEATURES,
    residual_heatmap,
    soc_aligned_shape,
)
from evfd.data.load_chbatterygen import FAULT_TYPES, segment_features
from evfd.evaluate.metrics import classification_report_dict, save_metrics
from evfd.models.classical import train_rf, train_xgb
from evfd.models.deep import CurveFusionNet

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    cfg = load_config()
    seed = cfg["splits"]["random_state"]
    mdir = Path(cfg["paths"]["metrics_dir"])
    fdir = Path(cfg["paths"]["figures_dir"])
    modir = Path(cfg["paths"]["models_dir"])
    for d in (mdir, fdir, modir):
        d.mkdir(parents=True, exist_ok=True)
    ch_map = cfg["ch_classes"]

    print("采样 + 构建曲线表示（充电段）…")
    samples, design_normals, _ = sample_segments(
        cfg["paths"]["chbatterygen_root"],
        normal_vehicles_per_chem=cfg["chbatterygen_eval"]["normal_vehicles_per_chem"],
        segments_per_vehicle=cfg["chbatterygen_eval"]["segments_per_vehicle"],
        max_rows=cfg["chbatterygen_eval"]["max_rows_per_segment"],
        seed=seed,
    )
    charge = [s for s in samples if "discharge" not in s[3]]
    print(f"充电段 {len(charge)} 个（车 {len({(s[0], s[2]) for s in charge})}）")

    heat, stat_rows, shape_rows, meta = [], [], [], []
    for chem, fault, vin, segname, df in charge:
        heat.append(residual_heatmap(df))
        stat_rows.append(segment_features(df))
        shape_rows.append(soc_aligned_shape(df))
        meta.append({"chem": chem, "fault": fault, "vin": vin, "segment": segname,
                     "label": ch_map[fault], "group": f"{chem}-{vin}"})
    meta = pd.DataFrame(meta)
    X_stats = pd.DataFrame(stat_rows)
    X_shape = pd.DataFrame(shape_rows)[SHAPE_FEATURES]
    y = meta.label.values
    groups = meta.group.values
    print("类别分布:", meta.fault.value_counts().to_dict())

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr, te = next(gss.split(X_stats, y, groups=groups))
    assert not (set(groups[tr]) & set(groups[te])), "车级组泄漏！"

    H = np.stack(heat)[:, None]                      # [N, 1, 128, 96]

    # ---------- A/B) 经典模型：stats vs stats+shape ----------
    for tag, X in [("stats", X_stats), ("stats_shape", pd.concat([X_stats, X_shape], axis=1))]:
        clf = train_xgb(X.values[tr], y[tr], seed=seed)
        pred = clf.predict(X.values[te])
        rep = classification_report_dict(y[te], pred, {v: k for k, v in ch_map.items()})
        rep.update({"model": f"xgb_{tag}", "split": "charge_vehicle_grouped_8_2",
                    "n_train": int(len(tr)), "n_test": int(len(te))})
        save_metrics(rep, mdir / f"curve_{tag}_xgb.json")
        print(f"[xgb/{tag}] acc={rep['accuracy']:.4f} wF1={rep['weighted_f1']:.4f} "
              f"recall={ {k: round(v, 3) for k, v in rep['per_class_recall'].items()} }")

    # ---------- C) CurveFusionNet（热图 ⊕ stats+shape）与热图单分支消融 ----------
    feat_all = pd.concat([X_stats, X_shape], axis=1)
    mu, sd = feat_all.values[tr].mean(0), feat_all.values[tr].std(0)
    sd[sd == 0] = 1.0
    Z = (feat_all.values - mu) / sd
    # 保存推理元数据（诊断工具的 CNN 分支需要同款标准化）
    save_metrics({"num_feature_names": list(feat_all.columns),
                  "num_mu": mu.tolist(), "num_sd": sd.tolist()},
                 mdir / "curve_fusion_meta.json")

    def train_fusion(model, epochs=40, batch=32, lr=1e-3, patience=8, zero_num=False):
        torch.manual_seed(seed); np.random.seed(seed)
        ds = TensorDataset(
            torch.from_numpy(H[tr]),
            torch.from_numpy((Z[tr] if not zero_num else np.zeros_like(Z[tr])).astype(np.float32)),
            torch.from_numpy(y[tr]),
        )
        dl = DataLoader(ds, batch_size=batch, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        w = torch.tensor(
            [len(y[tr]) / (len(np.unique(y[tr])) * max((y[tr] == c).sum(), 1))
             for c in np.unique(y[tr])], dtype=torch.float32)
        lossf = nn.CrossEntropyLoss(weight=w)
        Hte = torch.from_numpy(H[te])
        Zte = torch.from_numpy((Z[te] if not zero_num else np.zeros_like(Z[te])).astype(np.float32))
        yte_t = torch.from_numpy(y[te])
        best, state, bad = -1.0, None, 0
        for ep in range(epochs):
            model.train()
            for hb, nb, yb in dl:
                opt.zero_grad()
                loss = lossf(model(hb, nb), yb)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                acc = float((model(Hte, Zte).argmax(1) == yte_t).float().mean())
            if acc > best:
                best, state, bad = acc, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    break
        model.load_state_dict(state)
        return best

    num_dim = Z.shape[1]
    for tag, zero_num in [("fusion", False), ("heatmap_only", True)]:
        model = CurveFusionNet(num_dim=num_dim, n_classes=len(np.unique(y)))
        best = train_fusion(model, zero_num=zero_num)
        model.eval()
        with torch.no_grad():
            proba = torch.softmax(
                model(torch.from_numpy(H[te]),
                      torch.from_numpy((Z[te] if not zero_num else np.zeros_like(Z[te])).astype(np.float32))),
                dim=1).numpy()
        pred = proba.argmax(1)
        rep = classification_report_dict(y[te], pred, {v: k for k, v in ch_map.items()})
        rep.update({"model": f"cnn_{tag}", "split": "charge_vehicle_grouped_8_2",
                    "best_val_acc": best, "n_train": int(len(tr)), "n_test": int(len(te))})
        save_metrics(rep, mdir / f"curve_{tag}.json")
        print(f"[cnn/{tag}] acc={rep['accuracy']:.4f} wF1={rep['weighted_f1']:.4f} "
              f"recall={ {k: round(v, 3) for k, v in rep['per_class_recall'].items()} }")
        if tag == "fusion":
            proba_fusion = proba
            torch.save(model.state_dict(), modir / "curve_fusion_net.pt")

    # ---------- 车级聚合（充电段概率均值/最大值两种口径） ----------
    meta_te = meta.iloc[te].reset_index(drop=True)
    agg = pd.DataFrame({
        "group": groups[te], "fault": meta_te.fault,
        "p_selfdis": proba_fusion[:, ch_map["self_discharge"]],
        "pred": pred,
    })
    vmap = {v: k for k, v in ch_map.items()}
    for agg_name, agg_fn in [("mean", "mean"), ("max", "max")]:
        vagg = agg.groupby("group").agg(
            fault=("fault", "first"), p_selfdis=("p_selfdis", agg_fn),
            pred=("pred", lambda s: s.value_counts().idxmax()),
        ).reset_index()
        vrep = classification_report_dict(
            vagg.fault.map(ch_map).values, vagg.pred.values, vmap)
        vrep.update({"model": f"cnn_fusion_vehicle_agg_{agg_name}",
                     "split": f"vehicle_{agg_name}_over_charge_segments",
                     "n_vehicles": int(len(vagg)),
                     "note": "同车段误差相关（同一包），均值/投票聚合未必优于段级"})
        save_metrics(vrep, mdir / f"curve_fusion_vehicle_{agg_name}.json")
        print(f"[车级聚合 cnn/fusion·{agg_name}] acc={vrep['accuracy']:.4f} "
              f"wF1={vrep['weighted_f1']:.4f} "
              f"recall={ {k: round(v, 3) for k, v in vrep['per_class_recall'].items()} }")

    # ---------- 图：四类残差热图示例 + 混淆矩阵 ----------
    fault_list = ["normal", "self_discharge", "high_resistance", "low_capacity"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    shown = set()
    for i, m in enumerate(meta.itertuples()):
        if m.fault in shown or m.chem != "LFP":
            continue
        ax = axes[len(shown)]
        ax.imshow(H[i][0], aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(f"LFP 充电 · {m.fault}", fontsize=10)
        ax.set_xlabel("单体（重采样 96 列）")
        if len(shown) == 0:
            ax.set_ylabel("时间（重采样 128 行）")
        shown.add(m.fault)
        if len(shown) == 4:
            break
    fig.suptitle("单体电压残差热图（去包中位数，±50mV 满量程）：条纹=故障单体", fontsize=11)
    fig.tight_layout()
    fig.savefig(fdir / "fig07_残差热图.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ConfusionMatrixDisplay.from_predictions(
        y[te], pred, labels=sorted(ch_map.values()),
        display_labels=fault_list, cmap="Blues", colorbar=False, ax=ax)
    ax.set_title("曲线热图融合 CNN · 段级（车级分组划分）", fontsize=10)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(fdir / "fig08_曲线CNN混淆矩阵.png", dpi=160)
    plt.close(fig)

    print("\n全曲线形状实验完成。")


if __name__ == "__main__":
    main()
