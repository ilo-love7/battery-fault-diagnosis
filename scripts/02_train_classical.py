"""02 经典模型：随机森林基线 + XGBoost，双划分协议（P1 随机 / P2 时序分块）。"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evfd.config import load_config
from evfd.data.features import POINT_FEATURES, add_point_features, fit_standardizer, apply_standardizer
from evfd.data.load_mendeley import CLASS_NAMES, load_multi
from evfd.evaluate.metrics import classification_report_dict, save_metrics
from evfd.evaluate.splits import split_block_contiguous, split_random_stratified
from evfd.models.classical import feature_importance, save_model, train_rf, train_xgb

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def plot_cm(y_true, y_pred, title, out_path):
    labels = sorted(CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=labels, display_labels=[CLASS_NAMES[l] for l in labels],
        cmap="Blues", colorbar=False, ax=ax,
    )
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    seed = cfg["splits"]["random_state"]
    mdir = Path(cfg["paths"]["metrics_dir"])
    fdir = Path(cfg["paths"]["figures_dir"])
    modir = Path(cfg["paths"]["models_dir"])
    for d in (mdir, fdir, modir):
        d.mkdir(parents=True, exist_ok=True)

    md = load_multi(cfg["paths"]["mendeley_multi"])
    df = add_point_features(md.df)
    X = df[POINT_FEATURES].values
    y = df["label"].values

    splits = {
        "P1_random": split_random_stratified(
            X, y, cfg["splits"]["test_frac"], cfg["splits"]["val_frac"], seed
        ),
        "P2_block": split_block_contiguous(
            X, y, md.blocks,
            train_frac=cfg["splits"]["block_train_frac"],
            val_frac=cfg["splits"]["block_val_frac"],
            purge=max(cfg["splits"]["block_purge"], 5),  # ≥ 滚动窗口长度
        ),
    }

    for split_name, sp in splits.items():
        mu, sd = fit_standardizer(sp.X_train)
        Xtr, Xva, Xte = (apply_standardizer(sp.X_train, mu, sd),
                         apply_standardizer(sp.X_val, mu, sd),
                         apply_standardizer(sp.X_test, mu, sd))
        for model_name, model in [
            ("rf", train_rf(Xtr, sp.y_train, seed)),
            ("xgb", train_xgb(Xtr, sp.y_train, Xva, sp.y_val, seed)),
        ]:
            pred = model.predict(Xte)
            rep = classification_report_dict(sp.y_test, pred, CLASS_NAMES)
            rep["model"] = model_name
            rep["split"] = split_name
            rep["n_train"] = int(len(sp.y_train))
            rep["n_test"] = int(len(sp.y_test))
            save_metrics(rep, mdir / f"classical_{model_name}_{split_name}.json")
            print(f"[{model_name}/{split_name}] acc={rep['accuracy']:.4f} "
                  f"wF1={rep['weighted_f1']:.4f} recall={rep['per_class_recall']}")
            if model_name == "xgb" and split_name == "P2_block":
                rep_imp = feature_importance(model, POINT_FEATURES)
                save_metrics(rep_imp, mdir / "xgb_feature_importance.json")
            if split_name == "P2_block":
                save_model(model, modir / f"{model_name}_multi.joblib")
            plot_cm(
                sp.y_test, pred,
                f"{model_name.upper()} · {split_name}",
                fdir / f"fig_cm_{model_name}_{split_name}.png",
            )

    print("\n经典模型完成。")


if __name__ == "__main__":
    main()
