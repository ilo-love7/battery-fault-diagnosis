"""03 深度模型（递进第三步）：1D-CNN 与 LSTM 滑窗序列分类，双划分协议。

- P1：整条轨迹滑窗后按窗口末行标签分层随机划分（上限参考）；
- P2：每类区块内部滑窗、不跨划分边界（保守）。
窗口因果（只含过去与当前），标签取窗口末行 —— 对应在线流式诊断。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evfd.config import load_config
from evfd.data.features import apply_standardizer, fit_standardizer
from evfd.data.load_mendeley import CLASS_NAMES, load_multi
from evfd.evaluate.metrics import classification_report_dict, save_metrics
from evfd.evaluate.splits import window_split_indices
from evfd.models.deep import CNN1D, LSTMNet, build_windows, predict, train_deep


def main() -> None:
    cfg = load_config()
    dm = cfg["deep_model"]
    seed = cfg["splits"]["random_state"]
    mdir = Path(cfg["paths"]["metrics_dir"])
    modir = Path(cfg["paths"]["models_dir"])
    mdir.mkdir(parents=True, exist_ok=True)
    modir.mkdir(parents=True, exist_ok=True)

    md = load_multi(cfg["paths"]["mendeley_multi"])
    from evfd.data.features import POINT_FEATURES, add_point_features

    df = add_point_features(md.df)
    series = df[POINT_FEATURES].values.astype(np.float32)   # [N, 12] 与经典模型同信息
    y_all = df["label"].values

    W, stride = dm["window"], dm["stride"]

    def make_split(kind: str):
        if kind == "P1_random":
            # 全轨迹滑窗后随机分层
            spans = [(i, i + W) for i in range(0, len(series) - W + 1, stride)]
            labels = np.array([y_all[b - 1] for _, b in spans])
            rng = np.random.default_rng(seed)
            idx = rng.permutation(len(spans))
            n_te = int(len(idx) * cfg["splits"]["test_frac"])
            n_va = int(len(idx) * cfg["splits"]["val_frac"])
            te, va, tr = idx[:n_te], idx[n_te:n_te + n_va], idx[n_te + n_va:]
            return {"train": [spans[i] for i in tr],
                    "val": [spans[i] for i in va],
                    "test": [spans[i] for i in te]}
        return window_split_indices(
            md.blocks, W, stride,
            cfg["splits"]["block_train_frac"], cfg["splits"]["block_val_frac"],
            purge=2,
        )

    for kind in ("P1_random", "P2_block"):
        sp = make_split(kind)
        # 标准化：以训练窗覆盖的行统计
        tr_rows = np.concatenate([np.arange(a, b) for a, b in sp["train"]])
        mu, sd = fit_standardizer(series[tr_rows])
        z = apply_standardizer(series, mu, sd)

        Xtr, ytr = build_windows(z, sp["train"], y_all)
        Xva, yva = build_windows(z, sp["val"], y_all)
        Xte, yte = build_windows(z, sp["test"], y_all)

        for name, model in [
            ("cnn", CNN1D(channels=len(POINT_FEATURES), n_classes=3, window=W)),
            ("lstm", LSTMNet(channels=len(POINT_FEATURES), n_classes=3,
                             hidden=dm["lstm_hidden"])),
        ]:
            info = train_deep(
                model, Xtr, ytr, Xva, yva,
                epochs=dm["epochs"], batch_size=dm["batch_size"], lr=dm["lr"],
                patience=dm["early_stop_patience"], seed=seed,
            )
            pred = predict(model, Xte)
            rep = classification_report_dict(yte, pred, CLASS_NAMES)
            rep.update({
                "model": name, "split": kind, "window": W,
                "n_train_windows": int(len(ytr)), "n_test_windows": int(len(yte)),
                "best_val_acc": info["best_val_acc"], "epochs_run": info["epochs_run"],
            })
            save_metrics(rep, mdir / f"deep_{name}_{kind}.json")
            print(f"[{name}/{kind}] acc={rep['accuracy']:.4f} wF1={rep['weighted_f1']:.4f} "
                  f"epochs={info['epochs_run']}")
            if kind == "P2_block":
                import torch
                torch.save(model.state_dict(), modir / f"deep_{name}_multi.pt")

    print("\n深度模型完成。")


if __name__ == "__main__":
    main()
