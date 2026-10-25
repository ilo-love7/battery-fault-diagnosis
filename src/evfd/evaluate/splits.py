"""评测划分协议（对齐 ICLR 2026 BatteryFault 的两级划分思想）。

Mendeley 数据每类只有一条连续轨迹，因此：
- P1 segment 级（对应论文 segment-level split）：行级分层随机 8:2 —— 乐观上限；
- P2 轨迹分块级（对应论文 vehicle-level 防泄漏精神）：每类连续轨迹按
  60/20/20 切成块，块边界两侧各丢 purge 行 —— 保守下界。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass
class Split:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    name: str


def split_random_stratified(
    X: np.ndarray, y: np.ndarray, test_frac: float = 0.2,
    val_frac: float = 0.1, seed: int = 42,
) -> Split:
    """P1：行级分层随机划分（上报为上限参考）。"""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_frac, random_state=seed, stratify=y
    )
    rel = val_frac / (1.0 - test_frac)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=rel, random_state=seed, stratify=y_tr
    )
    return Split(X_tr, y_tr, X_va, y_va, X_te, y_te, "P1_random_stratified")


def split_block_contiguous(
    X: np.ndarray, y: np.ndarray, blocks: list[tuple[int, int, int]],
    feature_offset: int = 0,
    train_frac: float = 0.6, val_frac: float = 0.2,
    purge: int = 8,
) -> Split:
    """P2：每类连续区块按 60/20/20 切块，边界两侧丢 purge 行。

    X/y 为整条轨迹的特征矩阵（行对齐），blocks 为 (start, end, label)。
    特征若含因果滚动窗口，purge 同时覆盖窗口长度。
    """
    tr_idx, va_idx, te_idx = [], [], []
    for start, end, _ in blocks:
        n = end - start
        b1 = start + int(n * train_frac)
        b2 = start + int(n * (train_frac + val_frac))
        tr_idx.append(np.arange(start, max(b1 - purge, start)))
        va_idx.append(np.arange(b1 + purge, max(b2 - purge, b1 + purge)))
        te_idx.append(np.arange(b2 + purge, end))
    tr = np.concatenate(tr_idx)
    va = np.concatenate(va_idx)
    te = np.concatenate(te_idx)
    return Split(
        X[tr], y[tr], X[va], y[va], X[te], y[te], "P2_block_contiguous"
    )


def window_split_indices(
    blocks: list[tuple[int, int, int]], window: int, stride: int = 1,
    train_frac: float = 0.6, val_frac: float = 0.2, purge: int = 2,
) -> dict[str, list[tuple[int, int]]]:
    """深度模型滑窗：返回 {split: [(start_row, end_row_exclusive), ...]}。

    窗口 [i, i+window) 的标签取窗口末行；窗口不跨划分边界（各留 purge 行），
    且窗口只在同一类区块内生成。
    """
    out: dict[str, list[tuple[int, int]]] = {"train": [], "val": [], "test": []}
    for start, end, _ in blocks:
        n = end - start
        b1 = start + int(n * train_frac)
        b2 = start + int(n * (train_frac + val_frac))
        spans = {
            "train": (start, b1 - purge),
            "val": (b1 + purge, b2 - purge),
            "test": (b2 + purge, end),
        }
        for name, (a, b) in spans.items():
            for i in range(a, b - window + 1, stride):
                out[name].append((i, i + window))
    return out
