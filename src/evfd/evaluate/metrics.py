"""评测指标（对齐论文：Accuracy / per-class Recall / weighted-F1 / 混淆矩阵）。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)


def classification_report_dict(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: dict[int, str] | None = None
) -> dict:
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_recall": {
            (class_names or {}).get(int(c), str(c)): float(r)
            for c, r in zip(labels, per_recall)
        },
        "confusion_matrix": cm.tolist(),
        "confusion_labels": [str(c) for c in labels],
    }


def save_metrics(obj: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
