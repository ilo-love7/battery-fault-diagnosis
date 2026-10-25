"""深度模型（递进第三步）：1D-CNN 与 LSTM，CPU 上训练的滑窗序列分类器。

输入：滑窗序列 [B, window, channels]（V/T/SoC z-score 后），
标签取窗口末行——对应在线 BMS 的流式诊断。
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class CNN1D(nn.Module):
    def __init__(self, channels: int = 3, hidden=(32, 64, 64), n_classes: int = 3,
                 window: int = 16):
        super().__init__()
        layers, in_c = [], channels
        for h in hidden:
            layers += [
                nn.Conv1d(in_c, h, kernel_size=3, padding=1),
                nn.BatchNorm1d(h),
                nn.ReLU(),
            ]
            in_c = h
        self.conv = nn.Sequential(*layers)
        self.pool_out = in_c * window
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(self.pool_out, 64), nn.ReLU(), nn.Linear(64, n_classes)
        )

    def forward(self, x):           # x: [B, T, C]
        return self.head(self.conv(x.transpose(1, 2)))


class DilatedCNN(nn.Module):
    """膨胀因果卷积（感受野 1+Σ(k-1)·d，默认 31 步），适配长窗（64 步）趋势特征。"""

    def __init__(self, channels: int = 12, hidden=(32, 64, 64), n_classes: int = 3,
                 dilations=(1, 2, 4, 8)):
        super().__init__()
        layers, in_c = [], channels
        for h, d in zip(hidden * len(dilations), dilations):
            layers += [
                nn.Conv1d(in_c, h, kernel_size=3, padding=d, dilation=d),
                nn.BatchNorm1d(h),
                nn.ReLU(),
            ]
            in_c = h
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveMaxPool1d(1), nn.Flatten(),
            nn.Linear(in_c, 64), nn.ReLU(), nn.Linear(64, n_classes),
        )

    def forward(self, x):           # x: [B, T, C]
        return self.head(self.conv(x.transpose(1, 2)))


class LSTMNet(nn.Module):
    def __init__(self, channels: int = 3, hidden: int = 64, n_classes: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(channels, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):           # x: [B, T, C]
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])   # 取窗口末步（流式语义）


def build_windows(
    series: np.ndarray, spans: list[tuple[int, int]], labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """spans: [(start, end)] 窗口行区间，标签取窗口末行；series: [N, C]。"""
    xs = np.stack([series[a:b] for a, b in spans])
    ys = np.array([labels[b - 1] for a, b in spans])
    return xs.astype(np.float32), ys.astype(np.int64)


def train_deep(
    model: nn.Module,
    Xtr: np.ndarray, ytr: np.ndarray,
    Xva: np.ndarray, yva: np.ndarray,
    epochs: int = 60, batch_size: int = 64, lr: float = 1e-3,
    patience: int = 10, seed: int = 42, verbose: bool = False,
    class_weights: np.ndarray | None = None,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        batch_size=batch_size, shuffle=True,
    )
    va_X = torch.from_numpy(Xva)
    va_y = torch.from_numpy(yva)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    w = None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32)
    lossf = nn.CrossEntropyLoss(weight=w)
    best_acc, best_state, bad = -1.0, None, 0
    history = []
    for ep in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            acc = float((model(va_X).argmax(1) == va_y).float().mean())
        history.append({"epoch": ep + 1, "val_acc": acc})
        if acc > best_acc:
            best_acc, best_state, bad = acc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and (ep + 1) % 10 == 0:
            print(f"  epoch {ep+1}: val_acc={acc:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_acc": best_acc, "epochs_run": len(history), "history": history}


def predict(model: nn.Module, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            outs.append(model(torch.from_numpy(X[i:i + batch_size])).argmax(1).numpy())
    return np.concatenate(outs)
