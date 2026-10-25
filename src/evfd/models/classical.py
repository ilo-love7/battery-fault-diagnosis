"""经典模型：随机森林基线 与 XGBoost（递进第二步）。"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def train_rf(X_train, y_train, seed: int = 42) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )
    clf.fit(X_train, y_train)
    return clf


def train_xgb(
    X_train, y_train, X_val=None, y_val=None, seed: int = 42
) -> XGBClassifier:
    clf = XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=int(max(y_train) + 1),
        eval_metric="mlogloss",
        early_stopping_rounds=40 if X_val is not None else None,
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        verbosity=0,
    )
    if X_val is not None:
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        clf.fit(X_train, y_train)
    return clf


def save_model(model, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, p)


def load_model(path: str | Path):
    return joblib.load(path)


def feature_importance(model, feature_names: list[str]) -> dict[str, float]:
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return {}
    order = np.argsort(imp)[::-1]
    return {feature_names[i]: float(imp[i]) for i in order}
