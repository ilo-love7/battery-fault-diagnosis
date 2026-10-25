"""配置加载。"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]  # 仓库根（src/evfd/config.py → 上两级）


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else ROOT / "configs" / "config.yaml"
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # 相对路径一律以仓库根为基准
    for key, val in list(cfg.get("paths", {}).items()):
        pv = Path(val)
        if not pv.is_absolute():
            cfg["paths"][key] = str((ROOT / pv).resolve())
    return cfg
