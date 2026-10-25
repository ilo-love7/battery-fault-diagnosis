"""evfd — 车端电池故障诊断（A1）核心包。

模块：
- data:      Mendeley / CH-BatteryGen 加载与特征工程
- models:    随机森林 / XGBoost / 深度模型（1D-CNN、LSTM）
- alarm:     告警阈值设计与评估（误报率、检出率、检出延迟）
- evaluate:  评测协议（划分）与指标
"""

__version__ = "1.0.0"
