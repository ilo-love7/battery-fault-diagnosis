# 车端电池故障诊断：内短路 / 过放电识别与告警（A1）

实车 BMS 安全告警落地项目：从电压/温度数据中区分 **健康(0) / 过放电故障(1) / 内短路故障(2)**，
产出 **递进式故障分类算法（随机森林 → XGBoost → 深度模型）+ 告警阈值设计 + 基准评测报告 + 可直接运行的诊断小工具**。

## 数据

| 数据集 | 角色 | 位置 |
|---|---|---|
| Mendeley《EV Lithium Ion Battery Fault Diagnosis Dataset》(t42ynjk936, CC BY 4.0) | 主数据集：三类标签训练/评测 | `D:\项目\02_数据集\01_国际公开集\EV_FaultDiag_Mendeley\` |
| CH-BatteryGen V1.0（中汽研×华为，ICLR 2026 BatteryFault，CC BY-NC-SA 4.0） | 跨数据集基准验证（实车工况生成的 1000 车 / 4 类故障） | `D:\项目\02_数据集\02_国内公开集\CH-BatteryGen\` |

评测协议对齐 ICLR 2026 BatteryFault 论文：segment 级分层 8:2 + 严格轨迹/车级分组划分（防泄漏），
指标 Accuracy / per-class Recall / 加权 F1 / 混淆矩阵。

## 目录

```
configs/        路径与告警阈值配置（config.yaml）
src/evfd/       核心包：data(加载/特征) models(RF/XGB/深度) alarm(阈值设计) evaluate(协议/指标)
scripts/        01_eda → 02_经典模型 → 03_深度模型 → 04_告警设计 → 05_跨库验证 → 06_图表
tools/          diagnostic_tool.py 诊断小工具（CLI，输入 CSV 输出诊断+告警）
results/        指标 json + 图
reports/        基准评测报告
```

## 运行

```bash
# venv：D:\项目\venvs\flsoh-py312（Python 3.12，sklearn/xgboost/torch-cpu/pandas）
D:\项目\venvs\flsoh-py312\Scripts\python.exe scripts\01_eda.py
D:\项目\venvs\flsoh-py312\Scripts\python.exe scripts\02_train_classical.py
D:\项目\venvs\flsoh-py312\Scripts\python.exe scripts\03_train_deep.py
D:\项目\venvs\flsoh-py312\Scripts\python.exe scripts\04_alarm_design.py
D:\项目\venvs\flsoh-py312\Scripts\python.exe scripts\05_chbatterygen_eval.py
# 诊断小工具
D:\项目\venvs\flsoh-py312\Scripts\python.exe tools\diagnostic_tool.py --demo
```

## 数据与局限性声明（诚实性）

- Mendeley 数据为 **MATLAB 仿真**数据，三类各为一条连续轨迹（1152 行），非多电芯/多工况实测；
  其上的分类指标是**该仿真分布内的乐观界**，报告同时给出随机划分（上限）与时序分块划分（保守）两套结果。
- CH-BatteryGen 为 **机理约束 AI 生成**数据（真实车数据标定），用于验证告警逻辑的跨分布稳健性，非实测验证。
- 告警阈值在「健康数据分位数 + 目标误报率」下设计，在故障段上测检出率与检出延迟。
