# 车端电池故障诊断：内短路 / 过放电识别与告警（A1）

实车 BMS 安全告警落地项目：从电压/温度数据中区分 **健康(0) / 过放电故障(1) / 内短路故障(2)**，
产出 **递进式故障分类算法（随机森林 → XGBoost → 膨胀卷积深度模型）+ 告警阈值设计 + 基准评测报告 + 可直接运行的诊断小工具**。

**v1.1（2026-10-25）**：膨胀卷积 dcnn64（窗 64）P2 时序口径 100% / P1 99.1% 登顶；PSI 输入漂移监控量化树模型外推失效（P1 max 0.04 vs P2 max 4.23）；CH 侧车级跨段特征把自放电召回提到 0.875；两个假设（自放电静态偏移、按串数定标降误报）经实验证伪并留档（报告 §6.3）。

**v1.2（2026-10-26）**：全曲线形状路线（攻自放电）——单体电压残差热图（去包中位数，固定 ±50mV 物理尺度）+ SOC 对齐形状描述子 + CurveFusionNet（热图 CNN ⊕ 数值分支，论文 BatteryMultiModalCNN 的 CPU 规模版）。见 `scripts/06_curve_cnn.py` 与报告 §6.4。

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
scripts/        01_eda → 02_经典模型(+PSI) → 03_深度模型(cnn16/dcnn64/lstm64) → 04_告警设计 → 05_跨库验证
tools/          diagnostic_tool.py 诊断小工具（CLI，输入 CSV 输出诊断+告警）
results/        指标 json + 图
reports/        基准评测报告
```

## 运行

```bash
# venv：D:\项目\venvs\flsoh-py312（Python 3.12，sklearn/xgboost/torch-cpu/pandas）
D:\项目\venvs\flsoh-py312\Scripts\python.exe -m pytest tests/ -q      # 9 项通过
D:\项目\venvs\flsoh-py312\Scripts\python.exe scripts\01_eda.py
D:\项目\venvs\flsoh-py312\Scripts\python.exe scripts\02_train_classical.py   # + PSI 漂移报告
D:\项目\venvs\flsoh-py312\Scripts\python.exe scripts\03_train_deep.py        # cnn16 / dcnn64 / lstm64
D:\项目\venvs\flsoh-py312\Scripts\python.exe scripts\04_alarm_design.py
D:\项目\venvs\flsoh-py312\Scripts\python.exe scripts\05_chbatterygen_eval.py # 告警三方案 + 段级/车级分类
# 诊断小工具（单段或多段车级）
D:\项目\venvs\flsoh-py312\Scripts\python.exe tools\diagnostic_tool.py --demo
D:\项目\venvs\flsoh-py312\Scripts\python.exe tools\diagnostic_tool.py --input seg1.csv seg2.csv --chem LFP
```

## 数据与局限性声明（诚实性）

- Mendeley 数据为 **MATLAB 仿真**数据，三类各为一条连续轨迹（1152 行），非多电芯/多工况实测；
  其上的分类指标是**该仿真分布内的乐观界**，报告同时给出随机划分（上限）与时序分块划分（保守）两套结果。
- CH-BatteryGen 为 **机理约束 AI 生成**数据（真实车数据标定），用于验证告警逻辑的跨分布稳健性，非实测验证。
- 告警阈值在「健康数据分位数 + 目标误报率」下设计，在故障段上测检出率与检出延迟。
