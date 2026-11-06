# 车端电池故障诊断：内短路 / 过放电识别与告警（A1）

实车 BMS 安全告警落地项目：从电压/温度数据中区分 **健康(0) / 过放电故障(1) / 内短路故障(2)**，
产出 **递进式故障分类算法（随机森林 → XGBoost → 膨胀卷积深度模型）+ 告警阈值设计 + 基准评测报告 + 可直接运行的诊断小工具**。

**v1.1（2026-10-25）**：膨胀卷积 dcnn64（窗 64）P2 时序口径 100% / P1 99.1% 登顶；PSI 输入漂移监控量化树模型外推失效（P1 max 0.04 vs P2 max 4.23）；CH 侧车级跨段特征把自放电召回提到 0.875；两个假设（自放电静态偏移、按串数定标降误报）经实验证伪并留档（报告 §6.3）。

**v1.2（2026-10-26）**：全曲线形状路线（攻自放电）——单体电压残差热图（去包中位数，固定 ±50mV 物理尺度）+ SOC 对齐形状描述子 + CurveFusionNet（热图 CNN ⊕ 数值分支，论文 BatteryMultiModalCNN 的 CPU 规模版）。见 `scripts/06_curve_cnn.py` 与报告 §6.4。

**v1.3（2026-10-26）**：故障明细深挖（`scripts/07_fault_details.py`）——用 faults.details.xlsx 解锁三个新能力：**故障单体定位**（放电最深单体画像：高内阻 100% / 自放电 74.4%，实证 xlsx cell id 为 0-based）、**故障分级**（论文 Task 2，轻/中/重 XGB 0.58 vs 基线 0.52）、**严重度分层检出**（高内阻轻度即 100%，重度自放电 75% vs 轻度 25%）。诊断工具接入曲线 CNN 分支。

**v1.4（2026-11-05）**：接入理论项目 **B1 的析锂前兆**（`src/evfd/alarm/b1_bridge.py`）。B1 v1.6 交付的规格是「45 ℃ 静置 7 天的 OCV 漂移」，标称 −1.781 mV、对 SEI 扩散系数的弹性 1.01、SNR 24.9；桥接取标称值 80% 作**物理兜底硬规则** `ALM_B1_OCV_DRIFT`（阈值 −1.425 mV，`direction=low`），分位数定标仍归 A1。**默认关闭**：A1 现有两个数据集都没有 ≥7 天静置段（Mendeley 只有行序无时间戳、CH-BatteryGen 每段 ≤400 行 / 10 s 采样 ≈ 66 min），`applicable()` 会拦住并打印原因，**不产生 NaN 列**。打开只需 `configs/config.yaml` 里 `alarm.b1_precursors.enabled: true` + 数据带小时级时间列 `t_h`。验证：默认关闭时 `results/metrics/alarm_design.json` 的 md5 与接入前完全一致（零侵入），测试 13 → 35 项全过。

## 数据

| 数据集 | 角色 | 位置 |
|---|---|---|
| Mendeley《EV Lithium Ion Battery Fault Diagnosis Dataset》(t42ynjk936, CC BY 4.0) | 主数据集：三类标签训练/评测 | `D:\项目\02_数据集\01_国际公开集\EV_FaultDiag_Mendeley\` |
| CH-BatteryGen V1.0（中汽研×华为，ICLR 2026 BatteryFault，CC BY-NC-SA 4.0） | 跨数据集基准验证（实车工况生成的 1000 车 / 4 类故障） | `D:\项目\02_数据集\02_国内公开集\CH-BatteryGen\` |

评测协议对齐 ICLR 2026 BatteryFault 论文：segment 级分层 8:2 + 严格轨迹/车级分组划分（防泄漏），
指标 Accuracy / per-class Recall / 加权 F1 / 混淆矩阵。

## 目录

```
configs/        路径与告警阈值配置（config.yaml）；b1_precursor_spec.json = B1 交付的析锂前兆规格（只读）
src/evfd/       核心包：data(加载/特征) models(RF/XGB/深度) alarm(阈值设计 + b1_bridge 析锂前兆桥接) evaluate(协议/指标)
scripts/        01_eda → 02_经典模型(+PSI) → 03_深度模型(cnn16/dcnn64/lstm64) → 04_告警设计 → 05_跨库验证
tools/          diagnostic_tool.py 诊断小工具（CLI，输入 CSV 输出诊断+告警）
results/        指标 json + 图
reports/        基准评测报告
```

### 与 B1 的接口（析锂前兆）

B1（析锂归因）→ A1 只交付一条前兆，其余三条不交付（各自原因写在规格的 `not_delivered`）：

| 前兆 | 能否车端取 | 结论 |
|---|---|---|
| 45 ℃ 静置 7 天 OCV 漂移 | ✅ 免拆芯 | **已交付**：`ALM_B1_OCV_DRIFT`，阈值 −1.425 mV（标称 −1.781 的 80%） |
| SEI 厚度 | ❌ 需拆芯 | 不交付（能缩到 11.14 pp，但车端拿不到） |
| 充电末端电压拐点 | ⚠️ 可测 | 不交付（B1 未做弹性/可辨识性，不硬凑阈值） |
| 库仑效率 | ✅ 可测 | 不交付（CC-CV 恒压段把电荷补回，1−CE 恒为 0 ± 3 ppm） |

规格文件 `configs/b1_precursor_spec.json` 由 `b1-aging-attribution/src/to_a1.py` 生成，含
`feature_formula`、3 条 `preconditions`、`elasticity`、`sigma_mV`。改阈值请改 B1 侧后重新生成，
**不要手改 A1 这份副本**（副本应与原件 md5 一致）。

温度必须先按 Arrhenius 归一到 45 ℃ 才能比阈值：`Ea/R = 6600 K`，即 25 ℃ 实测的 −0.443 mV
等价于 45 ℃ 的 −1.781 mV（放大 4.02 倍）。归一只在静置段温度稳定（极差 ≤1 K）时有意义。

## 运行

```bash
# venv：D:\项目\venvs\flsoh-py312（Python 3.12，sklearn/xgboost/torch-cpu/pandas）
D:\项目\venvs\flsoh-py312\Scripts\python.exe -m pytest tests/ -q      # 35 项通过
D:\项目\venvs\flsoh-py312\Scripts\python.exe scripts\01_eda.py
D:\项目\venvs\flsoh-py312\Scripts\python.exe scripts\02_train_classical.py   # + PSI 漂移报告
D:\项目\venvs\flsoh-py312\Scripts\python.exe scripts\03_train_deep.py        # cnn16 / dcnn64 / lstm64
D:\项目\venvs\flsoh-py312\Scripts\python.exe scripts\04_alarm_design.py
D:\项目\venvs\flsoh-py312\Scripts\python.exe scripts\05_chbatterygen_eval.py # 告警三方案 + 段级/车级分类
D:\项目\venvs\flsoh-py312\Scripts\python.exe scripts\06_curve_cnn.py         # 曲线热图融合 CNN（充电段 560）
D:\项目\venvs\flsoh-py312\Scripts\python.exe scripts\07_fault_details.py     # 单体定位 / 故障分级 / 严重度分层
# 诊断小工具（单段或多段车级）
D:\项目\venvs\flsoh-py312\Scripts\python.exe tools\diagnostic_tool.py --demo
D:\项目\venvs\flsoh-py312\Scripts\python.exe tools\diagnostic_tool.py --input seg1.csv seg2.csv --chem LFP
```

## 数据与局限性声明（诚实性）

- Mendeley 数据为 **MATLAB 仿真**数据，三类各为一条连续轨迹（1152 行），非多电芯/多工况实测；
  其上的分类指标是**该仿真分布内的乐观界**，报告同时给出随机划分（上限）与时序分块划分（保守）两套结果。
- CH-BatteryGen 为 **机理约束 AI 生成**数据（真实车数据标定），用于验证告警逻辑的跨分布稳健性，非实测验证。
- 告警阈值在「健康数据分位数 + 目标误报率」下设计，在故障段上测检出率与检出延迟。
