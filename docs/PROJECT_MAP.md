# 项目地图：从可运行闭环到可检验结论

## 研究叙事

1. **搭建闭环。** MuJoCo 接触仿真 → 专家轨迹 → BC / DAgger → 独立评测 → 真实状态回放。先确认系统不是固定动画。
2. **暴露薄弱指标。** 单种子 100/100 不等于稳定可靠。多种子、位置、噪声、延迟、动力学因子研究反驳了原来的组合增强假设。
3. **提出可检验干预。** 从明确的时序失败样例出发，固定权重和 EMA 系数，在新场景上检验收益、速度代价和失效范围。
4. **检验任务依赖。** 使用独立安装的 robosuite Panda Lift，而不是只优化自建容易任务。保留官方成功与持续保持两个指标；不得预设一定能迁移。
5. **形成可投递成果。** 代码、数据、权重、协议、完整结果、失败案例、可视化与问答材料共同构成项目；不靠夸大“新算法”或只展示成功视频。

## 四个实验不是同一张排行榜

| 实验 | 作用 | 证据位置 | 不可声称 |
|---|---|---|---|
| 初始 BC / DAgger，1,000 回合 | 完整工程闭环和数据量对照 | `artifacts/results/` | 单种子高成功率代表全面泛化 |
| 坐标/噪声，15,000 回合 | 五种子因子研究与负结果 | `artifacts/study/v1/` | 组合方法有效、视觉或真机能力 |
| 固定权重滤波，4,500 回合 | 预先固定新场景下的干预 | `benchmarks/latency_guard/artifacts/v1/` | 平滑原创、适用于任意延迟或安全保证 |
| Panda Lift，1,600 回合 | 本地标准机械臂任务复验 | `benchmarks/robosuite/artifacts/study-v1_1/` | 第三方独立验证、标准榜单对齐、EMA 通用收益 |

前两个自建任务研究的数据量、训练协议、场景种子不同。Panda 则是另一个任务、控制器、观察空间和成功标准。不能横跨这些设置计算“整体提升”或把回合数当作同一分布的大样本。

## 读代码的顺序

- 环境与成功条件：`src/pickplace/env.py`、`src/pickplace/assets/scene.xml`。
- 专家与学习器边界：`src/pickplace/expert.py`、`src/pickplace/policy.py`。
- 数据与选模：`src/pickplace/data.py`、`src/pickplace/train.py`。
- 因子研究与统计：`src/pickplace/study.py`、`src/pickplace/study_analysis.py`。
- 观测和时延适配：`src/pickplace/robustness.py`。
- 滤波干预：`benchmarks/latency_guard/run.py`；注意 requested / filtered / applied 三个动作不能混用。
- 真实机械臂模型接入：`benchmarks/robosuite/smoke.py`、`learning.py`、`evaluate.py`、`study.py`。

## 什么是“影响力”

本地研究成果与分享包已经有内容，但外部影响力是另外一个维度。目前没有已核实的外部复现、引用、采用或社区贡献。标准任务的本地复验也不等于第三方验证。必须经过授权发布和真实反馈后才能更新这部分。
