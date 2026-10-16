# 交付索引与成果边界

## 先看这三个入口

1. `README.md`：项目动机、实际结果、运行与目录说明。
2. `artifacts/study/v1/index.html`：完整研究成果页，包含负结果、时延干预、Panda 标准任务本地复验与实际物理渲染。
3. `docs/PORTFOLIO.md`：可核验的简历事实条目、90 秒讲解和面试问答；需按个人实际参与范围调整，不能虚构独立完成程度。

## 已完成的实验

| 实验 | 正式闭环回合 | 结论位置 |
|---|---:|---|
| 原始 BC32 / BC256 / DAgger / 专家 / 随机 | 1,000 | `docs/EXPERIMENT.md` |
| 坐标与噪声因子研究，五训练/数据子集种子 | 15,000 | `artifacts/study/v1/REPORT.md` |
| 新场景下固定权重执行滤波干预 | 4,500 | `benchmarks/latency_guard/artifacts/v1/REPORT.md` |
| Panda Lift 五训练种子及专家/随机基线 | 1,600 | `benchmarks/robosuite/artifacts/study-v1_1/REPORT.md` |
| **合计** | **22,100** | 不同协议的回合数总和，不是同分布独立样本池 |

## 适合讲清楚的核心结果

- 原始组合增强假设未获支持：相对坐标 + 噪声增强的平均扰动成功率比原始 BC32 低 7.72 pp，保留负结果。
- 自建抓取搬运任务上，同一权重加固定 EMA(0.5)，新场景 50 ms 延迟成功率从 56.0% 到 86.6%；干净条件轻微下降，100 ms 下退化。
- Panda 标准任务复验并未支持同一干预在 50 ms 下的收益：原始 67.2%，EMA 60.8%，差值 -6.4 pp，区间跨零；干净条件 EMA 也有明显损失。
- 因此项目贡献是完整闭环、受控实验、失败诊断与适用边界，而不是一个已证明通用有效的新算法。

## 核心证据

- 原始数据、模型、结果一致性：`scripts/verify_artifacts.py`。
- 15,000 回合完整性、摘要与271条选定重放、冻结源码：`scripts/verify_study.py`。
- 滤波干预源/权重一致性和自动选择的改进/退化配对：`benchmarks/latency_guard/report.py`。
- Panda 数据分离、五模型、1,600 回合与55条选定精确重放：`benchmarks/robosuite/report.py`。
- 原项目/滤波42项测试，加独立 Panda 环境5项测试。不是一个环境里同时运行的47项。
- 分享包内部文件和科学溯源哈希：`scripts/verify_portable_release.py`。

每种验证器的范围不同。哈希一致不能证明科学正确；选定轨迹的精确复跑不等于逐条重跑所有22,100回合。

## 启动

```bash
# 项目根目录；原项目虚拟环境
.venv/bin/pickplace serve --port 8765
.venv/bin/python -m http.server 8766 --bind 127.0.0.1 --directory artifacts/study/v1
```

`127.0.0.1:8765` 是真实新场景推理与初版对照；`127.0.0.1:8766` 是完整研究成果页。Panda 使用独立锁定环境，不要混装。

## 分享与影响力

公开仓库为 `https://github.com/liangqianxing/embodied-pickplace-lab`，项目原创内容采用 MIT 许可。可用 `scripts/build_release.py` 重新构建完整发布包，以旁边 SHA-256 与内部 manifest 校验。

用户已授权公开发布，但仍没有已核实的外部采用、独立复现、导师/研究者反馈、引用或被采纳的上游贡献。兼容性问题草稿未提交。公开发布本身不等于已有影响力。

项目不包含视觉策略、语言动作模型、真实硬件结果或安全认证。不要将接触物理仿真写成真机实验。
