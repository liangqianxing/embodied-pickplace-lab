"""Build a dependency-free HTML study report directly from audited experiment data."""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def followup_section(root: Path, output: Path) -> str:
    source = root / "benchmarks/latency_guard/artifacts/v1"
    if not (source / "summary.json").exists():
        return ""
    results = json.loads((source / "summary.json").read_text())
    destination = output / "latency_followup"
    destination.mkdir(exist_ok=True)
    for name in ("summary.json", "REPORT.md", "paired-delay-50ms.gif", "paired-delay-50ms.png"):
        shutil.copyfile(source / name, destination / name)
    rows = []
    for row in results["rows"]:
        rows.append(
            f"<tr><th>{row['variant']}</th><td>{row['condition']}</td>"
            f"<td><b>{row['mean']:.1%}</b><small>SD {row['sd']:.1%}</small></td>"
            f"<td>{row['mean_successful_steps']:.1f}</td></tr>"
        )
    low, high = results["primary_bootstrap_95"]
    return f"""<section id="latency-followup"><div class="eyebrow">PROSPECTIVE FOLLOW-UP / NEW TEST SCENARIOS</div>
<h2>不换模型，先检验一个简单干预。</h2><p class="subtitle">在另外 {results["total_trials"]:,} 个回合中，固定原有 BC256 权重与新场景，对比原始执行、XYZ 增益减半、固定 0.5 系数的指数平滑。新测试前固定系数，没有搜索最好参数。</p>
<div class="card"><div class="contrast">{results["primary_mean_difference"] * 100:+.1f} pp</div><p>预先指定的 50 ms 延迟主比较：EMA 减原始执行。配对双轴 bootstrap 区间 [{low * 100:+.1f}, {high * 100:+.1f}] pp。</p><p>干净条件差值 {results["clean_mean_difference"] * 100:+.1f} pp。100 ms 下 EMA 反而退化：这是有适用边界的工程基线，不是通用时延算法，也不宣称平滑方法原创。</p></div>
<div class="scroll"><table><thead><tr><th>EXECUTION</th><th>CONDITION</th><th>SUCCESS</th><th>成功回合平均步数</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>
<p class="note">成功步数以成功为条件，不能直接作为配对速度比较。该后续研究由先前发现启发，模型家族选择已有事后偏差；场景种子 80000–80099 为新保留集。主比较以外的对比仅供探索。</p>
<a href="latency_followup/paired-delay-50ms.gif" aria-label="打开实际物理渲染的干预对照"><img src="latency_followup/paired-delay-50ms.gif" alt="相同模型和场景，左侧原始执行失败，右侧 EMA 平滑成功，MuJoCo 实际渲染" style="width:100%;height:auto;border-radius:8px"></a>
<p class="note">自动选择首个训练种子的首个由失败变成功的配对案例。原始记录同时保存干净条件和 100 ms 条件下的退化案例，不只保留改进视频。</p></section>"""


def panda_section(root: Path, output: Path) -> str:
    source = root / "benchmarks/robosuite/artifacts/study-v1_1"
    required = ("summary.json", "REPORT.md", "media/panda-lift.gif", "media/provenance.json")
    if not all((source / name).exists() for name in required):
        return ""
    results = json.loads((source / "summary.json").read_text())
    destination = output / "panda"
    destination.mkdir(exist_ok=True)
    for name in required:
        shutil.copyfile(source / name, destination / Path(name).name)
    rows = []
    for row in results["rows"]:
        rows.append(
            f"<tr><th>{row['condition']}</th><td><b>{row['official_mean']:.1%}</b>"
            f"<small>SD {row['official_sd']:.1%}</small></td>"
            f"<td><b>{row['hold_mean']:.1%}</b><small>SD {row['hold_sd']:.1%}</small></td></tr>"
        )
    low, high = results["primary_bootstrap_95"]
    return f"""<section id="panda"><div class="eyebrow">LOCAL STANDARD-TASK REPLICATION / PANDA LIFT</div>
<h2>换成七关节机械臂，重新检验结论。</h2><p class="subtitle">robosuite Panda Lift：128 条专家训练轨迹、16 条验证轨迹、五个优化随机种子，完成 {results["total_learned_trials"]:,} 个学习策略回合和 {results["baseline_trials"]} 个基线回合。使用上游任务，不把自建夹爪成绩当作迁移成绩。</p>
<div class="card"><div class="contrast">{results["primary_mean_difference"] * 100:+.1f} pp</div><p>原样迁移 EMA(0.5)，50 ms 时延下官方成功率的配对差值。探索性双轴 bootstrap 区间 [{low * 100:+.1f}, {high * 100:+.1f}] pp。</p><p>不论效果正负，都保留结果。任务、底层控制器与动作状态分布变化后，原任务的滤波收益不能直接外推。</p></div>
<div class="scroll"><table><thead><tr><th>CONDITION</th><th>官方高度成功</th><th>连续保持 8 步</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>
<p class="note">官方阈值与持续保持分开报告。200 控制步 horizon、固定旋转动作与提前停止规则不同于完整标准榜单协议。这里是本地标准任务复验，不是第三方独立验证；相同结果产生的退化 bootstrap 区间不代表总体成功率为 100%。</p>
<a href="panda/panda-lift.gif" aria-label="打开 Panda 真实物理渲染"><img src="panda/panda-lift.gif" alt="已训练行为克隆策略控制 Panda 机械臂抓起方块的 MuJoCo 实际渲染" style="display:block;max-width:640px;width:100%;height:auto;border-radius:8px;margin:20px auto"></a>
<p class="note">首个干净条件成功样例，经实际策略复跑和逐字段一致性校验。无图像输入、无硬件、无 VLA 或安全认证。版本兼容失败、修复记录和依赖锁文件一并保留。</p></section>"""


def build(root: Path):
    output = root / "artifacts/study/v1"
    report = json.loads((output / "summary.json").read_text())
    config = report["configuration"]
    condition_labels = {
        "id": "Clean ID",
        "position": "New position",
        "sensor_3mm": "Sensor noise",
        "delay_50ms": "50 ms delay",
        "dynamics": "Mass / friction",
        "combined": "Combined",
    }
    model_labels = {
        "raw32": "BC32",
        "relative32": "+ Relative",
        "noise32": "+ Noise",
        "relative_noise32": "+ Both",
        "raw256": "BC256 reference",
    }
    rows = []
    for spec in config["models"]:
        name = spec["name"]
        cells = []
        for condition in config["conditions"]:
            row = next(
                r
                for r in report["rows"]
                if r["model"] == name and r["condition"] == condition["name"]
            )
            rate = row["mean"]
            hue = 20 + rate * 100
            cells.append(
                f'<td style="background:hsl({hue:.1f},25%,{94 - rate * 17:.1f}%)"><b>{rate:.1%}</b><small>SD {row["sd_across_training_runs"]:.1%}</small></td>'
            )
        rows.append("<tr><th>" + model_labels[name] + "</th>" + "".join(cells) + "</tr>")
    clean_reference = next(
        r["mean"] for r in report["rows"] if r["model"] == "raw256" and r["condition"] == "id"
    )
    delayed_reference = next(
        r["mean"]
        for r in report["rows"]
        if r["model"] == "raw256" and r["condition"] == "delay_50ms"
    )
    primary = report["primary"]
    lo, hi = primary["bootstrap_95"]
    seed_rows = []
    for row in report["rows"]:
        seed_rows.append(
            "<tr><th>"
            + html.escape(row["model"])
            + "</th><td>"
            + condition_labels[row["condition"]]
            + "</td>"
            + "".join(f"<td>{rate:.0%}</td>" for rate in row["per_seed_rates"])
            + "</tr>"
        )
    headers = "".join(f"<th>{condition_labels[c['name']]}</th>" for c in config["conditions"])
    followup = followup_section(root, output)
    panda = panda_section(root, output)
    content = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Pickplace Lab | Robustness Study v1</title><style>
:root{{--paper:#f3f2ea;--ink:#263d30;--muted:#737d6d;--line:#dce0cf}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:14px/1.8 "Avenir Next","PingFang SC",sans-serif}}main{{max-width:1120px;margin:auto;padding:44px 30px}}header,footer{{font-size:10px;letter-spacing:1.5px;color:var(--muted);display:flex;justify-content:space-between;border-bottom:1px solid var(--line);padding-bottom:18px}}.eyebrow{{font-size:10px;letter-spacing:2px;color:var(--muted);margin-top:42px}}h1{{font:64px/1.08 "Iowan Old Style",Georgia,serif;letter-spacing:-2px;margin:18px 0}}h1 em{{color:#8b704d}}.subtitle{{max-width:760px;color:var(--muted)}}.stats{{display:grid;grid-template-columns:repeat(3,1fr);border:1px solid var(--line);border-radius:8px;margin:32px 0}}.stats>div{{padding:20px 24px;border-right:1px solid var(--line)}}.stats>div:last-child{{border:0}}.stats strong{{font:40px Georgia,serif;display:block}}.stats span{{font-size:10px;color:var(--muted)}}h2{{font-size:22px;font-weight:500;margin-top:36px}}.scroll{{overflow:auto}}table{{width:100%;border-collapse:separate;border-spacing:4px;font-size:11px}}th{{text-align:left;font-weight:500;white-space:nowrap;padding:10px 8px}}td{{padding:15px 10px;text-align:center;border-radius:3px}}td b{{font-size:18px;font-weight:500;display:block}}td small{{font-size:9px;color:#65735f;display:block}}.note{{font-size:11px;color:var(--muted)}}.findings{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:30px 0}}.card{{border:1px solid var(--line);border-radius:8px;padding:24px;background:#fafaf5}}.card h3{{font-size:15px;margin:0 0 12px}}.card p{{font-size:12px;color:var(--muted)}}.contrast{{font:35px Georgia,serif;color:#a35736}}.trace{{background:#e8ecdf;padding:25px;border-radius:8px;margin:30px 0;font-size:12px}}details{{border-top:1px solid var(--line);margin-top:35px;padding-top:20px}}summary{{cursor:pointer}}.seed-table td{{padding:7px;border-bottom:1px solid var(--line);font-size:10px}}.seed-table th{{font-size:10px}}footer{{border-bottom:0;border-top:1px solid var(--line);margin-top:40px;padding:18px 0}}@media(max-width:720px){{main{{padding:22px 16px}}h1{{font-size:43px}}.stats{{grid-template-columns:1fr}}.stats>div{{border-right:0;border-bottom:1px solid var(--line);padding:15px 20px}}.stats strong{{font-size:30px}}.findings{{grid-template-columns:1fr}}header{{font-size:8px}}h2{{font-size:19px}}}}@media print{{details{{display:block}}main{{padding:0}}}}
</style></head><body><main><header><span>PICKPLACE LAB / RESEARCH NOTE 001</span><span>FROZEN PROTOCOL · 2026-10-15</span></header><div class="eyebrow">FROM A DEMO TO A FALSIFIABLE EXPERIMENT</div><h1>Success is not<br><em>robustness.</em></h1><p class="subtitle">演示成功，不代表部署可靠。我们在五个训练/数据子集种子、六种部署条件下检查低数据模仿学习，保留没有支持原假设的结果。</p>
<div class="stats"><div><strong>{report["total_trials"]:,}</strong><span>闭环测试回合 · 所有结果保留</span></div><div><strong>5 × 5 × 6</strong><span>训练种子 × 模型 × 部署条件</span></div><div><strong>50 ms</strong><span>一次控制步的指令延迟</span></div></div>
<h2>完整结果，而不是最佳随机种子。</h2><p class="note">均值与样本标准差来自五个训练/数据子集种子。每个种子、每个条件评测 100 个场景。小屏幕可横向滑动表格查看全部条件。</p><div class="scroll"><table><thead><tr><th>POLICY</th>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>
<div class="findings"><article class="card"><h3>01 / 预先指定的主假设未获支持</h3><div class="contrast">{primary["mean_difference"] * 100:+.2f} pp</div><p>相对坐标 + 噪声增强，相对原始 BC32，在五种扰动条件的平均成功率差值。</p><p>配对双轴 bootstrap 95% 区间：[{lo * 100:+.2f}, {hi * 100:+.2f}] pp。仅五个训练种子，属于探索性不确定性估计，不能泛化成算法优劣定论。</p></article><article class="card"><h3>02 / 静态指标掩盖时序脆弱性</h3><div class="contrast">{clean_reference:.1%} → {delayed_reference:.1%}</div><p>BC256 从干净条件到 50 ms 指令延迟的平均成功率变化。更低的离线验证损失没有保证这项闭环部署测试表现。</p><p>这是本任务内的观察，不是“更多数据导致更差性能”的因果结论；相同 epoch 下，数据预算和更新次数同时变化。</p></article></div>
<h2>同一模型、同一场景：时延前后的轨迹。</h2><a href="latency_trace.svg" aria-label="打开完整轨迹图"><img src="latency_trace.svg" alt="BC256 同一场景在无延迟和 50 ms 延迟下的夹爪高度与 Z 轴指令对照" style="width:100%;height:auto;border:1px solid var(--line);border-radius:8px"></a><p class="note">事后诊断样例：首个训练种子的首个延迟失败场景。无延迟 61 步成功；有延迟 220 步超时。末尾 40 步的 Z 指令每步变号，只证明这个样例存在持续振荡，不代表解释了全部失败。</p><div class="trace"><strong>可以检验的下一步</strong><p>在真实机械臂模型任务中重复时延测试；比较不改输入的延迟训练、显式动作队列观测、时序模型，分别控制数据预算与优化步数。在新测试前固定协议，而不是在当前测试上调到满意为止。</p></div>
<h2>这项结果能说明什么，不能说明什么。</h2><p class="subtitle">使用自建 MuJoCo 笛卡尔夹爪、状态输入、固定物体形状。没有相机输入、语言模型、真机或外部复验。噪声训练使用已知测试噪声等级，不能称为未知传感器泛化。主比较之外的结果没有进行多重比较校正；完整假设和选择规则已在训练前保存在 protocol.json。</p>
<details><summary>查看全部种子，不隐藏退化运行</summary><div class="scroll"><table class="seed-table"><thead><tr><th>MODEL</th><th>CONDITION</th>{"".join(f"<th>{seed}</th>" for seed in config["training_seeds"])}</tr></thead><tbody>{"".join(seed_rows)}</tbody></table></div></details>
{followup}{panda}<footer><span>LOCAL RESEARCH ARTIFACT · NOT EXTERNAL VALIDATION</span><span>协议、权重、逐回合结果与源代码快照均保留</span></footer></main></body></html>"""
    target = output / "index.html"
    target.write_text(content)
    print(target)


if __name__ == "__main__":
    build(ROOT)
