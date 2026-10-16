"""Build traceable reports and render saved physics states (never synthetic metrics)."""

from __future__ import annotations

import importlib.metadata
import json
import platform
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import mujoco
from PIL import Image

from .env import PickPlaceEnv


def build_report(root: Path) -> dict:
    root = root.resolve()
    artifacts = root / "artifacts"
    evaluation = json.loads((artifacts / "results/evaluation.json").read_text())
    dataset = json.loads((artifacts / "data/expert.json").read_text())
    training = {}
    for name in ("bc32", "bc256", "dagger"):
        path = artifacts / f"checkpoints/{name}.json"
        if path.exists():
            training[name] = json.loads(path.read_text())
    report = {
        "title": "Embodied Pickplace Lab",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("mujoco", "gymnasium", "torch", "numpy")
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dataset": {k: v for k, v in dataset.items() if k != "episodes"},
        "training": training,
        "evaluation": evaluation,
        "protocol": {
            "physics_hz": 500,
            "control_hz": 20,
            "horizon": 220,
            "observation_size": 26,
            "action_size": 4,
            "train_seed": 42,
            "success": "Lift > 10cm, place within 4.5cm, release and retreat, settle for 8 steps",
            "limitation": "State observations, fixed cube and dynamics, Cartesian gripper, one training seed; no real-robot or VLA claim.",
        },
    }
    (artifacts / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Experiment Report",
        "",
        f"Generated: {report['created_at']}",
        "",
        "## Held-out evaluation",
        "",
        "Identical initial-condition seeds across policies; no test episodes in training.",
        "",
        "| Policy | Distribution | Success | Wilson 95% CI | Mean successful steps |",
        "|---|---|---:|---|---:|",
    ]
    for row in evaluation["results"]:
        low, high = row["confidence_interval"]
        steps = (
            f"{row['mean_steps_success']:.1f}" if row["mean_steps_success"] is not None else "N/A"
        )
        lines.append(
            f"| {row['policy']} | {row['distribution']} | {row['successes']}/{row['episodes']} | {low:.1%} - {high:.1%} | {steps} |"
        )
    lines += [
        "",
        "## Failure counts",
        "",
        "No lift / misplaced or dropped / release or settle:",
        "",
    ]
    for row in evaluation["results"]:
        lines.append(f"- {row['policy']} ({row['distribution']}): {row['failures']}")
    lines += [
        "",
        "## Protocol and limitations",
        "",
        "- Training uses 256 expert episodes, with a nested 32-episode data ablation. Validation: 32 separate episodes.",
        "- DAgger uses additional expert labels on mixed learner/expert trajectories; this is NOT an equal-label-budget comparison.",
        "- DAgger validation loss measures the original expert validation distribution, not off-policy recovery quality.",
        "- Training seeds: 0-255; validation: 10000-10031; diagnostic development rollouts: 11000-11019.",
        "- DAgger seeds: 2000-2063, 3000-3063, 4000-4063. Test: 20000-20099 and 30000-30099 for the default 100-trial run.",
        "- OOD: cube begins outside the training rectangle; position support changes, not visual appearance, mass, or friction.",
        "- Wilson intervals describe finite evaluation episodes, not variance across independent training seeds.",
        "- Failure labels are final-state diagnostics, not proven causal explanations.",
        "- The expert has a finite-state controller; the learned policy only gets 26 observation values, including its previous action.",
        "- No images, natural-language instructions, real hardware, real-robot validation, or sim-to-real claims.",
        "- Exact bitwise reproducibility across MuJoCo/PyTorch versions or different platforms is not guaranteed.",
        "",
        "## Provenance",
        "",
        f"- Dataset SHA-256: `{dataset['sha256']}`",
        f"- Versions: `{report['versions']}`",
        "",
    ]
    (root / "docs/EXPERIMENT.md").write_text("\n".join(lines))
    print(f"Report written to {artifacts / 'report.json'}", flush=True)
    return report


def render_replay(root: Path, replay_id: str | None = None) -> Path:
    replays = json.loads((root / "artifacts/results/replays.json").read_text())
    if replay_id:
        replay = next((r for r in replays if r["id"] == replay_id), None)
        if replay is None:
            raise ValueError(f"Replay not found: {replay_id}")
    else:
        replay = next((r for r in replays if r["policy"] == "dagger" and r["success"]), replays[0])
    out = root / "artifacts/media"
    out.mkdir(parents=True, exist_ok=True)
    env = PickPlaceEnv(render_mode="rgb_array")
    env.reset(seed=replay["seed"], options={"distribution": replay["distribution"]})
    images = []
    try:
        for frame in replay["frames"]:
            env.data.qpos[:] = frame["qpos"]
            mujoco.mj_forward(env.model, env.data)
            images.append(Image.fromarray(env.render()))
    finally:
        env.close()
    images[0].save(out / "scene.png")
    images[len(images) // 2].save(out / "grasp.png")
    images[0].save(out / "demo.gif", save_all=True, append_images=images[1:], duration=50, loop=0)
    (out / "demo.json").write_text(
        json.dumps(
            {
                "replay_id": replay["id"],
                "seed": replay["seed"],
                "policy": replay["policy"],
                "success": replay["success"],
            },
            indent=2,
        )
    )
    print(f"Rendered actual MuJoCo replay: {out / 'demo.gif'}", flush=True)
    return out / "demo.gif"
