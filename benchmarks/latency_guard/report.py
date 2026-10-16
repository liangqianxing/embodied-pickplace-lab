"""Verify the prospective latency follow-up and preserve paired intervention evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from run import ROOT, rollout, summarize

from pickplace.study import write_json
from pickplace.study_policy import StudyPolicy

HERE = Path(__file__).resolve().parent
OUT = HERE / "artifacts/v1"


def verify_and_report():
    torch.set_num_threads(1)
    result = summarize(OUT)
    frozen = json.loads((OUT / "frozen.json").read_text())
    config = frozen["config"]
    for name, expected in frozen["source_sha256"].items():
        actual = hashlib.sha256((OUT / "source_snapshot" / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen source snapshot mismatch: {name}")
    for seed, expected in frozen["checkpoint_sha256"].items():
        checkpoint = ROOT / config["base_checkpoint_directory"] / f"raw256-{seed}.pt"
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Checkpoint changed: {checkpoint}")
    comparisons = []
    for condition, description, outcome in [
        (config["conditions"][1], "first identity failure repaired by EMA", (False, True)),
        (config["conditions"][0], "first clean success regressed by EMA", (True, False)),
        (config["conditions"][2], "first 100ms success regressed by EMA", (True, False)),
    ]:
        selected = None
        for training_seed in config["training_seeds"]:
            a = json.loads(
                (OUT / f"trials/identity-{training_seed}-{condition['name']}.json").read_text()
            )["trials"]
            b = json.loads(
                (OUT / f"trials/ema_half-{training_seed}-{condition['name']}.json").read_text()
            )["trials"]
            selected_pair = next(
                (
                    (x, y)
                    for x, y in zip(a, b, strict=True)
                    if (x["success"], y["success"]) == outcome
                ),
                None,
            )
            if selected_pair is not None:
                selected = (training_seed, *selected_pair)
                break
        if selected is None:
            continue
        seed, a, b = selected
        model = StudyPolicy.load(ROOT / config["base_checkpoint_directory"] / f"raw256-{seed}.pt")
        pair = {
            "training_seed": seed,
            "scenario_seed": a["seed"],
            "condition": condition,
            "selection": description,
        }
        for variant, previous in (("identity", a), ("ema_half", b)):
            replay = rollout(model, condition, previous["seed"], variant, record=True)
            for key, value in previous.items():
                if replay[key] != value:
                    raise ValueError(f"Replayed trial differs: {variant} {key}")
            pair[variant] = replay
        path = OUT / f"paired-{condition['name']}.json"
        write_json(path, pair)
        comparisons.append(
            {
                "path": path.name,
                "selection": description,
                "training_seed": seed,
                "scenario_seed": a["seed"],
            }
        )
    write_json(OUT / "pair_index.json", comparisons)
    lo, hi = result["primary_bootstrap_95"]
    clean_lo, clean_hi = result["clean_bootstrap_95"]
    lines = [
        "# Fixed-Weight Latency Follow-up",
        "",
        f"Protocol frozen: {frozen['frozen_at']}",
        "",
        "**4,500 newly reserved closed-loop test trials; five frozen BC256 training runs.**",
        "",
        "This prospective follow-up was motivated by the earlier study's observed failures; it is not an original preregistered hypothesis or an external benchmark.",
        "",
        "| Execution | Condition | Mean success | Across-run SD | Successful steps | All-trial steps |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in result["rows"]:
        lines.append(
            f"| {row['variant']} | {row['condition']} | {row['mean']:.1%} | {row['sd']:.1%} | {row['mean_successful_steps']:.1f} | {row['mean_all_steps']:.1f} |"
        )
    lines += [
        "",
        "## Prespecified primary contrast",
        "",
        f"EMA(0.5) minus identity at 50ms: **{100 * result['primary_mean_difference']:+.1f} percentage points**; exploratory paired bootstrap interval [{100 * lo:+.1f}, {100 * hi:+.1f}] pp.",
        f"Clean difference: {100 * result['clean_mean_difference']:+.1f} pp; interval [{100 * clean_lo:+.1f}, {100 * clean_hi:+.1f}] pp.",
        "",
        "## What changed",
        "",
        "- Same frozen neural-network weights and same scene seeds across execution wrappers.",
        "- identity sends the raw action; gain_half halves XYZ only; ema_half computes 0.5 * requested XYZ + 0.5 * previous filtered XYZ.",
        "- Gripper commands are unchanged. Filter state resets to zero at each episode. Observation previous-action channels describe actually executed commands.",
        "- Every wrapper uses the original 220-step horizon and success criteria. No coefficient sweep or retraining.",
        "",
        "## Honest interpretation",
        "",
        "- EMA helps this 50ms latency setting with a small clean-success decrease, but NOT the 100ms setting; do not claim universal delay robustness.",
        "- Reduced XYZ gain is not an equivalent safe fallback: it damages clean performance substantially for some training runs.",
        "- Successful-episode length is conditional on success and not a matched speed comparison. All-trial length includes failures/timeouts.",
        "- Dynamics and feedback distributions change under filtering. Results demonstrate a wrapper intervention, not proof of one exclusive failure mechanism.",
        "- Only five training/data-subset seeds. Model family was selected after study v1; remaining comparisons are exploratory and not multiplicity-corrected.",
        "- EMA filtering is a standard baseline, not a novel algorithm. There is no hardware or external-adoption claim.",
        "",
        "## Paired replay evidence",
        "",
    ]
    for pair in comparisons:
        lines.append(
            f"- {pair['path']}: {pair['selection']}; training {pair['training_seed']}, scene {pair['scenario_seed']}."
        )
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(
        f"Verified frozen sources/checkpoints and {len(comparisons)} exactly reproducible paired cases."
    )
    return result


if __name__ == "__main__":
    verify_and_report()
