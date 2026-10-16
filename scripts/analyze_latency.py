"""Post-hoc diagnostic on the first failed latency trial of the first training seed."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from pickplace.study import episode, write_json
from pickplace.study_policy import StudyPolicy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/study/v1"


def diagnose():
    torch.set_num_threads(1)
    config = json.loads((OUT / "protocol.json").read_text())["configuration"]
    training_seed = config["training_seeds"][0]
    trials = json.loads((OUT / f"trials/raw256-{training_seed}-delay_50ms.json").read_text())[
        "trials"
    ]
    first = next(record for record in trials if not record["success"])
    model = StudyPolicy.load(OUT / f"checkpoints/raw256-{training_seed}.pt")
    clean = episode(model, {"name": "id", "distribution": "id"}, first["seed"], record=True)
    delayed_path = OUT / f"replays/raw256-{training_seed}-delay_50ms-{first['seed']}.json"
    delayed = json.loads(delayed_path.read_text())
    actions = np.asarray([frame["requested_action"] for frame in delayed["frames"][1:]])
    tail = actions[-40:, 2]
    flip_rate = float(np.mean(tail[:-1] * tail[1:] < 0))
    groups = [
        json.loads((OUT / f"trials/raw256-{seed}-delay_50ms.json").read_text())["trials"]
        for seed in config["training_seeds"]
    ]
    failures = [trial for group in groups for trial in group if not trial["success"]]
    report = {
        "status": "post-hoc diagnostic, not a prespecified causal test",
        "model": "raw256",
        "training_seed": training_seed,
        "scenario_seed": first["seed"],
        "selection": "first failed latency trial of first configured training seed",
        "clean_success": clean["success"],
        "clean_steps": clean["steps"],
        "delayed_success": delayed["success"],
        "delayed_steps": delayed["steps"],
        "last_40_z_command_sign_flip_rate": flip_rate,
        "latency_total_failures": len(failures),
        "latency_release_or_settle_failures": sum(
            t["failure"] == "release_or_settle" for t in failures
        ),
        "caution": "Final-state diagnostic categories are not causes. Oscillation is demonstrated for this selected case, not proved for all failures.",
    }
    write_json(OUT / "latency_diagnostic.json", report)
    write_json(
        OUT / "latency_clean_pair.json",
        {"model": "raw256", "training_seed": training_seed, **clean},
    )
    w, h = 1000, 570
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#f3f2ea"/>',
        '<g font-family="Georgia,serif" fill="#263d30">',
        '<text x="60" y="48" font-size="26">A 50 ms delay, persistent oscillation.</text>',
        f'<text x="60" y="75" font-size="13">BC256 / training seed {training_seed} / scene {first["seed"]} / post-hoc paired trace</text>',
    ]
    charts = [
        ("Gripper Z (m)", lambda f: f["grip"][2], 0.1, 0.32, 110),
        (
            "Requested Z action",
            lambda f: f.get("requested_action", [0, 0, 0, 0])[2],
            -1.1,
            1.1,
            325,
        ),
    ]
    for label, getter, low, high, top in charts:
        x0, plot_w, plot_h = 90, 835, 145
        pieces.append(f'<text x="60" y="{top - 8}" font-size="13">{label}</text>')
        for tick in np.linspace(low, high, 5):
            y = top + plot_h * (high - tick) / (high - low)
            pieces.append(
                f'<line x1="{x0}" y1="{y:.2f}" x2="{x0 + plot_w}" y2="{y:.2f}" stroke="#d5dccb"/><text x="{x0 - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="10">{tick:.2f}</text>'
            )
        for seconds in (0, 2, 4, 6, 8, 10):
            x = x0 + plot_w * seconds / 11
            pieces.append(
                f'<text x="{x:.2f}" y="{top + plot_h + 19}" text-anchor="middle" font-size="10">{seconds}s</text>'
            )
        for name, replay, color in (
            ("No delay", clean, "#386049"),
            ("50 ms delay", delayed, "#cc6a35"),
        ):
            points = " ".join(
                f"{x0 + plot_w * f['time'] / 11:.2f},{top + plot_h * (high - getter(f)) / (high - low):.2f}"
                for f in replay["frames"]
            )
            pieces.append(
                f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.7"/>'
            )
    pieces += [
        '<text x="60" y="532" font-size="12" fill="#386049">Green: no delay (stops at success)</text>',
        '<text x="390" y="532" font-size="12" fill="#cc6a35">Orange: one-step command delay (timeout)</text>',
        '<text x="60" y="553" font-size="11">Single-case evidence. It does not identify a causal mechanism for every failure.</text>',
        "</g></svg>",
    ]
    (OUT / "latency_trace.svg").write_text("\n".join(pieces))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    diagnose()
