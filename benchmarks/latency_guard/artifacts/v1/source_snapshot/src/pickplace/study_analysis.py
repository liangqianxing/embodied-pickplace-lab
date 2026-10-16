"""Complete-case paired analysis of the frozen study, never selected best seeds."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np

from .data import file_hash
from .study import write_json


def two_axis_bootstrap(values: np.ndarray, *, repetitions: int = 5000, seed: int = 20261015):
    """Mean interval for [training seed, shared scenario]; preserves paired contrasts."""
    if values.ndim != 2 or min(values.shape) < 1 or repetitions < 1:
        raise ValueError("Expected a nonempty two-dimensional array and positive repetitions")
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions)
    for offset in range(0, repetitions, 100):
        count = min(100, repetitions - offset)
        train_ids = rng.integers(0, values.shape[0], size=(count, values.shape[0]))
        scenario_ids = rng.integers(0, values.shape[1], size=(count, values.shape[1]))
        selected = values[train_ids[:, :, None], scenario_ids[:, None, :]]
        samples[offset : offset + count] = selected.mean(axis=(1, 2))
    return np.quantile(samples, [0.025, 0.975]).tolist()


def load_cube(output: Path):
    protocol = json.loads((output / "protocol.json").read_text())
    protocol_hash = file_hash(output / "protocol.json")
    config = protocol["configuration"]
    models = [spec["name"] for spec in config["models"]]
    conditions = [condition["name"] for condition in config["conditions"]]
    shape = (len(models), len(config["training_seeds"]), len(conditions), config["test_episodes"])
    cube = np.empty(shape, dtype=float)
    all_trials = []
    expected = list(
        range(config["test_seed_start"], config["test_seed_start"] + config["test_episodes"])
    )
    for m, name in enumerate(models):
        for s, training_seed in enumerate(config["training_seeds"]):
            checkpoint = output / f"checkpoints/{name}-{training_seed}.pt"
            meta = json.loads(checkpoint.with_suffix(".json").read_text())
            if (
                meta["checkpoint_sha256"] != file_hash(checkpoint)
                or meta["protocol_sha256"] != protocol_hash
            ):
                raise ValueError(f"Checkpoint hash mismatch: {checkpoint}")
            if set(meta["selected_episodes"]) & set(expected):
                raise ValueError("Train/test seed overlap")
            if set(meta["validation_episode_ids"]) & set(expected):
                raise ValueError("Validation/test seed overlap")
            for c, condition in enumerate(conditions):
                path = output / f"trials/{name}-{training_seed}-{condition}.json"
                result = json.loads(path.read_text())
                if result["protocol_sha256"] != protocol_hash:
                    raise ValueError(f"Trial protocol mismatch: {path}")
                if (
                    result["model"] != name
                    or result["training_seed"] != training_seed
                    or result["condition"] != config["conditions"][c]
                ):
                    raise ValueError(f"Trial identity mismatch: {path}")
                if [trial["seed"] for trial in result["trials"]] != expected:
                    raise ValueError(f"Missing, duplicated or reordered scenarios: {path}")
                cube[m, s, c] = [float(trial["success"]) for trial in result["trials"]]
                all_trials.extend(
                    {"model": name, "training_seed": training_seed, **trial}
                    for trial in result["trials"]
                )
    return protocol, models, conditions, cube, all_trials


def analyze(output: Path):
    protocol, models, conditions, cube, trials = load_cube(output)
    config = protocol["configuration"]
    boot = {"repetitions": config["bootstrap_repetitions"], "seed": config["bootstrap_seed"]}
    rows = []
    for m, model in enumerate(models):
        for c, condition in enumerate(conditions):
            seed_rates = cube[m, :, c, :].mean(axis=1)
            rows.append(
                {
                    "model": model,
                    "condition": condition,
                    "mean": float(seed_rates.mean()),
                    "sd_across_training_runs": float(seed_rates.std(ddof=1))
                    if len(seed_rates) > 1
                    else None,
                    "per_seed_rates": seed_rates.tolist(),
                    "successes": int(cube[m, :, c, :].sum()),
                    "episodes": int(cube[m, :, c, :].size),
                    "bootstrap_95": two_axis_bootstrap(cube[m, :, c, :], **boot),
                }
            )
    shifted = [c for c, name in enumerate(conditions) if name != "id"]
    # Average conditions first; resample shared scenario indices consistently across conditions.
    primary_by_model = cube[:, :, shifted, :].mean(axis=2)
    proposed, baseline = config["primary_comparison"]
    difference = primary_by_model[models.index(proposed)] - primary_by_model[models.index(baseline)]
    primary = {
        "comparison": f"{proposed} minus {baseline}",
        "mean_difference": float(difference.mean()),
        "per_seed_differences": difference.mean(axis=1).tolist(),
        "bootstrap_95": two_axis_bootstrap(difference, **boot),
    }
    effects = {}
    for condition in conditions:
        c = conditions.index(condition)
        raw, relative, noise, both = (
            cube[models.index(name), :, c, :]
            for name in ("raw32", "relative32", "noise32", "relative_noise32")
        )
        interaction = both - relative - noise + raw
        effects[condition] = {
            "interaction_mean": float(interaction.mean()),
            "bootstrap_95": two_axis_bootstrap(interaction, **boot),
        }
    report = {
        "schema_version": 1,
        "protocol_sha256": file_hash(output / "protocol.json"),
        "configuration": config,
        "total_trials": len(trials),
        "rows": rows,
        "primary": primary,
        "factorial_interactions": effects,
        "shifted_condition_means": {
            name: float(primary_by_model[i].mean()) for i, name in enumerate(models)
        },
        "statistical_caution": "Exploratory paired two-axis bootstrap with only five training/data-subset seeds; no multiple-comparison correction.",
    }
    write_json(output / "summary.json", report)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(trials[0]))
    writer.writeheader()
    writer.writerows(trials)
    (output / "trials.csv").write_text(buffer.getvalue())
    lines = [
        "# Coordinate / Observation Noise Study v1",
        "",
        f"Protocol frozen: {protocol['frozen_at']}",
        "",
        f"**{len(trials):,} closed-loop trials; {len(config['training_seeds'])} training/data-subset seeds.**",
        "",
        "Success rate: mean +/- sample SD across training runs; each run uses 100 scenarios by default.",
        "",
        "| Model | " + " | ".join(conditions) + " | Shifted mean |",
        "|---|" + "---:|" * (len(conditions) + 1),
    ]
    for model in models:
        cells = []
        for condition in conditions:
            row = next(r for r in rows if r["model"] == model and r["condition"] == condition)
            sd = row["sd_across_training_runs"]
            cells.append(
                f"{row['mean']:.1%} +/- {sd:.1%}" if sd is not None else f"{row['mean']:.1%}"
            )
        lines.append(
            f"| {model} | "
            + " | ".join(cells)
            + f" | {report['shifted_condition_means'][model]:.1%} |"
        )
    lo, hi = primary["bootstrap_95"]
    lines += [
        "",
        "## Prespecified primary comparison",
        "",
        f"{primary['comparison']}: **{primary['mean_difference'] * 100:+.2f} percentage points**.",
        f"Exploratory paired two-axis bootstrap 95% interval: [{lo * 100:+.2f}, {hi * 100:+.2f}] percentage points.",
        "",
        "This is not an independent external replication or evidence of real-world deployment success.",
        "Five seeds are too few for precise tail inference; per-seed outcomes are retained in summary.json.",
        "The 32-episode subsets and optimizer seeds vary together. raw256 is a larger-data reference, not an equal-budget competitor.",
        "Training augmentation matches the prescribed sensor-noise level; do not call this unknown-sensor generalization.",
        "The remaining condition comparisons and factorial interactions are exploratory and not multiplicity-corrected.",
        "",
        "## Artifacts",
        "",
        "- protocol.json: frozen configuration, source hashes, dependency versions.",
        "- checkpoints/: weights, selected episode IDs, validation curves and hashes.",
        "- trials/: every scenario outcome grouped by model, training seed and condition.",
        "- trials.csv: all per-episode records in one machine-readable file.",
        "- summary.json: seed-level rates, intervals and prespecified primary contrast.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines[:20]), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/study/v1"))
    analyze(parser.parse_args().output.resolve())


if __name__ == "__main__":
    main()
