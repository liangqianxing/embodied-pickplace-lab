"""Verify the complete study, source snapshot, summary arithmetic and replay traces."""

import hashlib
import json
from pathlib import Path

import numpy as np

from pickplace.study_analysis import load_cube

root = Path(__file__).resolve().parents[1]
output = root / "artifacts/study/v1"
protocol, models, conditions, cube, trials = load_cube(output)
summary = json.loads((output / "summary.json").read_text())
assert summary["total_trials"] == cube.size == len(trials)
assert cube.size == 15000
for name, expected in protocol["source_sha256"].items():
    actual = hashlib.sha256((output / "source_snapshot" / name).read_bytes()).hexdigest()
    assert actual == expected, f"Frozen source mismatch: {name}"
for row in summary["rows"]:
    rates = cube[models.index(row["model"]), :, conditions.index(row["condition"]), :].mean(1)
    np.testing.assert_array_equal(row["per_seed_rates"], rates)
    assert abs(row["mean"] - rates.mean()) < 1e-12
    assert abs(row["sd_across_training_runs"] - rates.std(ddof=1)) < 1e-12
    assert row["successes"] == int(
        cube[models.index(row["model"]), :, conditions.index(row["condition"]), :].sum()
    )
shifted = [i for i, name in enumerate(conditions) if name != "id"]
contrast = (
    cube[models.index("relative_noise32"), :, shifted, :].mean()
    - cube[models.index("raw32"), :, shifted, :].mean()
)
assert abs(summary["primary"]["mean_difference"] - contrast) < 1e-12
replays = json.loads((output / "replay_index.json").read_text())
for item in replays:
    path = output / item["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    replay = json.loads(path.read_text())
    group = f"{item['model']}-{item['training_seed']}-{item['condition']}"
    records = json.loads((output / f"trials/{group}.json").read_text())["trials"]
    first = next(row for row in records if row["success"] == item["success"])
    assert item["seed"] == first["seed"]
    assert len(replay["frames"]) == replay["steps"] + 1 == item["frames"]
    for key, value in first.items():
        assert replay[key] == value
print(
    f"Verified {len(trials):,} trials, {len(summary['rows'])} summary cells, "
    f"{len(replays)} replay records and {len(protocol['source_sha256'])} frozen sources."
)
