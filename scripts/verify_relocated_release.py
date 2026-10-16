"""Check relocated ZIP sources using existing dependency wheels, not an independent install."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from verify_portable_release import verify

ROOT = Path(__file__).resolve().parents[1]


def run(dependency_root: Path = ROOT):
    archive = ROOT / "releases/pickplace-lab-research-v1.zip"
    verify(archive)
    base_python = sys._base_executable
    dependency_root = dependency_root.resolve()
    original_site = dependency_root / ".venv/lib/python3.11/site-packages"
    panda_site = dependency_root / "benchmarks/robosuite/.venv/lib/python3.11/site-packages"
    if not original_site.is_dir() or not panda_site.is_dir():
        raise RuntimeError(
            "This relocation check expects the two prepared Python 3.11 environments"
        )
    checks = []
    with tempfile.TemporaryDirectory(prefix="pickplace-relocation-") as temporary:
        destination = Path(temporary)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(destination)
        relocated = destination / "pickplace-lab-research-v1"
        commands = [
            (
                "source_resolution",
                original_site,
                [
                    "-c",
                    "import pickplace,pathlib; p=pathlib.Path(pickplace.__file__).resolve(); assert p.is_relative_to(pathlib.Path.cwd()); print(p.relative_to(pathlib.Path.cwd()))",
                ],
            ),
            ("relocated_manifest", original_site, ["scripts/verify_manifest.py"]),
            (
                "original_and_filter_tests",
                original_site,
                ["-m", "pytest", "-q", "tests", "benchmarks/latency_guard/test_guard.py"],
            ),
            ("original_artifacts", original_site, ["scripts/verify_artifacts.py"]),
            ("research_artifacts", original_site, ["scripts/verify_study.py"]),
            (
                "panda_tests",
                panda_site,
                [
                    "-m",
                    "pytest",
                    "-q",
                    "benchmarks/robosuite/test_learning.py",
                    "benchmarks/robosuite/test_evaluation.py",
                ],
            ),
        ]
        for label, dependency_site, command in commands:
            environment = os.environ.copy()
            environment["PYTHONPATH"] = os.pathsep.join(
                [str(relocated / "src"), str(dependency_site)]
            )
            environment["PYTHONNOUSERSITE"] = "1"
            environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
            completed = subprocess.run(
                [base_python, "-S", *command],
                cwd=relocated,
                env=environment,
                text=True,
                capture_output=True,
                timeout=180,
                check=False,
            )
            # Do not store temporary personal-machine absolute paths in the portable report.
            stdout = completed.stdout.replace(str(destination), "<temporary-directory>")
            stderr = completed.stderr.replace(str(destination), "<temporary-directory>")
            for source, replacement in (
                (ROOT, "<project-root>"),
                (dependency_root, "<original-dependency-root>"),
            ):
                stdout = stdout.replace(str(source), replacement)
                stderr = stderr.replace(str(source), replacement)
            checks.append(
                {
                    "check": label,
                    "returncode": completed.returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                }
            )
            print(f"{label}: {'PASS' if completed.returncode == 0 else 'FAIL'}", flush=True)
            if completed.returncode:
                raise RuntimeError(f"Relocated check failed: {label}\n{stdout}\n{stderr}")
    report = {
        "scope": "Relocated source/data/weights and isolated import resolution; dependencies reused from local environments. NOT a fresh dependency installation, cross-platform test or independent replication.",
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "checks": checks,
    }
    path = ROOT / "artifacts/relocation-verification.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dependency-root", type=Path, default=ROOT)
    run(parser.parse_args().dependency_root)
