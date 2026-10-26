"""Runs the toy example's train -> evaluate -> aggregate chain through the real Hydra
launcher (local submitit executor). Slow (~30 s); select with ``pytest -m e2e``."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("hydra")
pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "examples" / "toy_mlp"
pytestmark = pytest.mark.e2e
# Absolute paths so the subprocesses work from a temp cwd whether or not the package is installed.
_ENV = {**os.environ, "PYTHONPATH": os.pathsep.join([str(EXAMPLE), str(REPO / "src"), os.environ.get("PYTHONPATH", "")])}


def _run(script, *args, cwd, outputs, check=True):
    env = _ENV
    cmd = [sys.executable, str(EXAMPLE / script), *args, f"outputs={outputs}", f"ledger.root={outputs}/ledger",
           "trainer.epochs=20"]
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, check=check)


def test_train_evaluate_aggregate(tmp_path):
    outputs = tmp_path / "outputs"
    _run("train.py", "-m", "sweep=hidden_sizes", cwd=tmp_path, outputs=outputs)
    assert len(list((outputs / "train").iterdir())) == 6
    skip = "hydra.launcher.skip_finished=true"
    rerun = _run("train.py", "-m", "sweep=hidden_sizes", skip, cwd=tmp_path, outputs=outputs)
    assert "All 6 grid point(s) skipped; nothing submitted." in rerun.stdout + rerun.stderr
    assert "ARRAY_JOB_ID" not in rerun.stdout
    overlap = _run("train.py", "-m", "sweep=learning_rates", skip, cwd=tmp_path, outputs=outputs)
    assert "Skipped 2 of 6 grid point(s)." in overlap.stdout + overlap.stderr
    assert len(list((outputs / "train").iterdir())) == 10, "2 of 6 grid points overlap and must be reused"
    _run("evaluate.py", "-m", "sweep=hidden_sizes", cwd=tmp_path, outputs=outputs)
    assert len(list((outputs / "ledger" / "sweeps" / "evaluate" / "hidden_sizes").glob("*.csv"))) == 1
    out = subprocess.run([sys.executable, str(EXAMPLE / "aggregate.py"), "--sweep", "hidden_sizes",
                          "--outputs", str(outputs)], env=_ENV, capture_output=True, text=True, check=True).stdout
    assert "WARNING" not in out and out.count("\n") >= 6


def test_manifest_only_then_scoped_parts(tmp_path):
    outputs = tmp_path / "outputs"
    manifests = outputs / "ledger" / "sweeps" / "train" / "hidden_sizes"
    parent = "+ledger.parent_sweep=hidden_sizes"
    missing = _run("train.py", "-m", "sweep=learning_rates", parent, cwd=tmp_path, outputs=outputs,
                   check=False)
    assert missing.returncode != 0 and "ParentSweepManifestMissingError" in missing.stderr

    only = _run("train.py", "-m", "sweep=hidden_sizes", "hydra.launcher.manifest_only=true",
                cwd=tmp_path, outputs=outputs)
    assert "ARRAY_JOB_ID" not in only.stdout
    assert not (outputs / "train").exists(), "manifest_only must not run anything"
    assert len(list(manifests.glob("*.csv"))) == 1

    # One scoped part of the parent's grid: runs, and files no CSV of its own.
    _run("train.py", "-m", "sweep=hidden_sizes", "model.hidden=16", parent, cwd=tmp_path,
         outputs=outputs)
    assert len(list((outputs / "train").iterdir())) == 2
    assert len(list(manifests.glob("*.csv"))) == 1
    assert not (manifests.parent / "learning_rates").exists()

    # learning_rates shares only 2 of its 6 points with hidden_sizes, so it is not a part of it.
    outside = _run("train.py", "-m", "sweep=learning_rates", parent, cwd=tmp_path, outputs=outputs,
                   check=False)
    assert outside.returncode != 0
    assert "4 of 6 grid point(s) are not in parent sweep" in outside.stderr
    assert len(list((outputs / "train").iterdir())) == 2


def test_overrides_to_manifest_records_cli_sweeps(tmp_path):
    outputs = tmp_path / "outputs"
    manifests = outputs / "ledger" / "sweeps" / "train" / "hidden_sizes"
    _run("train.py", "-m", "sweep=hidden_sizes", "trainer.learning_rate=0.01,0.1",
         "hydra.launcher.manifest_only=true", "hydra.launcher.overrides_to_manifest=true",
         cwd=tmp_path, outputs=outputs)
    (csv,) = manifests.glob("*.csv")
    df = pd.read_csv(csv, dtype=str)
    assert len(df) == 12
    assert sorted(df["trainer.learning_rate"].unique()) == ["0.01", "0.1"]
    assert set(df["trainer.epochs"]) == {"20"}, "constant CLI overrides are recorded too"
