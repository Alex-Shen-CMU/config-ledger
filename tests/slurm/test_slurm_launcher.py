"""Integration tests of ``hydra/launcher=config_ledger_slurm`` and the ``os.link()`` claim on a
real Slurm cluster. Opt-in: ``pytest -m slurm`` with ``CONFIG_LEDGER_SLURM_DIR``
set (see ``conftest.py`` and the README's "Testing on a Slurm cluster")."""

from __future__ import annotations

import json
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from tests.slurm.cluster import APP, SlurmCluster, pythonpath

pytest.importorskip("hydra")
pytest.importorskip("submitit")
pytestmark = pytest.mark.slurm

sys.path.insert(0, str(APP))
from slurm_app import DONE, FAIL_SENTINEL, SEEN  # noqa: E402
from slurm_app.race import race_task  # noqa: E402


def manifest(base: Path, namespace: str, sweep: str) -> pd.DataFrame:
    """The one sweep manifest CSV the driver wrote; row ``i`` is array task ``i``."""
    csvs = sorted((base / "ledger" / "sweeps" / namespace / sweep).glob("*.csv"))
    assert len(csvs) == 1, csvs
    return pd.read_csv(csvs[0], dtype=str, keep_default_na=False)


def sidecars(base: Path, namespace: str) -> list[Path]:
    return sorted((base / "ledger" / namespace).glob("[0-9a-f][0-9a-f]/*.json"))


def temp_files(base: Path) -> list[Path]:
    return list(base.rglob("*.tmp.*"))


def test_submit_only_returns_immediately(slurm: SlurmCluster):
    sleep_s = 60
    run = slurm.drive("stage1.py", "sweep=submit_only", name="submit_only")
    queued = slurm.squeue(run.job_id)
    df = manifest(slurm.base, "stage1", "submit_only")
    folders = [slurm.base / "stage1" / f for f in df["folder_id"]]

    # Every task sleeps `sleep_s` before writing done.json: a driver that waited would have
    # seen them, a submit-only one returns before any exists.
    assert run.elapsed_s < sleep_s, run.elapsed_s
    assert not any((f / DONE).exists() for f in folders)
    assert queued, f"array {run.job_id} not in squeue right after submission"
    assert {state for _, state, _ in queued} <= {"PENDING", "RUNNING", "CONFIGURING"}
    assert list(df["work.value"]) == ["1", "2", "3", "4"] and df["folder_id"].nunique() == 4

    slurm.wait_jobs(run.job_id)
    for folder, value in zip(folders, df["work.value"], strict=True):
        assert json.loads((folder / DONE).read_text())["value"] == int(value)
    assert len(sidecars(slurm.base, "stage1")) == 4


def test_grid_points_sharing_an_identity_share_one_folder(slurm: SlurmCluster):
    run = slurm.drive("stage1.py", "sweep=duplicates", name="duplicates")
    slurm.wait_jobs(run.job_id)

    df = manifest(slurm.base, "stage1", "duplicates")
    assert len(df) == 4 and df["folder_id"].nunique() == 1
    (folder_id,) = set(df["folder_id"])
    assert [p.name for p in (slurm.base / "stage1").iterdir()] == [folder_id]
    assert len(sidecars(slurm.base, "stage1")) == 1
    folder = slurm.base / "stage1" / folder_id
    seen = [json.loads(p.read_text()) for p in (folder / SEEN).iterdir()]
    assert sorted(s["task"] for s in seen) == ["0", "1", "2", "3"]
    assert {s["folder_id"] for s in seen} == {folder_id}
    manifest_json = json.loads((folder / "run_manifest.json").read_text())
    assert len(manifest_json["config_hashes"]) == 1
    assert not temp_files(slurm.base / "ledger")


def test_dependency_chaining(slurm: SlurmCluster):
    first = slurm.drive("stage1.py", "sweep=chain", name="chain_stage1")
    second = slurm.drive("stage2.py", "sweep=chain",
                         f"+hydra.launcher.additional_parameters.dependency=afterok:{first.job_id}",
                         name="chain_stage2")

    # Stage 1 tasks sleep 90 s, so stage 2 is still held when we look. The scheduler may need
    # a cycle before it records the reason.
    reasons: set[str] = set()

    def held() -> bool:
        assert slurm.squeue(first.job_id), "stage 1 finished before stage 2's hold was observed"
        rows = slurm.squeue(second.job_id)
        assert rows and all(state == "PENDING" for _, state, _ in rows), rows
        reasons.update(reason for _, _, reason in rows)
        return "Dependency" in reasons

    slurm.wait_until(held, "stage 2 to be held with reason Dependency", poll_s=2.0)

    slurm.wait_jobs(first.job_id, second.job_id)
    df = manifest(slurm.base, "stage2", "chain")
    assert len(df) == 2
    for folder_id in df["folder_id"]:
        done = json.loads((slurm.base / "stage2" / folder_id / DONE).read_text())
        assert done["started"] >= done["stage1_finished"] - 2.0, done  # 2 s for clock skew
        assert done["result"] == 2 * done["value"]


def test_failed_task_is_reraised(slurm: SlurmCluster):
    run = slurm.drive("stage1.py", "sweep=failing", name="failing")
    slurm.wait_jobs(run.job_id)

    df = manifest(slurm.base, "stage1", "failing")
    done = {v: (slurm.base / "stage1" / f / DONE).exists()
            for v, f in zip(df["work.value"], df["folder_id"], strict=True)}
    assert done == {"21": True, "22": False, "23": True}
    failing_task = list(df["work.value"]).index("22")

    # submitit pickles ("error", traceback) only if the task raised out of the launcher's
    # __call__; a swallowed failure would have pickled ("success", JobReturn(FAILED)).
    results = {}
    for path in (slurm.base / "hydra" / "failing").rglob("*_result.pkl"):
        match = re.fullmatch(rf"{run.job_id}_(\d+)(?:_\d+)?_result\.pkl", path.name)
        if match:
            with open(path, "rb") as file:
                results[int(match.group(1))] = pickle.load(file)
    assert sorted(results) == [0, 1, 2], results
    assert results[failing_task][0] == "error" and FAIL_SENTINEL in results[failing_task][1]
    assert all(results[i][0] == "success" for i in results if i != failing_task)

    states = slurm.sacct_states(run.job_id)
    if states is not None:
        failed = f"{run.job_id}_{failing_task}"
        assert states[failed] == "FAILED", states
        assert all(s == "COMPLETED" for k, s in states.items() if k != failed), states


def test_concurrent_claim_race(slurm: SlurmCluster, monkeypatch):
    """Many tasks, possibly on several nodes, resolve the same identities at the same instant
    in a fresh ledger; each identity must end up with exactly one sidecar and one folder."""
    import submitit

    n_tasks, n_ids = slurm.race_tasks, slurm.race_identities
    root = slurm.base / "race"
    nodes = slurm.race_nodes or [None]
    counts = [n_tasks // len(nodes) + (i < n_tasks % len(nodes)) for i in range(len(nodes))]
    # sbatch exports the submitting environment, so tasks can import slurm_app.
    monkeypatch.setenv("PYTHONPATH", pythonpath())

    args = (str(root), n_tasks, n_ids, 60.0, 1200.0)
    race_jobs = []
    for node, count in zip(nodes, counts, strict=True):
        executor = submitit.AutoExecutor(folder=root / "submitit" / "%j", cluster="slurm")
        params: dict[str, Any] = {"slurm_partition": slurm.partition} if slurm.partition else {}
        if node:
            params["slurm_additional_parameters"] = {"nodelist": node}
        # tasks_per_node: without it srun may start one rank per hardware thread of the core.
        executor.update_parameters(name=f"{slurm.job_prefix}race", timeout_min=30, cpus_per_task=1,
                                   tasks_per_node=1, mem_gb=1, **params)
        jobs = executor.map_array(race_task, *zip(*[args] * count, strict=True))
        race_jobs.append(jobs[0].job_id.split("_")[0])
    slurm.submitted.extend(race_jobs)

    reports_dir = root / "reports"
    drained_at: list[float] = []

    def finished() -> bool:
        if reports_dir.is_dir() and len(os.listdir(reports_dir)) >= n_tasks:
            return True
        # Once the jobs are gone, keep looking for a while: this host's NFS client may serve a
        # cached directory listing for up to `acdirmax` (60 s by default).
        if not drained_at and not any(slurm.squeue(job) for job in race_jobs):
            drained_at.append(time.monotonic())
        return bool(drained_at) and time.monotonic() - drained_at[0] > 120

    slurm.wait_until(finished, f"{n_tasks} race reports")
    reports = [json.loads(p.read_text()) for p in reports_dir.glob("*.json")]
    assert len(reports) == n_tasks, \
        f"only {len(reports)} of {n_tasks} race tasks reported; see {root / 'submitit'}"
    hosts = sorted({r["host"] for r in reports})
    won = sum(r["claims_won"] for r in reports)
    lost = sum(r["claims_lost"] for r in reports)
    print(f"race: {len(reports)} tasks on {hosts}; claims won={won} lost={lost}; "
          f"start lateness max={max(r['lateness_s'] for r in reports):.3f}s; "
          f"slowest task {max(r['elapsed_s'] for r in reports):.1f}s")

    assert not any(r["barrier_timed_out"] for r in reports), \
        "not all race tasks were running at once; the cluster could not co-schedule them"
    if slurm.race_nodes:
        assert len(hosts) == len(slurm.race_nodes), hosts
    for index in map(str, range(n_ids)):
        assert len({r["folder_ids"][index] for r in reports}) == 1, f"identity {index} split"
    assert won == n_ids, "every identity must have exactly one winning os.link()"
    assert lost > 0, "no task ever lost a claim: the race was not exercised"

    assert len(sidecars(root, "race")) == n_ids
    folders = sorted((root / "outputs").iterdir())
    assert {f.name for f in folders} == {fid for r in reports for fid in r["folder_ids"].values()}
    for folder in folders:
        run_manifest = json.loads((folder / "run_manifest.json").read_text())
        assert run_manifest["folder_id"] == folder.name and len(run_manifest["config_hashes"]) == 1
    assert len(list((root / "ledger" / "race" / "_versions").glob("*/*/fresh/*.json"))) == n_ids
    assert not temp_files(root / "ledger") and not temp_files(root / "outputs")
