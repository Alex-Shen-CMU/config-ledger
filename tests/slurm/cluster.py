"""Helpers for the opt-in Slurm integration tests (``pytest -m slurm``).

Configuration comes from the environment, since only the person at the cluster knows it:

* ``CONFIG_LEDGER_SLURM_DIR`` (required): a directory on the filesystem shared by the submit
  host and every compute node, at the same path. pytest's ``tmp_path`` is usually node-local.
* ``CONFIG_LEDGER_SLURM_PARTITION``: partition to submit to (cluster default if unset).
* ``CONFIG_LEDGER_SLURM_JOB_PREFIX``: job-name prefix (default ``config_ledger_test_``).
* ``CONFIG_LEDGER_SLURM_RACE_TASKS`` / ``CONFIG_LEDGER_SLURM_RACE_IDENTITIES``: size of the
  claim race (default 32 tasks x 200 identities).
* ``CONFIG_LEDGER_SLURM_RACE_NODES``: comma-separated node names; the race then submits one
  array per node (``--nodelist``) so it crosses NFS clients. Otherwise a single array goes
  wherever the scheduler packs it, which may be one node.
* ``CONFIG_LEDGER_SLURM_TIMEOUT_MIN``: how long to wait for any one set of jobs (default 30).

Completion is detected from ``squeue`` and on-disk artifacts; ``sacct`` (flaky on some
clusters) is only consulted, with retries, for the final state of a deliberately failed task.
Jobs a test submitted are cancelled at teardown if still queued.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
APP = Path(__file__).resolve().parent / "app"
ARRAY_JOB_ID_RE = re.compile(r"^CONFIG_LEDGER_ARRAY_JOB_ID=(\d+)$", re.MULTILINE)


def pythonpath() -> str:
    return os.pathsep.join([str(APP), str(REPO / "src"), os.environ.get("PYTHONPATH", "")])


@dataclass
class Driven:
    """One finished driver invocation (``python <stage>.py -m ...``)."""

    job_id: str
    elapsed_s: float
    stdout: str
    stderr: str


@dataclass
class SlurmCluster:
    base: Path
    partition: str | None
    job_prefix: str
    timeout_s: float
    race_tasks: int
    race_identities: int
    race_nodes: list[str]
    submitted: list[str] = field(default_factory=list)

    # --- submission ----------------------------------------------------------------

    def drive(self, script: str, *overrides: str, name: str) -> Driven:
        """Run a stage's multirun driver with the Slurm launcher; return its array job id."""
        args = [f"outputs={self.base}", f"hydra.sweep.dir={self.base}/hydra/{name}",
                f"hydra.launcher.name={self.job_prefix}{name}", *overrides]
        if self.partition:
            args.append(f"hydra.launcher.partition={self.partition}")
        start = time.monotonic()
        proc = subprocess.run([sys.executable, str(APP / script), "-m", *args], cwd=self.base,
                              env={**os.environ, "PYTHONPATH": pythonpath()},
                              capture_output=True, text=True, timeout=600)
        elapsed = time.monotonic() - start
        assert proc.returncode == 0, f"driver failed:\n{proc.stdout}\n{proc.stderr}"
        ids = ARRAY_JOB_ID_RE.findall(proc.stdout)
        assert len(ids) == 1, f"expected one array job id marker, got {ids}:\n{proc.stdout}"
        self.submitted.append(ids[0])
        return Driven(ids[0], elapsed, proc.stdout, proc.stderr)

    # --- queue state ---------------------------------------------------------------

    def squeue(self, job_id: str) -> list[tuple[str, str, str]]:
        """``(job id, state, reason)`` of every queued task of ``job_id``; empty once it left."""
        proc = subprocess.run(["squeue", "-h", "-j", job_id, "-o", "%i|%T|%r"],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            if "Invalid job id" in proc.stderr:
                return []
            raise RuntimeError(f"squeue failed: {proc.stderr}")
        rows = [line.split("|", 2) for line in proc.stdout.splitlines() if line.strip()]
        return [(job, state, reason) for job, state, reason in rows]

    def wait_until(self, condition: Callable[[], bool], what: str, poll_s: float = 10.0) -> None:
        deadline = time.monotonic() + self.timeout_s
        while not condition():
            if time.monotonic() > deadline:
                pytest.fail(f"timed out after {self.timeout_s:.0f}s waiting for {what}")
            time.sleep(poll_s)

    def wait_jobs(self, *job_ids: str) -> None:
        self.wait_until(lambda: not any(self.squeue(j) for j in job_ids),
                        f"jobs {', '.join(job_ids)} to leave the queue")

    def sacct_states(self, job_id: str, attempts: int = 12,
                     poll_s: float = 10.0) -> dict[str, str] | None:
        """``{<job>_<task>: state}`` from ``sacct``, retried; None if ``sacct`` never answered."""
        for _ in range(attempts):
            proc = subprocess.run(["sacct", "-n", "-X", "-P", "-j", job_id, "-o", "JobID,State"],
                                  capture_output=True, text=True)
            if proc.returncode == 0 and proc.stdout.strip():
                lines = [line for line in proc.stdout.splitlines() if "|" in line]
                states = dict(line.split("|", 1) for line in lines)
                if not any(s in ("PENDING", "RUNNING", "COMPLETING") for s in states.values()):
                    return {k: v.split()[0] for k, v in states.items()}
            time.sleep(poll_s)
        warnings.warn(f"sacct gave no final states for {job_id}; skipping that check", stacklevel=2)
        return None

    def cancel_leftovers(self) -> None:
        live = [j for j in self.submitted if self.squeue(j)]
        if live:
            subprocess.run(["scancel", *live], check=False)
