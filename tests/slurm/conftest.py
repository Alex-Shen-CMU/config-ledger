"""The ``slurm`` fixture; configuration and helpers are documented in :mod:`tests.slurm.cluster`."""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pytest

from tests.slurm.cluster import SlurmCluster


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name) or default)


@pytest.fixture
def slurm(request) -> Iterator[SlurmCluster]:
    if shutil.which("sbatch") is None or shutil.which("squeue") is None:
        pytest.skip("Slurm (sbatch/squeue) not available")
    root = os.environ.get("CONFIG_LEDGER_SLURM_DIR")
    if not root:
        pytest.skip("set CONFIG_LEDGER_SLURM_DIR to a directory shared with the compute nodes")
    base = Path(root).expanduser().resolve() / f"{datetime.now():%Y%m%dT%H%M%S}_{request.node.name}"
    base.mkdir(parents=True)
    cluster = SlurmCluster(
        base=base,
        partition=os.environ.get("CONFIG_LEDGER_SLURM_PARTITION") or None,
        job_prefix=os.environ.get("CONFIG_LEDGER_SLURM_JOB_PREFIX", "config_ledger_test_"),
        timeout_s=60.0 * _env_int("CONFIG_LEDGER_SLURM_TIMEOUT_MIN", 30),
        race_tasks=_env_int("CONFIG_LEDGER_SLURM_RACE_TASKS", 32),
        race_identities=_env_int("CONFIG_LEDGER_SLURM_RACE_IDENTITIES", 200),
        race_nodes=[n for n in os.environ.get("CONFIG_LEDGER_SLURM_RACE_NODES", "").split(",")
                    if n],
    )
    print(f"\nslurm test dir: {base}")
    yield cluster
    cluster.cancel_leftovers()
