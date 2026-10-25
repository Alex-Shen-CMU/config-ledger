"""Array task for the concurrent claim race: many tasks claim the same identities at once.

Submitted directly with submitit (no Hydra driver), because the launcher's driver claims every
grid point while writing the sweep manifest, after which tasks only ever *find* a sidecar.

Synchronization: each task touches ``ready/<task>``, waits until it sees ``n_tasks`` of them,
then sleeps until ``max(ready mtimes) + margin_s``. The mtimes are set by the file server, so
every task computes the same start instant; the margin absorbs NFS directory caching (a task
may see the last ready file late). Node clocks only need to agree to well under a claim's
duration for the claims to overlap; each task reports how late it actually started.
"""

from __future__ import annotations

import os
import socket
import time
from pathlib import Path

from config_ledger import Ledger, store
from config_ledger._files import write_json_atomic
from slurm_app.schema import RaceNode

NAMESPACE = "race"


def task_name() -> str:
    job = os.environ.get("SLURM_ARRAY_JOB_ID", str(os.getpid()))
    return f"{job}_{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}"


def race_task(root: str, n_tasks: int, n_identities: int, margin_s: float,
              barrier_timeout_s: float) -> None:
    base = Path(root)
    me = task_name()
    ready = base / "ready"
    ready.mkdir(parents=True, exist_ok=True)
    (ready / me).touch()
    deadline = time.time() + barrier_timeout_s
    timed_out = False
    while True:
        names = os.listdir(ready)
        if len(names) >= n_tasks:
            break
        if time.time() > deadline:
            timed_out = True
            break
        time.sleep(0.2)
    go = max(os.stat(ready / name).st_mtime for name in names) + margin_s
    time.sleep(max(0.0, go - time.time()))
    lateness = time.time() - go

    # Count, per sidecar claim this task attempted (a `find` miss), whether its own payload
    # won the os.link() or it adopted another task's.
    won = lost = 0
    claim_file = store.claim_file

    def tracking_claim(path, payload):
        nonlocal won, lost
        stored = claim_file(path, payload)
        if stored["folder_id"] == payload["folder_id"]:
            won += 1
        else:
            lost += 1
        return stored

    store.claim_file = tracking_claim
    try:
        ledger = Ledger(base / "ledger")
        folder_ids = {}
        for index in range(n_identities):
            _, entry = ledger.resolve_output_folder(NAMESPACE, base / "outputs",
                                                    RaceNode(index=index))
            folder_ids[str(index)] = entry.folder_id
    finally:
        store.claim_file = claim_file
    write_json_atomic(base / "reports" / f"{me}.json", {
        "task": me,
        "host": socket.gethostname(),
        "barrier_timed_out": timed_out,
        "ready_seen": len(names),
        "lateness_s": lateness,
        "elapsed_s": time.time() - go,
        "claims_won": won,
        "claims_lost": lost,
        "folder_ids": folder_ids,
    })
