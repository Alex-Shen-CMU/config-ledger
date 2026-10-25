"""The ledger: a lock-free, database-free map from config identity to output folder.

Layout, one small JSON sidecar per identity::

    <root>/<namespace>/<hash[:2]>/<hash>.json
        {"folder_id": "<uuid4>", "namespace": ..., "hash": ...,
         "config": {<materialized row>}, "tree": {<materialized tree>}, "created": ...}

inside every output folder, a reverse index ``run_manifest.json``::

    {"folder_id": ..., "namespace": ..., "config_hashes": [<oldest>, ..., <latest>],
     "configs": {<hash>: {"config": {<row>}, "tree": {<tree>}}, ...},
     "config": {<latest row>}, "tree": {<latest tree>}, "updated": ...}

listing every hash that resolves to the folder with the row and tree it was registered with,
so the whole ledger (sidecars and version index) can be rebuilt by walking the outputs
(``config-ledger rebuild``, :func:`config_ledger.maintenance.rebuild`); and, per namespace,
the version index (:mod:`config_ledger.versions`).

Why files and ``os.link()`` rather than SQLite/LMDB: the ledger lives on the same NFS export
the jobs write to, and both of those rely on POSIX byte-range locks / mmap that are
unreliable there. Sidecars need only *insert-if-absent, never update*, and an atomic
``os.link()`` gives exactly that without a lock file: two Slurm array tasks that miss on the
same identity simultaneously each write a temp file and race to link it; the loser reads the
winner's record and adopts its folder id. (Version index statuses change by ``os.rename``.)

Folder ids are opaque uuids, not the hash. That is what makes migration cheap: after a
version bump, ``config-ledger upgrade`` registers the *new* hash against the folder that
already holds the results instead of moving anything, so one folder can legitimately be
reachable by several hashes.

A sidecar means "this identity owns this folder", **never** "the run finished". Completion
must be judged from the artifacts the run writes.

Every materialized value is hashed, so a hit cannot disagree with the current
config: the defaults-excluded policy and its drift verification were removed.

Status (2026-09-10): ported from an internal project as a class with an
explicit root. 2026-10-17: full-value identity only, stored trees, version index.
2026-10-18: the run manifest keeps the row and tree of every hash, not only the latest.
2026-10-18: busy run manifest replaces on NFS are retried.
"""

from __future__ import annotations

import os
import random
import time
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config_ledger._files import claim_file, is_busy, read_json, utcnow, write_json_atomic
from config_ledger.identity import Identity, Row, compute_identity
from config_ledger.versions import VersionIndex, version_check_skipped

__all__ = ["RUN_MANIFEST_FILENAME", "Entry", "Ledger", "claim_file"]

RUN_MANIFEST_FILENAME = "run_manifest.json"
# Retries of a busy run manifest replace (see `Ledger.write_run_manifest`); with jittered
# exponential backoff capped at 1 s these span roughly 15 s in the worst case.
RUN_MANIFEST_ATTEMPTS = 20


@dataclass(frozen=True)
class Entry:
    """One resolved ledger record."""

    namespace: str
    hash: str
    folder_id: str
    materialized: Row
    created: str


class Ledger:
    """Identity -> folder map rooted at one directory, split into namespaces.

    A *namespace* is typically the name of the script that produces the outputs
    (``"train"``, ``"evaluate"``): the same config can legitimately own one folder per stage.
    Pass the namespace explicitly rather than deriving it from a job name that people
    override for cosmetic reasons (a lesson learned the hard way).
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)

    # --- paths ---------------------------------------------------------------------

    def namespace_dir(self, namespace: str) -> Path:
        return self.root / namespace

    def sidecar_path(self, namespace: str, hash_key: str) -> Path:
        """Two-character fanout keeps NFS directory listings fast as the ledger grows."""
        return self.namespace_dir(namespace) / hash_key[:2] / f"{hash_key}.json"

    def versions(self, namespace: str) -> VersionIndex:
        """The version index of ``namespace``."""
        return VersionIndex(self.root, namespace)

    # --- identity ------------------------------------------------------------------

    def identity(self, config: Any) -> Identity:
        return compute_identity(config)

    def check_configs(self, namespace: str, configs: Iterable[Any]) -> None:
        """Stale discovery for a whole batch of configs at once (a sweep): one
        :meth:`~config_ledger.versions.VersionIndex.check` over the union of every
        ``(class, version)`` any of them uses. Raises
        :class:`~config_ledger.versions.StaleRunsError` and friends."""
        keys: set[tuple[str, str]] = set()
        for config in configs:
            keys.update(self.identity(config).version_nodes())
        self.versions(namespace).check(keys)

    # --- lookups -------------------------------------------------------------------

    def _entry(self, namespace: str, stored: Mapping[str, Any]) -> Entry:
        return Entry(
            namespace=namespace,
            hash=str(stored["hash"]),
            folder_id=str(stored["folder_id"]),
            materialized=dict(stored.get("config", {})),
            created=str(stored.get("created", "")),
        )

    def find(self, namespace: str, identity: Identity) -> Entry | None:
        """Read-only lookup; None if this identity was never registered in ``namespace``."""
        stored = read_json(self.sidecar_path(namespace, identity.hash))
        return None if stored is None else self._entry(namespace, stored)

    def _payload(self, namespace: str, identity: Identity, folder_id: str) -> dict[str, Any]:
        return {
            "folder_id": folder_id,
            "namespace": namespace,
            "hash": identity.hash,
            "config": identity.materialized,
            "tree": identity.tree,
            "created": utcnow(),
        }

    def get_or_create(self, namespace: str, identity: Identity) -> Entry:
        """The folder owned by this identity, minting a fresh uuid4 if it is new."""
        return self.register(namespace, identity, str(uuid.uuid4()))

    def register(self, namespace: str, identity: Identity, folder_id: str) -> Entry:
        """Point ``identity`` at an *existing* folder (or a freshly minted id).

        Returns the entry actually stored, whose ``folder_id`` may differ from the one passed
        if another writer won the race or the identity was already registered.
        """
        existing = self.find(namespace, identity)
        if existing is not None:
            return existing
        stored = claim_file(self.sidecar_path(namespace, identity.hash),
                            self._payload(namespace, identity, folder_id))
        return self._entry(namespace, stored)

    # --- reverse index --------------------------------------------------------------

    def write_run_manifest(self, output_folder: str | os.PathLike[str], entry: Entry,
                           identity: Identity) -> Path:
        """Record, inside the output folder, its own id and every hash resolving to it.

        Several hashes legitimately map to one folder once migrations accumulate, so an
        existing manifest is extended rather than overwritten. Each hash keeps the row and tree
        it was registered with (``configs``), so its sidecar can be rebuilt; the top-level row
        and tree are the latest registration's, so an upgrade starts from the newest one.

        Concurrent tasks resolving the same identity all reach here until one of them has
        written the hash. On NFS their replaces can fail as busy; each retry
        re-reads first and stops as soon as the hash is there. This is still a
        read-modify-write: two *different* hashes registered at the same instant can lose one.
        """
        path = Path(output_folder) / RUN_MANIFEST_FILENAME
        for attempt in range(RUN_MANIFEST_ATTEMPTS):
            manifest = read_json(path) or {
                "folder_id": entry.folder_id,
                "namespace": entry.namespace,
                "config_hashes": [],
                "configs": {},
            }
            hashes = list(manifest.get("config_hashes", []))
            if identity.hash in hashes:
                return path
            manifest["config_hashes"] = [*hashes, identity.hash]
            manifest["configs"] = {**manifest.get("configs", {}),
                                   identity.hash: {"config": identity.materialized,
                                                   "tree": identity.tree}}
            manifest["config"] = identity.materialized
            manifest["tree"] = identity.tree
            manifest["updated"] = utcnow()
            try:
                write_json_atomic(path, manifest)
                return path
            except OSError as error:
                if not is_busy(error) or attempt == RUN_MANIFEST_ATTEMPTS - 1:
                    raise
            time.sleep(random.uniform(0.5, 1.0) * min(1.0, 0.02 * 2**attempt))
        raise AssertionError("unreachable")

    # --- one-stop helper ------------------------------------------------------------

    def resolve_output_folder(self, namespace: str, parent_dir: str | os.PathLike[str],
                              config: Any) -> tuple[Path, Entry]:
        """Check versions, hash, claim a folder id, record the reverse and version indexes.

        Raises :class:`~config_ledger.versions.StaleRunsError` before claiming anything if an
        old version of a class this config uses still has fresh runs (skipped inside
        :func:`~config_ledger.versions.skip_version_check`, i.e. in sweep tasks whose driver
        already checked). Returns ``(output_folder, entry)``; the folder is created.
        """
        identity = self.identity(config)
        index = self.versions(namespace)
        nodes = identity.version_nodes()
        if not version_check_skipped():
            index.check(nodes)
        entry = self.get_or_create(namespace, identity)
        output_folder = Path(parent_dir) / entry.folder_id
        output_folder.mkdir(parents=True, exist_ok=True)
        self.write_run_manifest(output_folder, entry, identity)
        index.record(nodes, entry.folder_id, output_folder)
        return output_folder, entry

    def lookup_output_folder(self, namespace: str, parent_dir: str | os.PathLike[str],
                             config: Any) -> Path | None:
        """Read-only counterpart of :meth:`resolve_output_folder`, for a downstream stage
        that needs an upstream stage's folder without claiming anything."""
        entry = self.find(namespace, self.identity(config))
        return None if entry is None else Path(parent_dir) / entry.folder_id
