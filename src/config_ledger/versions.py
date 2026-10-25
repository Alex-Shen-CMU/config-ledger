"""The version index: which runs use which ``(config class, version chain)``, and in what status.

A version bump used to leave old runs reachable only by a full scan of every run manifest.
The index records, per namespace, every run under every versioned class it uses::

    <root>/<namespace>/_versions/<_target_>/<version slug>/
        meta.json                 {"target", "version_chain", "fields", "created"}
        fresh/<folder_id>.json    {"folder_id", "namespace", "target", "version_chain",
        stale/<folder_id>.json     "nodes": [<row prefixes>], "output_folder", "created"}
        migrated/<folder_id>.json

A run appears once in the map of every ``(class, version)`` its config tree contains. Maps are
created lazily by the first run that uses them (:meth:`VersionIndex.record`).

**Status is the subdirectory an entry sits in**, and belongs to that one config of the run, not
to the run: a run can be ``stale`` in ``Trainer:1`` and ``fresh`` in ``Data:1``.

* ``fresh``: a run whose results are current for this version. Under an old version this is
  what blocks new runs: the user has not yet decided what to do with it.
* ``stale``: the user declared these results outdated (``config-ledger mark-stale``). Allowed
  to sit under an old version; ``config-ledger purge-stale`` deletes such runs.
* ``migrated``: the run's config was upgraded (``config-ledger upgrade``) and the run now
  appears under the live version's map. Ignored.

A status change is one ``os.rename`` (atomic on NFS, like the store's ``os.link``), and "does
this old version still have fresh runs?" is one directory listing.

**Stale discovery.** :meth:`VersionIndex.check` refuses a run (or a whole sweep, checked once
for the union of its classes) while any non-live version of one of its classes still has
``fresh`` entries, raising :class:`StaleRunsError`. A stored version *newer* than the live
code (an old checkout) raises :class:`NewerVersionError` instead.

**Schema guard.** ``meta.json`` of the live version records the class's identity-bearing
field names when the map is created. Since every value is hashed, adding or removing a field
without bumping ``VERSION`` would change every hash with no stale signal at all; any later
run whose class declares a different field set under the same version raises
:class:`SchemaChangedError`. This compares declared fields, never source code.

Status (2026-10-17): introduced.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import os
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config_ledger._files import claim_file, read_json, utcnow, write_json_atomic
from config_ledger.versioning import (
    INSTANTIATE_KEYS,
    identity_excluded_fields,
    parse_chain,
    resolve_target,
    schema_fields,
    target_version,
)

FRESH = "fresh"
STALE = "stale"
MIGRATED = "migrated"
STATUSES = (FRESH, STALE, MIGRATED)

VERSIONS_DIRNAME = "_versions"
META_FILENAME = "meta.json"

VersionNodes = Mapping[tuple[str, str], Iterable[str]]
"""``(_target_, version chain) -> row prefixes`` (:func:`config_ledger.identity.version_nodes`)."""


class StaleRunsError(RuntimeError):
    """An old version of a class this run uses still has ``fresh`` runs."""


class NewerVersionError(RuntimeError):
    """The index holds runs under a version newer than the live code (an old checkout?)."""


class SchemaChangedError(RuntimeError):
    """A class's identity-bearing fields changed but its version chain did not."""


_SKIP_CHECK: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "config_ledger_skip_version_check", default=False)


@contextlib.contextmanager
def skip_version_check() -> Iterator[None]:
    """Skip :meth:`VersionIndex.check` in :meth:`~config_ledger.store.Ledger.resolve_output_folder`.

    For sweep tasks whose driver already checked the whole grid once
    (:meth:`~config_ledger.store.Ledger.check_configs`). Runs are still recorded in the index,
    and the schema guard still applies."""
    token = _SKIP_CHECK.set(True)
    try:
        yield
    finally:
        _SKIP_CHECK.reset(token)


def version_check_skipped() -> bool:
    return _SKIP_CHECK.get()


def version_slug(chain: str) -> str:
    """Filesystem-safe form of a version chain: ``"Trainer:1|Gold:2"`` -> ``"Trainer-1+Gold-2"``."""
    return chain.replace(":", "-").replace("|", "+")


def compare_chains(stored: str, live: str) -> str:
    """``"live"`` if equal, ``"newer"`` if any class in both chains is at a higher version in
    ``stored``, else ``"older"`` (including chains whose classes differ: a restructured
    hierarchy is a change of meaning)."""
    if stored == live:
        return "live"
    old, new = parse_chain(stored), parse_chain(live)
    if any(old[name] > new[name] for name in old.keys() & new.keys()):
        return "newer"
    return "older"


def live_chain(target: str) -> str | None:
    """The live version chain of ``target``, or None if the class no longer imports."""
    try:
        return target_version(target)
    except (ImportError, AttributeError):
        return None


def identity_fields(cls: type) -> list[str] | None:
    """Sorted identity-bearing field names of ``cls`` (what the schema guard records)."""
    declared = schema_fields(cls)
    if declared is None:
        return None
    excluded = identity_excluded_fields(cls) | INSTANTIATE_KEYS
    return sorted(name for name in declared if name not in excluded)


@dataclass(frozen=True)
class IndexEntry:
    """One run's record in one ``(class, version)`` map."""

    namespace: str
    target: str
    version_chain: str
    status: str
    folder_id: str
    output_folder: str
    nodes: tuple[str, ...]
    path: Path


@dataclass(frozen=True)
class VersionInfo:
    """One ``(class, version)`` map and its entry counts."""

    namespace: str
    target: str
    version_chain: str
    fields: list[str] | None
    counts: dict[str, int]
    directory: Path


class VersionIndex:
    """The version index of one namespace under a ledger root."""

    def __init__(self, root: str | os.PathLike[str], namespace: str) -> None:
        self.root = Path(root)
        self.namespace = namespace
        self.directory = self.root / namespace / VERSIONS_DIRNAME

    # --- paths ---------------------------------------------------------------------

    def version_dir(self, target: str, chain: str) -> Path:
        return self.directory / target / version_slug(chain)

    def entry_path(self, target: str, chain: str, status: str, folder_id: str) -> Path:
        return self.version_dir(target, chain) / status / f"{folder_id}.json"

    # --- reading -------------------------------------------------------------------

    def _version_dirs(self, target: str | None = None) -> Iterator[tuple[str, Path]]:
        if not self.directory.is_dir():
            return
        targets = [target] if target is not None else sorted(os.listdir(self.directory))
        for name in targets:
            target_dir = self.directory / name
            if not target_dir.is_dir():
                continue
            for slug in sorted(os.listdir(target_dir)):
                if (target_dir / slug).is_dir():
                    yield name, target_dir / slug

    def versions(self, target: str | None = None) -> list[VersionInfo]:
        infos = []
        for name, directory in self._version_dirs(target):
            meta = read_json(directory / META_FILENAME)
            if meta is None:
                continue
            counts = {status: _count(directory / status) for status in STATUSES}
            infos.append(VersionInfo(self.namespace, name, str(meta["version_chain"]),
                                     meta.get("fields"), counts, directory))
        return infos

    def entries(self, target: str | None = None, chain: str | None = None,
                status: str | None = None) -> Iterator[IndexEntry]:
        for info in self.versions(target):
            if chain is not None and info.version_chain != chain:
                continue
            for state in [status] if status is not None else STATUSES:
                folder = info.directory / state
                if not folder.is_dir():
                    continue
                for filename in sorted(os.listdir(folder)):
                    if not filename.endswith(".json") or ".tmp." in filename:
                        continue
                    stored = read_json(folder / filename)
                    if stored is not None:
                        yield _entry(self.namespace, info, state, stored, folder / filename)

    def folder_entries(self, folder_id: str) -> list[IndexEntry]:
        """Every entry of one run, in every map and status."""
        found = []
        for info in self.versions():
            for status in STATUSES:
                path = info.directory / status / f"{folder_id}.json"
                stored = read_json(path)
                if stored is not None:
                    found.append(_entry(self.namespace, info, status, stored, path))
        return found

    # --- stale discovery -------------------------------------------------------------

    def check(self, nodes: VersionNodes | Iterable[tuple[str, str]]) -> None:
        """Raise if a run using these ``(target, live chain)`` pairs must not start.

        Checks every class once, however many runs or nodes use it. See the module docstring."""
        keys = set(nodes)
        older: list[str] = []
        newer: list[str] = []
        for target, chain in sorted(keys):
            self._guard_schema(target, chain)
            for info in self.versions(target):
                if info.version_chain == chain or not info.counts[FRESH]:
                    continue
                line = (f"  {target} @ {info.version_chain}: {info.counts[FRESH]} fresh run(s) "
                        f"(live: {chain})")
                is_newer = compare_chains(info.version_chain, chain) == "newer"
                (newer if is_newer else older).append(line)
        where = f"--ledger {self.root} --namespace {self.namespace}"
        if newer:
            raise NewerVersionError(
                f"namespace {self.namespace!r} has runs under versions newer than this code:\n"
                + "\n".join(newer)
                + "\nIs this an old checkout? Update the code, or inspect with "
                f"`config-ledger versions {where}`."
            )
        if older:
            raise StaleRunsError(
                f"namespace {self.namespace!r} still has fresh runs under old versions of classes "
                "this run uses:\n" + "\n".join(older)
                + "\nDecide what they are before running under the new version, with "
                f"`config-ledger <command> {where}`:\n"
                "  versions               inspect\n"
                "  upgrade --apply        carry them forward via upgrade_config\n"
                "  mark-stale --apply     declare them outdated"
            )

    def diagnose(self, nodes: VersionNodes | Iterable[tuple[str, str]],
                 found: bool) -> list[str]:
        """Read-only counterpart of :meth:`check` for lookups: problems as warning
        lines instead of exceptions. Newer versions and schema changes are always reported;
        old versions with fresh or stale runs only on a miss (``found=False``), where they are
        the likely reason: the run exists under an old version and was not upgraded."""
        warnings: list[str] = []
        for target, chain in sorted(set(nodes)):
            try:
                self._guard_schema(target, chain)
            except SchemaChangedError as error:
                warnings.append(str(error))
            for info in self.versions(target):
                if info.version_chain == chain:
                    continue
                if compare_chains(info.version_chain, chain) == "newer":
                    warnings.append(
                        f"{target} @ {info.version_chain} is newer than this code ({chain}); "
                        "is this an old checkout? The hash may not match what newer code computes.")
                elif not found and (info.counts[FRESH] or info.counts[STALE]):
                    warnings.append(
                        f"{target} @ {info.version_chain} (live: {chain}) has "
                        f"{info.counts[FRESH]} fresh and {info.counts[STALE]} stale run(s); "
                        "this config may have run under the old version. Inspect with "
                        f"`config-ledger versions --ledger {self.root} --namespace "
                        f"{self.namespace}`, carry runs forward with `config-ledger upgrade`.")
        return warnings

    # --- writing -------------------------------------------------------------------

    def _meta(self, target: str, chain: str) -> dict[str, Any]:
        """``meta.json`` of a map, claimed on first use. Fields are recorded only for the live
        version (an old version's fields are unknowable from the live class)."""
        path = self.version_dir(target, chain) / META_FILENAME
        stored = read_json(path)
        if stored is not None:
            return stored
        fields = identity_fields(resolve_target(target)) if live_chain(target) == chain else None
        return claim_file(path, {"target": target, "version_chain": chain, "fields": fields,
                                 "created": utcnow()})

    def _guard_schema(self, target: str, chain: str) -> None:
        meta = read_json(self.version_dir(target, chain) / META_FILENAME)
        if meta is None:
            return
        _compare_fields(target, chain, meta.get("fields"))

    def record(self, nodes: VersionNodes, folder_id: str,
               output_folder: str | os.PathLike[str], status: str = FRESH) -> None:
        """Add a run to the map of every ``(class, version)`` it uses, as ``status`` (``fresh``
        for a new run), unless it is already in that map under any status."""
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}")
        for (target, chain), prefixes in nodes.items():
            meta = self._meta(target, chain)
            if live_chain(target) == chain:
                _compare_fields(target, chain, meta.get("fields"))
            if any(self.entry_path(target, chain, s, folder_id).exists() for s in STATUSES):
                continue
            claim_file(self.entry_path(target, chain, status, folder_id), {
                "folder_id": folder_id,
                "namespace": self.namespace,
                "target": target,
                "version_chain": chain,
                "nodes": sorted(prefixes),
                "output_folder": str(Path(output_folder).resolve()),
                "created": utcnow(),
            })

    def move(self, entry: IndexEntry, status: str) -> IndexEntry:
        """Change one entry's status with a single ``os.rename``."""
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}")
        destination = self.entry_path(entry.target, entry.version_chain, status, entry.folder_id)
        if destination == entry.path:
            return entry
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.rename(entry.path, destination)
        return dataclasses.replace(entry, status=status, path=destination)

    def relocate(self, entry: IndexEntry, output_folder: str | os.PathLike[str]) -> IndexEntry:
        """Point an entry at the folder's new location (outputs moved; ``config-ledger rebuild``).
        A whole-file replace in place: the entry keeps its status."""
        stored = read_json(entry.path)
        if stored is None:
            raise FileNotFoundError(entry.path)
        resolved = str(Path(output_folder).resolve())
        write_json_atomic(entry.path, {**stored, "output_folder": resolved})
        return dataclasses.replace(entry, output_folder=resolved)

    def remove(self, entry: IndexEntry) -> None:
        with contextlib.suppress(FileNotFoundError):
            os.remove(entry.path)


def _compare_fields(target: str, chain: str, recorded: list[str] | None) -> None:
    if recorded is None:
        return
    current = identity_fields(resolve_target(target))
    if current is None or current == recorded:
        return
    added = sorted(set(current) - set(recorded))
    removed = sorted(set(recorded) - set(current))
    raise SchemaChangedError(
        f"{target}: identity fields changed (added {added}, removed {removed}) but its version is "
        f"still {chain!r}. Every value enters the hash, so bump VERSION (and define "
        "upgrade_config to carry old runs forward)."
    )


def _count(directory: Path) -> int:
    try:
        names = os.listdir(directory)
    except FileNotFoundError:
        return 0
    return sum(1 for name in names if name.endswith(".json") and ".tmp." not in name)


def _entry(namespace: str, info: VersionInfo, status: str, stored: Mapping[str, Any],
           path: Path) -> IndexEntry:
    return IndexEntry(
        namespace=namespace,
        target=info.target,
        version_chain=info.version_chain,
        status=status,
        folder_id=str(stored["folder_id"]),
        output_folder=str(stored.get("output_folder", "")),
        nodes=tuple(stored.get("nodes", ())),
        path=path,
    )


def index_namespaces(root: str | os.PathLike[str]) -> list[str]:
    """Namespaces under a ledger root that have a version index."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(name for name in os.listdir(root) if (root / name / VERSIONS_DIRNAME).is_dir())
