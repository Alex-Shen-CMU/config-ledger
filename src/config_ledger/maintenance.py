"""Operations on the version index behind the ``config-ledger`` CLI.

Each operation selects index entries with :class:`Selection` (namespaces, ``_target_``s,
versions) and returns one :class:`Outcome` per affected run, so the CLI only prints. All of
them are dry runs unless ``apply=True``.

* :func:`report`: ``(class, version) -> fresh/stale/migrated`` counts, live version marked.
* :func:`mark_stale`: ``fresh`` -> ``stale`` for configs under *old* versions. Refuses the
  live version: a current run that should be redone is rerun with ``overwrite``.
* :func:`upgrade`: for runs with a ``fresh`` config under an old version, upgrade the whole
  stored tree (:func:`~config_ledger.upgrade.upgrade_tree`), register the new hash against the
  same folder, move every old-version entry of the run to ``migrated`` and record the run
  under the live versions. Runs with any ``stale`` config are skipped unless
  ``include_stale``. A new hash already owned by a *different* folder (the config was rerun
  under the new version first) is a ``CONFLICT`` and left alone.
* :func:`purge_stale`: delete runs with a ``stale`` config: output folder, sidecars of every
  hash in its run manifest, and all of its index entries. Sweep CSVs are history and stay.
* :func:`rebuild`: regenerate sidecars and index entries from the ``run_manifest.json`` files
  under output roots, reconciling with what exists (replaces ``index backfill``,
  which rebuilt the index only).

Status (2026-10-17): introduced. 2026-10-18: ``rebuild``.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from config_ledger._files import read_json
from config_ledger.identity import Identity, compute_identity, hash_row, version_nodes
from config_ledger.store import RUN_MANIFEST_FILENAME, Ledger
from config_ledger.upgrade import upgrade_tree
from config_ledger.versions import (
    FRESH,
    MIGRATED,
    STALE,
    IndexEntry,
    SchemaChangedError,
    VersionIndex,
    compare_chains,
    index_namespaces,
    live_chain,
)

# Outcome statuses.
WOULD_MARK = "WOULD-MARK"
MARKED = "MARKED"
REFUSED = "REFUSED"
WOULD_UPGRADE = "WOULD-UPGRADE"
UPGRADED = "UPGRADED"
FAILED = "FAILED"
CONFLICT = "CONFLICT"
SKIPPED = "SKIPPED"
WOULD_PURGE = "WOULD-PURGE"
PURGED = "PURGED"
WOULD_CREATE = "WOULD-CREATE"
CREATED = "CREATED"
WOULD_ADD = "WOULD-ADD"
ADDED = "ADDED"
WOULD_MOVE = "WOULD-MOVE"
MOVED = "MOVED"
CURRENT = "CURRENT"
ORPHAN = "ORPHAN"


@dataclass(frozen=True)
class Selection:
    """Which index entries an operation looks at. Empty means all.

    A target matches its full ``_target_`` or its class name (``"Trainer"`` matches
    ``"pkg.schema.Trainer"``). A version matches the full chain (``"Trainer:1|Gold:2"``) or,
    as an integer, the leaf class's version."""

    namespaces: Sequence[str] = ()
    targets: Sequence[str] = ()
    versions: Sequence[str] = ()

    def namespaces_in(self, ledger: Ledger) -> list[str]:
        return list(self.namespaces) or index_namespaces(ledger.root)

    def matches(self, target: str, chain: str) -> bool:
        if self.targets and not any(t in (target, target.rsplit(".", 1)[-1]) for t in self.targets):
            return False
        if self.versions:
            leaf = chain.rsplit(":", 1)[-1]
            return any(v == chain or v == leaf for v in self.versions)
        return True


@dataclass
class Outcome:
    """What an operation did (or would do) to one run or one index entry."""

    status: str
    namespace: str
    folder_id: str
    output_folder: str
    detail: str = ""
    causes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class VersionRow:
    namespace: str
    target: str
    version_chain: str
    state: str  # "live", "older", "newer" or "unresolvable"
    fresh: int
    stale: int
    migrated: int


def _is_old(target: str, chain: str) -> bool:
    live = live_chain(target)
    return live is None or chain != live


def _describe(entry: IndexEntry) -> str:
    return f"{entry.target} @ {entry.version_chain} [{entry.status}] at {', '.join(entry.nodes)}"


ALL = Selection()


def report(ledger: Ledger, selection: Selection = ALL) -> list[VersionRow]:
    rows = []
    for namespace in selection.namespaces_in(ledger):
        for info in ledger.versions(namespace).versions():
            if not selection.matches(info.target, info.version_chain):
                continue
            live = live_chain(info.target)
            state = "unresolvable" if live is None else compare_chains(info.version_chain, live)
            rows.append(VersionRow(namespace, info.target, info.version_chain, state,
                                   info.counts[FRESH], info.counts[STALE], info.counts[MIGRATED]))
    return rows


def mark_stale(ledger: Ledger, selection: Selection = ALL, apply: bool = False) -> list[Outcome]:
    """Mark ``fresh`` entries under old versions ``stale``. Live-version entries the selection
    names explicitly (by version) are ``REFUSED``; without a version filter they are not touched."""
    outcomes = []
    for namespace in selection.namespaces_in(ledger):
        index = ledger.versions(namespace)
        for entry in index.entries(status=FRESH):
            if not selection.matches(entry.target, entry.version_chain):
                continue
            if not _is_old(entry.target, entry.version_chain):
                if selection.versions:
                    outcomes.append(Outcome(
                        REFUSED, namespace, entry.folder_id, entry.output_folder,
                        f"{_describe(entry)} is the live version; rerun it with overwrite=true "
                        "instead"))
                continue
            if apply:
                index.move(entry, STALE)
            outcomes.append(Outcome(MARKED if apply else WOULD_MARK, namespace, entry.folder_id,
                                    entry.output_folder, _describe(entry)))
    return outcomes


def _by_folder(index: VersionIndex) -> dict[str, list[IndexEntry]]:
    grouped: dict[str, list[IndexEntry]] = {}
    for entry in index.entries():
        grouped.setdefault(entry.folder_id, []).append(entry)
    return grouped


def upgrade(ledger: Ledger, selection: Selection = ALL, apply: bool = False,
            include_stale: bool = False) -> list[Outcome]:
    outcomes = []
    for namespace in selection.namespaces_in(ledger):
        index = ledger.versions(namespace)
        for folder_id, entries in sorted(_by_folder(index).items()):
            triggers = [e for e in entries if e.status == FRESH
                        and _is_old(e.target, e.version_chain)
                        and selection.matches(e.target, e.version_chain)]
            if triggers:
                outcomes.append(_upgrade_run(ledger, index, folder_id, entries, triggers, apply,
                                             include_stale))
    return outcomes


def _upgrade_run(ledger: Ledger, index: VersionIndex, folder_id: str, entries: list[IndexEntry],
                 triggers: list[IndexEntry], apply: bool, include_stale: bool) -> Outcome:
    namespace = index.namespace
    output_folder = triggers[0].output_folder
    causes = [_describe(e) for e in triggers]

    def outcome(status: str, detail: str = "") -> Outcome:
        return Outcome(status, namespace, folder_id, output_folder, detail, causes)

    stale = [e for e in entries if e.status == STALE]
    if stale and not include_stale:
        return outcome(SKIPPED, "has stale config(s): " + "; ".join(map(_describe, stale))
                       + " (pass --include-stale to upgrade them too)")
    manifest = read_json(Path(output_folder) / RUN_MANIFEST_FILENAME)
    if manifest is None or "tree" not in manifest:
        return outcome(FAILED, f"no run manifest with a stored config tree in {output_folder}")
    errors: list[str] = []
    tree = upgrade_tree(manifest["tree"], errors)
    if tree is None:
        return outcome(FAILED, "\n".join(errors))
    identity = compute_identity(tree)
    owner = ledger.find(namespace, identity)
    if owner is not None and owner.folder_id != folder_id:
        return outcome(CONFLICT, f"upgraded hash {identity.hash} already belongs to folder "
                                 f"{owner.folder_id}")
    if not apply:
        return outcome(WOULD_UPGRADE, identity.hash)
    entry = ledger.register(namespace, identity, folder_id)
    ledger.write_run_manifest(output_folder, entry, identity)
    for old in entries:
        if old.status != MIGRATED and _is_old(old.target, old.version_chain):
            index.move(old, MIGRATED)
    index.record(identity.version_nodes(), folder_id, output_folder)
    return outcome(UPGRADED, identity.hash)


def purge_stale(ledger: Ledger, selection: Selection = ALL, apply: bool = False) -> list[Outcome]:
    outcomes = []
    for namespace in selection.namespaces_in(ledger):
        index = ledger.versions(namespace)
        for folder_id, entries in sorted(_by_folder(index).items()):
            causes = [_describe(e) for e in entries
                      if e.status == STALE and selection.matches(e.target, e.version_chain)]
            if not causes:
                continue
            output_folder = next(e.output_folder for e in entries if e.output_folder)
            detail = _purge(ledger, index, folder_id, entries, output_folder) if apply else ""
            status = PURGED if apply else WOULD_PURGE
            outcomes.append(Outcome(status, namespace, folder_id, output_folder, detail, causes))
    return outcomes


def _purge(ledger: Ledger, index: VersionIndex, folder_id: str, entries: list[IndexEntry],
           output_folder: str) -> str:
    """Sidecars first (nothing resolves to the folder any more), then the folder, then the index
    entries last, so an interrupted purge is found and finished by the next one."""
    manifest = read_json(Path(output_folder) / RUN_MANIFEST_FILENAME)
    if manifest is not None:
        hashes = manifest.get("config_hashes", [])
        sidecars = [ledger.sidecar_path(index.namespace, h) for h in hashes]
    else:  # folder already gone: find this run's sidecars by scanning the namespace
        sidecars = [p for p in ledger.namespace_dir(index.namespace).glob("??/*.json")
                    if (read_json(p) or {}).get("folder_id") == folder_id]
    removed = 0
    for path in sidecars:
        stored = read_json(path)
        if stored is not None and stored.get("folder_id") == folder_id:
            os.remove(path)
            removed += 1
    shutil.rmtree(output_folder, ignore_errors=True)
    for entry in entries:
        index.remove(entry)
    return f"removed {removed} sidecar(s), the output folder and {len(entries)} index entr(ies)"


@dataclass(frozen=True)
class _RunManifest:
    """A validated ``run_manifest.json``: every hash with the row and tree it was registered
    with, oldest first."""

    namespace: str
    folder_id: str
    folder: str
    hashes: list[str]
    configs: dict[str, dict]


def _read_run_manifest(path: Path) -> _RunManifest | str:
    """The manifest at ``path``, or why it can't be rebuilt from. Rows are re-hashed as stored,
    never re-materialized: the live classes would re-resolve old version chains."""
    try:
        manifest = read_json(path) or {}
    except ValueError as exc:
        return f"unreadable run_manifest.json: {exc}"
    namespace, folder_id = manifest.get("namespace"), manifest.get("folder_id")
    hashes, configs = manifest.get("config_hashes") or [], manifest.get("configs") or {}
    if not namespace or not folder_id or not hashes:
        return "run_manifest.json has no namespace/folder_id/config_hashes"
    if path.parent.name != folder_id:
        return (f"folder is named {path.parent.name!r} but its manifest says folder_id "
                f"{folder_id!r}; lookups resolve <parent>/<folder_id>, so rename it back")
    problems = []
    for hash_key in hashes:
        row = (configs.get(hash_key) or {}).get("config")
        if row is None:
            problems.append(f"no stored row for hash {hash_key}")
        elif hash_row(row) != hash_key:
            problems.append(f"stored row of hash {hash_key} hashes to {hash_row(row)}")
    if problems:
        return "\n".join(problems)
    return _RunManifest(str(namespace), str(folder_id), str(path.parent.resolve()),
                        list(hashes), dict(configs))


def rebuild(ledger: Ledger, roots: Iterable[str | os.PathLike[str]],
            namespaces: Sequence[str] = (), apply: bool = False) -> list[Outcome]:
    """Regenerate sidecars and version index entries from the run manifests under ``roots``.

    Reconciles with what is there and never overwrites: a sidecar already owned by another
    folder, or a hash claimed by two manifests, is a ``CONFLICT`` (and that run is not indexed).
    Index entries are recorded ``fresh`` under the versions of the latest hash and ``migrated``
    under versions only older hashes used (the run was upgraded); ``stale`` cannot be recovered.
    Entries whose ``output_folder`` is not where the manifest was found are relocated. Sidecars
    and index entries of folders not found under ``roots`` are reported as ``ORPHAN`` only.
    """
    outcomes: list[Outcome] = []
    runs: list[_RunManifest] = []
    seen: set[tuple[str, str]] = set()
    for root in roots:
        for path in _run_manifests(root):
            run = _read_run_manifest(path)
            stored = read_json(path) if isinstance(run, str) else None
            namespace = run.namespace if isinstance(run, _RunManifest) else str(
                (stored or {}).get("namespace") or "")
            if namespaces and namespace not in namespaces:
                continue
            folder = str(path.parent.resolve())
            if isinstance(run, str):
                outcomes.append(Outcome(FAILED, namespace, path.parent.name, folder, run))
                seen.add((namespace, str((stored or {}).get("folder_id") or path.parent.name)))
                continue
            seen.add((namespace, run.folder_id))
            runs.append(run)

    places: dict[tuple[str, str], list[str]] = {}
    for run in runs:
        places.setdefault((run.namespace, run.folder_id), []).append(run.folder)
    owners: dict[tuple[str, str], set[str]] = {}
    for run in runs:
        for hash_key in run.hashes:
            owners.setdefault((run.namespace, hash_key), set()).add(run.folder_id)

    for run in runs:
        if len(places[(run.namespace, run.folder_id)]) > 1:
            outcomes.append(Outcome(FAILED, run.namespace, run.folder_id, run.folder,
                                    "the same folder_id was found at: "
                                    + ", ".join(places[(run.namespace, run.folder_id)])))
            continue
        outcomes.extend(_rebuild_run(ledger, run, owners, apply))
    outcomes.extend(_orphans(ledger, namespaces, seen))
    return outcomes


def _rebuild_run(ledger: Ledger, run: _RunManifest, owners: dict[tuple[str, str], set[str]],
                 apply: bool) -> list[Outcome]:
    def outcome(status: str, detail: str) -> Outcome:
        return Outcome(status, run.namespace, run.folder_id, run.folder, detail)

    outcomes = []
    for hash_key in run.hashes:
        others = sorted(owners[(run.namespace, hash_key)] - {run.folder_id})
        if others:
            outcomes.append(outcome(CONFLICT, f"sidecar {hash_key}: also claimed by the run "
                                              f"manifest of folder(s) {', '.join(others)}"))
            continue
        stored = read_json(ledger.sidecar_path(run.namespace, hash_key))
        if stored is None and apply:
            config = run.configs[hash_key]
            identity = Identity(tree=config.get("tree") or {}, materialized=config["config"])
            winner = ledger.register(run.namespace, identity, run.folder_id)
            stored = {"folder_id": winner.folder_id}
            if stored["folder_id"] == run.folder_id:
                outcomes.append(outcome(CREATED, f"sidecar {hash_key}"))
                continue
        if stored is None:
            outcomes.append(outcome(WOULD_CREATE, f"sidecar {hash_key}"))
        elif stored.get("folder_id") == run.folder_id:
            outcomes.append(outcome(CURRENT, f"sidecar {hash_key}"))
        else:
            outcomes.append(outcome(CONFLICT, f"sidecar {hash_key} belongs to folder "
                                              f"{stored.get('folder_id')}"))
    if any(o.status == CONFLICT for o in outcomes):
        return [*outcomes, outcome(SKIPPED, "version index: not indexed until the conflict is "
                                            "resolved")]
    try:
        return [*outcomes, *_rebuild_index(ledger.versions(run.namespace), run, apply)]
    except SchemaChangedError as exc:
        return [*outcomes, outcome(FAILED, f"version index: {exc}")]


def _rebuild_index(index: VersionIndex, run: _RunManifest, apply: bool) -> list[Outcome]:
    """Missing entries: ``fresh`` under the latest hash's versions, ``migrated`` under versions
    only older hashes used. Existing entries (any status) are kept, relocated if moved."""
    def outcome(status: str, detail: str) -> Outcome:
        return Outcome(status, run.namespace, run.folder_id, run.folder, detail)

    live = version_nodes(run.configs[run.hashes[-1]]["config"])
    migrated: dict[tuple[str, str], list[str]] = {}
    for hash_key in run.hashes[:-1]:
        for key, prefixes in version_nodes(run.configs[hash_key]["config"]).items():
            if key not in live:
                migrated.setdefault(key, prefixes)
    existing = {(e.target, e.version_chain): e for e in index.folder_entries(run.folder_id)}
    outcomes = []
    for status, nodes in ((FRESH, live), (MIGRATED, migrated)):
        missing = {key: prefixes for key, prefixes in nodes.items() if key not in existing}
        if not missing:
            continue
        if apply:
            index.record(missing, run.folder_id, run.folder, status)
        outcomes.append(outcome(ADDED if apply else WOULD_ADD, "version index: " + ", ".join(
            f"{t} @ {c} [{status}]" for t, c in sorted(missing))))
    moved = [e for e in existing.values() if e.output_folder != run.folder]
    if moved:
        if apply:
            for entry in moved:
                index.relocate(entry, run.folder)
        sources = sorted({e.output_folder for e in moved})
        outcomes.append(outcome(MOVED if apply else WOULD_MOVE,
                                f"version index: {len(moved)} entr(ies) pointed at "
                                f"{', '.join(sources)}"))
    if not outcomes:
        outcomes.append(outcome(CURRENT, "version index"))
    return outcomes


def _orphans(ledger: Ledger, namespaces: Sequence[str],
             seen: set[tuple[str, str]]) -> list[Outcome]:
    """Sidecars and index entries whose folder has no run manifest under the scanned roots.
    Reported only: the roots may cover only part of the outputs."""
    if namespaces:
        candidates = list(namespaces)
    elif ledger.root.is_dir():
        candidates = sorted(p.name for p in ledger.root.iterdir() if p.is_dir())
    else:
        candidates = []
    outcomes = []
    for namespace in candidates:
        sidecars: dict[str, int] = {}
        for path in ledger.namespace_dir(namespace).glob("??/*.json"):
            stored = read_json(path) or {}
            if stored.get("hash") == path.stem and stored.get("folder_id"):
                folder_id = str(stored["folder_id"])
                sidecars[folder_id] = sidecars.get(folder_id, 0) + 1
        entries: dict[str, list[IndexEntry]] = {}
        for entry in ledger.versions(namespace).entries():
            entries.setdefault(entry.folder_id, []).append(entry)
        for folder_id in sorted(sidecars.keys() | entries.keys()):
            if (namespace, folder_id) in seen:
                continue
            where = next((e.output_folder for e in entries.get(folder_id, ())), "")
            outcomes.append(Outcome(ORPHAN, namespace, folder_id, where,
                                    f"{sidecars.get(folder_id, 0)} sidecar(s), "
                                    f"{len(entries.get(folder_id, ()))} index entr(ies); no run "
                                    "manifest for this folder under the scanned roots"))
    return outcomes


def _run_manifests(root: str | os.PathLike[str]) -> Iterable[Path]:
    for dirpath, _dirnames, filenames in os.walk(root):
        if RUN_MANIFEST_FILENAME in filenames:
            yield Path(dirpath) / RUN_MANIFEST_FILENAME


def summarize(outcomes: Iterable[Outcome]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in outcomes:
        counts[item.status] = counts.get(item.status, 0) + 1
    return counts
