"""Read-only lookups: which output folder already owns a config?

:func:`lookup` wraps :meth:`~config_ledger.store.Ledger.find` with the diagnostics a bare miss
lacks since full-value hashing: after a version bump a config's hash is new, so a run that exists under
the old version (and was not upgraded) is a miss. :meth:`VersionIndex.diagnose
<config_ledger.versions.VersionIndex.diagnose>` turns that, an index newer than the code, and a
schema change into warnings. Nothing is claimed or written, and the stale-run check never
blocks a lookup.

:func:`load_config_file` reads an already-composed config: ``.json`` with the stdlib, ``.yaml``
/ ``.yml`` through OmegaConf (extra ``config-ledger[omegaconf]``), so it hashes exactly like the
``DictConfig`` a run sees. Composing Hydra configs and sweeps is :mod:`config_ledger.hydra.lookup`.

Status (2026-10-18): introduced.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config_ledger.store import Entry, Ledger

YAML_SUFFIXES = (".yaml", ".yml")


@dataclass(frozen=True)
class LookupResult:
    """The outcome of one lookup; ``entry`` is None on a miss."""

    namespace: str
    hash: str
    entry: Entry | None
    warnings: list[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.entry is not None

    def output_folder(self, parent_dir: str | os.PathLike[str] | None) -> Path | None:
        if self.entry is None or parent_dir is None:
            return None
        return Path(parent_dir) / self.entry.folder_id

    def to_json(self, parent_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
        folder = self.output_folder(parent_dir)
        return {
            "namespace": self.namespace,
            "hash": self.hash,
            "found": self.found,
            "folder_id": None if self.entry is None else self.entry.folder_id,
            "created": None if self.entry is None else self.entry.created,
            "output_folder": None if folder is None else str(folder),
            "warnings": list(self.warnings),
        }


def lookup(ledger: Ledger, namespace: str, config: Any) -> LookupResult:
    """Find the folder owning ``config`` in ``namespace`` without claiming anything."""
    identity = ledger.identity(config)
    entry = ledger.find(namespace, identity)
    warnings = ledger.versions(namespace).diagnose(identity.version_nodes(),
                                                   found=entry is not None)
    return LookupResult(namespace, identity.hash, entry, warnings)


def load_config_file(path: str | os.PathLike[str]) -> Any:
    """A composed config from ``.json`` (stdlib) or ``.yaml`` / ``.yml`` (OmegaConf)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text())
    if suffix in YAML_SUFFIXES:
        try:
            from omegaconf import OmegaConf
        except ImportError as exc:
            raise ImportError(
                f"reading {path.name} needs OmegaConf: pip install 'config-ledger[omegaconf]' "
                "(or pass the config as .json)") from exc
        return _hydra_dump_safe(OmegaConf.load(path))
    raise ValueError(f"unsupported config file {path.name!r}: expected .json, .yaml or .yml")


def _hydra_dump_safe(config: Any) -> Any:
    """A loaded yaml whose top-level interpolations that cannot resolve here are kept as their
    raw strings, so a Hydra dump (``.hydra/config.yaml``) can be looked up outside a job.

    Iterating a ``DictConfig``'s top level resolves its values, and ``${hydra:...}`` only
    resolves inside a job (the xfail in ``tests/test_adapters.py``). A loose top-level key is
    execution control, never identity, so its raw string is as good as its value. Nested
    nodes stay ``DictConfig`` children of the loaded root, so interpolations inside identity
    nodes still resolve exactly as in the run."""
    from omegaconf import DictConfig, OmegaConf

    if not isinstance(config, DictConfig):
        return config
    top: dict[str, Any] = {}
    for key in config:
        try:
            top[str(key)] = config[key]
        except Exception:
            if not OmegaConf.is_interpolation(config, str(key)):
                raise
            node = config._get_node(str(key))
            assert node is not None
            top[str(key)] = node._value()  # the raw "${...}" string
    return top
