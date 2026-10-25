"""Dataclass adapter (core, stdlib only): reads plain ``@dataclass`` instances.

The zero-framework path: a config can be a tree of dataclass instances with no OmegaConf or
pydantic installed. Its row equals that of ``dataclasses.asdict(config)`` and of
``OmegaConf.structured(config)``, except where a non-versioned dataclass leaf declares
exclusions (below).

* A :class:`~config_ledger.versioning.VersionedConfig` instance is a schema node. Its node
  target is always its class's import path; an instance whose ``_target_`` field was set to
  anything else (``Trainer(_target_="x")``) raises ``ValueError`` rather than being read as
  another class.
* Any other dataclass instance is read as a mapping of its fields: as the whole config, it is
  walked like a top-level dict (its versioned fields are identity roots, the rest execution
  control); as a field value, it is read like a nested plain dict: one JSON leaf, or a
  structured collection if its fields are all schema nodes.

Either way, fields the class declares excluded (``IDENTITY_EXCLUDE`` /
:func:`~config_ledger.versioning.execution_field`) are left out of the view. For a
non-versioned leaf this differs from ``asdict``, which keeps them; it matches how a
non-versioned pydantic model is read.

Always active (no framework to wait for in ``sys.modules``), and consulted after the framework
adapters, so a framework's own dataclass-based objects stay with that framework's adapter.

Status (2026-10-18): introduced, replacing the ``TypeError`` that pointed users
at ``dataclasses.asdict``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from config_ledger.versioning import TARGET_KEY, VersionedConfig, identity_excluded_fields


class DataclassAdapter:
    def handles(self, value: Any) -> bool:
        return dataclasses.is_dataclass(value) and not isinstance(value, type)

    def as_mapping(self, value: Any) -> Mapping[str, Any] | None:
        cls = type(value)
        excluded = identity_excluded_fields(cls)
        view = {
            field.name: getattr(value, field.name)
            for field in dataclasses.fields(value)
            if field.name not in excluded
        }
        if isinstance(value, VersionedConfig):
            # Not `cls._target_`: a `slots=True` dataclass replaces class defaults by slots.
            target = f"{cls.__module__}.{cls.__qualname__}"
            if view.get(TARGET_KEY, target) != target:
                raise ValueError(
                    f"{cls.__qualname__} instance has {TARGET_KEY}={view[TARGET_KEY]!r}; a "
                    f"versioned dataclass instance always stands for its own class ({target!r})"
                )
            view[TARGET_KEY] = target
        return view

    def to_plain(self, value: Any) -> Any:
        # Shallow: nested framework objects are converted again by the caller
        # (`plain_leaf` / `encode_leaf`).
        return dict(self.as_mapping(value))  # type: ignore[arg-type]


ADAPTER = DataclassAdapter()
