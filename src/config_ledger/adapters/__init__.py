"""The seam between config frameworks and the plain-dict identity core.

:func:`config_ledger.identity.materialize` walks a config tree without knowing what produced
it. Everything framework-specific about *reading* that tree goes through a
:class:`ConfigAdapter`, so the materialized row, and everything downstream of it (hashing,
the store, the version index, upgrades), stays plain-dict and framework-agnostic.

An adapter answers three questions about one value of the tree:

* :meth:`~ConfigAdapter.handles`: is this value mine? Must be cheap and must not resolve
  interpolations or import the framework (see "Discovery" below).
* :meth:`~ConfigAdapter.as_mapping`: a ``Mapping[str, Any]`` view of a node's fields, or
  None if the value is a leaf of that framework (e.g. a list container). The view may be
  lazy: values are read only through ``.items()`` / ``__contains__`` of identity nodes, so
  execution-control blocks are never resolved. Values it yields may themselves be framework
  objects; they are dispatched again.
* :meth:`~ConfigAdapter.to_plain`: a list/dict *leaf* as builtin containers (interpolations
  resolved), ready for :func:`~config_ledger.identity.encode_leaf`.

A schema node is recognised by a ``_target_`` key in :meth:`~ConfigAdapter.as_mapping`; the
row format it must map onto is documented in :mod:`config_ledger.identity`.

Schema classes are a second, separate seam. Most frameworks' schemas are dataclasses, which
:mod:`config_ledger.versioning` reads itself. A framework whose schema classes are not (pydantic)
also exposes a :class:`SchemaAdapter`, which :mod:`config_ledger.versioning` consults to prepare
a :class:`~config_ledger.versioning.VersionedConfig` subclass and to read its fields, defaults
and excluded fields.

Discovery: the adapter list is internal and fixed (no public ``register_adapter()``; an
in-tree entry is enough until a third-party adapter needs one). A framework adapter
is only consulted once its framework is already in ``sys.modules``: an object or class of that
framework cannot exist otherwise, so the core never imports an optional dependency and needs
no registration step.

Status (2026-10-17): introduced; the OmegaConf handling previously inlined in
``identity._to_plain`` / ``_as_mapping`` moved to :mod:`config_ledger.adapters.omegaconf`.
Schema adapters and :mod:`config_ledger.adapters.pydantic` added.
2026-10-18: :mod:`config_ledger.adapters.dataclass`, the always-active,
stdlib-only adapter for plain dataclass instances, consulted after the framework adapters.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping
from typing import Any, Protocol


class ConfigAdapter(Protocol):
    """Reads one config framework's objects for the materializer."""

    def handles(self, value: Any) -> bool: ...

    def as_mapping(self, value: Any) -> Mapping[str, Any] | None: ...

    def to_plain(self, value: Any) -> Any: ...


class SchemaAdapter(Protocol):
    """Reads one framework's schema classes for :mod:`config_ledger.versioning`."""

    def handles_class(self, cls: type) -> bool: ...

    def prepare_class(self, cls: type) -> bool:
        """Adapt a new ``VersionedConfig`` subclass, mid-creation. Returns whether ``_target_``
        should be added as a field annotation (False if the framework would mangle it)."""
        ...

    def schema_defaults(self, cls: type) -> dict[str, Any]: ...

    def schema_fields(self, cls: type) -> tuple[str, ...] | None:
        """Declared field names, or None if the class accepts undeclared fields."""
        ...

    def excluded_fields(self, cls: type) -> frozenset[str]:
        """Fields excluded by this framework's own declarations (``IDENTITY_EXCLUDE`` is read
        by the caller)."""
        ...


class MappingAdapter:
    """Builtin mappings: already plain, read as they are."""

    def handles(self, value: Any) -> bool:
        return isinstance(value, Mapping)

    def as_mapping(self, value: Any) -> Mapping[str, Any] | None:
        return value

    def to_plain(self, value: Any) -> Any:
        return value


_MAPPING_ADAPTER = MappingAdapter()

# (framework top-level module, adapter module exposing ADAPTER and optionally SCHEMA),
# consulted in order before the builtin dataclass and mapping adapters, since framework
# containers are often Mappings (and sometimes dataclasses) too.
_FRAMEWORK_ADAPTERS: tuple[tuple[str, str], ...] = (
    ("omegaconf", "config_ledger.adapters.omegaconf"),
    ("pydantic", "config_ledger.adapters.pydantic"),
)


def _loaded_adapter_modules() -> list[Any]:
    return [
        importlib.import_module(adapter_module)
        for framework, adapter_module in _FRAMEWORK_ADAPTERS
        if framework in sys.modules
    ]


def _active_adapters() -> list[ConfigAdapter]:
    adapters: list[ConfigAdapter] = [module.ADAPTER for module in _loaded_adapter_modules()]
    # Stdlib-only, so always active; imported lazily because it reads `versioning`, which
    # imports this module.
    adapters.append(importlib.import_module("config_ledger.adapters.dataclass").ADAPTER)
    adapters.append(_MAPPING_ADAPTER)
    return adapters


def schema_adapter_for(cls: type) -> SchemaAdapter | None:
    """The schema adapter for ``cls``, or None for dataclasses and plain classes."""
    for module in _loaded_adapter_modules():
        schema = getattr(module, "SCHEMA", None)
        if schema is not None and schema.handles_class(cls):
            return schema
    return None


def adapter_for(value: Any) -> ConfigAdapter | None:
    """The first adapter that handles ``value``, or None for a builtin leaf."""
    for adapter in _active_adapters():
        if adapter.handles(value):
            return adapter
    return None


def as_mapping(value: Any) -> Mapping[str, Any] | None:
    """A mapping view of ``value`` if some adapter reads it as a node; None for leaves."""
    adapter = adapter_for(value)
    return None if adapter is None else adapter.as_mapping(value)


def to_plain(value: Any) -> Any:
    """``value`` as builtins if an adapter handles it; unchanged otherwise."""
    adapter = adapter_for(value)
    return value if adapter is None else adapter.to_plain(value)
