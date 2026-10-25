"""Declared, inheritance-propagated config versions.

Every config class that participates in run identity inherits :class:`VersionedConfig`
and declares its own ``VERSION: ClassVar[int]``. The version is the deliberate,
coarse-grained lever for "the meaning of this class's fields changed, so results computed
under it are stale". It is *not* derived from source code: renaming a variable, adding a
comment, or refactoring must not invalidate results, and only the author knows when a
change is semantic. (AiiDA reached the same conclusion and removed the package version
from its cache hash in 2.6; exca and AiiDA both expose a single declared integer.)

What this module adds over a single integer is *propagation through inheritance*:
:func:`version_chain` records every versioned class in the MRO, base to leaf, e.g.
``"Trainer:1|MiningGoldTrainer:2"``. Bumping a base class therefore invalidates all of its
subclasses' results too, which a plain class attribute cannot do (a subclass that overrides
the attribute silently drops the base's bump).

The version is always resolved *from the live class* named by a node's ``_target_`` string
at hash time, never stored in yaml, so it cannot go stale when someone bumps a version
without regenerating config files.

Schema classes need not be dataclasses: pydantic models mix in :class:`VersionedConfig` the
same way, and their framework-specific handling is read through
:func:`config_ledger.adapters.schema_adapter_for`.

A version bump does not orphan old results silently. The ledger indexes every run by each
``(class, version chain)`` it uses (:mod:`config_ledger.versions`) and refuses new runs while an
old version still has "fresh" runs. A class may define :meth:`VersionedConfig.upgrade_config`
to carry old configs forward (:mod:`config_ledger.upgrade`).

Status (2026-09-10): ported from an internal project's version mixin and composite
version chain; identity-exclusion helpers are new. 2026-10-17:
schema-adapter dispatch for pydantic models. 2026-10-17: ``upgrade_config``,
:func:`parse_chain`, :func:`schema_fields`. 2026-10-18: Hydra's reserved
instantiate keys (:data:`INSTANTIATE_KEYS`) are never identity.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
from collections.abc import Iterable
from typing import Any, ClassVar

from config_ledger.adapters import schema_adapter_for

TARGET_KEY = "_target_"
"""The key under which a config node names its class (Hydra's instantiate convention)."""

VERSION_KEY = "config_version"
"""Per-node key under which the class version chain is recorded in materialized configs."""

INSTANTIATE_KEYS = frozenset({"_convert_", "_recursive_", "_partial_"})
"""Hydra's reserved instantiate keys that control *how* a node is built, not what the run is.
Excluded from identity in every node, whether written in yaml or declared as a field, by
:mod:`config_ledger.identity` and the schema guard (:func:`config_ledger.versions.identity_fields`). Hydra's
fourth reserved key, ``ARGS_KEY``, is rejected instead (see :mod:`config_ledger.identity`)."""

ARGS_KEY = "_args_"
"""Hydra's positional-arguments key. It carries real constructor values, so dropping it could
give two different runs one identity; schema classes take named fields only."""


METADATA_KEY = "config_ledger"
"""Namespace used inside ``dataclasses.field(metadata=...)`` for per-field declarations."""


class UpgradeImpossible(Exception):
    """Raised by :meth:`VersionedConfig.upgrade_config` when an old config cannot be carried
    forward to the current version. The message is reported by ``config-ledger upgrade``."""


class VersionedConfig:
    """Mixin for every config class whose fields bear on run identity.

    * Injects a ``_target_`` field holding the class's import path, so Hydra can
      instantiate it and so the flattener can resolve the class back from a config tree.
    * Requires each subclass to declare its own ``VERSION: ClassVar[int]`` and raises
      ``TypeError`` at class-creation time otherwise. Every class in the chain carries one
      so that a change to *any* level can invalidate results.
    * Optionally declares ``IDENTITY_EXCLUDE: ClassVar[Iterable[str]]``, field names that are
      execution control (devices, cache dirs, logging flags) and must not enter identity.
      Per-field ``dataclasses.field(metadata={"config_ledger": {"identity": False}})`` does
      the same thing; see :func:`execution_field`.

    Example::

        @dataclass
        class Trainer(VersionedConfig):
            VERSION: ClassVar[int] = 1
            learning_rate: float = 1e-3
            device: str = execution_field(default="cpu")

    A pydantic model mixes it in the same way (``class Trainer(VersionedConfig, BaseModel)``);
    there ``_target_`` is a plain class attribute rather than a field, and ``IDENTITY_EXCLUDE``
    must be annotated ``ClassVar`` (see :mod:`config_ledger.adapters.pydantic`).

    ``config_version`` is reserved: it is the key under which a node's version chain is
    recorded, so no class may declare a field of that name.
    """

    VERSION: ClassVar[int]
    IDENTITY_EXCLUDE: ClassVar[Iterable[str]] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "VERSION" not in cls.__dict__:
            raise TypeError(
                f"{cls.__qualname__} inherits VersionedConfig but does not declare its own "
                "`VERSION: ClassVar[int]`. Every config class must carry one so that changes to "
                "how its fields are interpreted can invalidate stale results. Add "
                "`VERSION: ClassVar[int] = 1` for a new class and increment it whenever the "
                "meaning of an existing class's fields changes."
            )
        if VERSION_KEY in cls.__dict__.get("__annotations__", {}):
            raise TypeError(
                f"{cls.__qualname__}: `{VERSION_KEY}` is reserved and cannot be a field name")
        schema = schema_adapter_for(cls)
        if schema is None or schema.prepare_class(cls):
            if "__annotations__" not in cls.__dict__:
                cls.__annotations__ = {}
            # Appended last, so a dataclass never sees a defaulted field before a required one.
            cls.__annotations__[TARGET_KEY] = str
        setattr(cls, TARGET_KEY, f"{cls.__module__}.{cls.__qualname__}")

    @classmethod
    def upgrade_config(cls, old: dict[str, Any]) -> dict[str, Any] | None:
        """Carry one stored config node forward to this class's current version.

        ``old`` is the node as stored when the run was registered: plain JSON values, its
        ``_target_``, and its stored version chain under ``"config_version"`` (read it with
        :func:`parse_chain`). Nested nodes have already been upgraded. Returns:

        * ``None`` if no upgrade is necessary (the stored chain equals :func:`version_chain`);
        * a new node (fields, optionally ``_target_``) valid under the current version; fields
          left out take their schema defaults, and ``config_version`` is filled in afterwards;
        * or raises :class:`UpgradeImpossible` with a reason if this old config cannot be
          carried forward.

        The default knows no upgrade paths. Override it per class; ``super().upgrade_config``
        handles a base class's part of the chain if the subclass wants to delegate.
        """
        stored = old.get(VERSION_KEY)
        if stored == version_chain(cls):
            return None
        raise UpgradeImpossible(f"{cls.__qualname__} defines no upgrade from {stored!r}")


def execution_field(*, default: Any = dataclasses.MISSING, default_factory: Any = dataclasses.MISSING,
                    **field_kwargs: Any) -> Any:
    """``dataclasses.field`` whose value is execution control and never enters identity.

    Use for anything that changes *how* a run executes but not *what* it computes: device,
    number of workers, cache directories, checkpoint cadence, logging verbosity.

    Also usable on a pydantic :class:`VersionedConfig` model, where it is converted to a pydantic
    field at class creation; only ``default`` / ``default_factory`` carry over there.
    """
    metadata = dict(field_kwargs.pop("metadata", None) or {})
    metadata[METADATA_KEY] = {**metadata.get(METADATA_KEY, {}), "identity": False}
    if default_factory is not dataclasses.MISSING:
        return dataclasses.field(default_factory=default_factory, metadata=metadata, **field_kwargs)
    return dataclasses.field(default=default, metadata=metadata, **field_kwargs)


def version_chain(cls: type) -> str:
    """Version string for one config node, e.g. ``"Trainer:1|MiningGoldTrainer:2"``.

    Every :class:`VersionedConfig` subclass in ``cls``'s MRO contributes, ordered base to
    leaf. Recording only the leaf's ``VERSION`` would fail exactly in the highest-blast-radius
    case, a shared base class changing meaning.
    """
    chain = [
        klass
        for klass in reversed(cls.__mro__)
        if isinstance(klass, type)
        and issubclass(klass, VersionedConfig)
        and klass is not VersionedConfig
        and "VERSION" in klass.__dict__
    ]
    return "|".join(f"{klass.__name__}:{klass.VERSION}" for klass in chain)


@functools.lru_cache(maxsize=None)
def resolve_target(target: str) -> type:
    """Import the class named by a ``_target_`` string such as ``"pkg.module.Trainer"``."""
    module_path, _, class_name = target.rpartition(".")
    if not module_path:
        raise ImportError(f"`_target_` {target!r} is not a dotted import path")
    module = importlib.import_module(module_path)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise AttributeError(f"module {module_path!r} has no attribute {class_name!r}") from exc


def parse_chain(chain: str) -> dict[str, int]:
    """``"Trainer:1|MiningGoldTrainer:2"`` -> ``{"Trainer": 1, "MiningGoldTrainer": 2}``."""
    parsed: dict[str, int] = {}
    for part in filter(None, chain.split("|")):
        name, _, version = part.rpartition(":")
        parsed[name] = int(version)
    return parsed


def is_versioned(cls: Any) -> bool:
    return isinstance(cls, type) and issubclass(cls, VersionedConfig)


def target_version(target: str) -> str:
    """:func:`version_chain` of the class named by ``target``."""
    return version_chain(resolve_target(target))


def identity_excluded_fields(cls: type) -> frozenset[str]:
    """Field names of ``cls`` declared as execution control, from both mechanisms (Hydra's
    :data:`INSTANTIATE_KEYS` are excluded on top of these by the identity code)."""
    excluded: set[str] = set()
    for klass in cls.__mro__:
        excluded.update(klass.__dict__.get("IDENTITY_EXCLUDE", ()))
    schema = schema_adapter_for(cls)
    if schema is not None:
        excluded.update(schema.excluded_fields(cls))
    elif dataclasses.is_dataclass(cls):
        for field in dataclasses.fields(cls):
            if not field.metadata.get(METADATA_KEY, {}).get("identity", True):
                excluded.add(field.name)
    return frozenset(excluded)


def schema_fields(cls: type) -> tuple[str, ...] | None:
    """Declared field names of ``cls`` (``_target_`` not included), or None if the class does
    not pin down a field set (a plain class taking ``**kwargs``, a pydantic model allowing
    extra fields, or no introspectable signature). A known field set is what identity fills
    defaults from, rejects unknown keys against, and what the schema guard records."""
    schema = schema_adapter_for(cls)
    if schema is not None:
        return schema.schema_fields(cls)
    if dataclasses.is_dataclass(cls):
        return tuple(f.name for f in dataclasses.fields(cls) if f.name != TARGET_KEY)
    import inspect

    try:
        signature = inspect.signature(cls)
    except (TypeError, ValueError):
        return None
    names = []
    for name, parameter in signature.parameters.items():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return None
        if parameter.kind is not inspect.Parameter.VAR_POSITIONAL:
            names.append(name)
    return tuple(names)


def schema_defaults(cls: type) -> dict[str, Any]:
    """Default value of every field of ``cls`` that has one.

    Dataclasses are read via ``dataclasses.fields`` (``default_factory`` is called), classes of
    a framework with a schema adapter (pydantic) via that adapter; any other class falls back to
    the defaults in its ``__init__`` signature. Fields without a default are absent from the
    result.
    """
    schema = schema_adapter_for(cls)
    if schema is not None:
        return schema.schema_defaults(cls)
    defaults: dict[str, Any] = {}
    if dataclasses.is_dataclass(cls):
        for field in dataclasses.fields(cls):
            if field.default is not dataclasses.MISSING:
                defaults[field.name] = field.default
            elif field.default_factory is not dataclasses.MISSING:
                defaults[field.name] = field.default_factory()
        return defaults
    import inspect

    try:
        signature = inspect.signature(cls)
    except (TypeError, ValueError):
        return defaults
    for name, parameter in signature.parameters.items():
        if parameter.default is not inspect.Parameter.empty:
            defaults[name] = parameter.default
    return defaults
