"""Pydantic adapter (extra ``config-ledger[pydantic]``): reads pydantic v2 models.

A pydantic schema class mixes in :class:`~config_ledger.versioning.VersionedConfig` like a
dataclass does (``class Trainer(VersionedConfig, BaseModel)``), so both share one version
chain. Two things differ from dataclasses:

* ``_target_`` is a plain class attribute, not a field (pydantic would turn a ``_``-prefixed
  annotation into a private attribute). A versioned model's mapping view therefore adds
  ``_target_`` from ``type(model)``; the row equals that of the same config written as a dict.
* Class-level ``ClassVar`` declarations must be annotated (``IDENTITY_EXCLUDE: ClassVar = ...``);
  pydantic rejects non-annotated class attributes.

A field is execution control (never enters identity) if any of these declares it:

* ``IDENTITY_EXCLUDE`` (read by :func:`~config_ledger.versioning.identity_excluded_fields`);
* :func:`~config_ledger.versioning.execution_field`, converted to a pydantic field carrying an
  :data:`EXECUTION` metadata marker at class creation;
* exca's ``_exclude_from_cls_uid``: a list/tuple/set, or a classmethod / method returning one;
  ``"."`` excludes every field (the node then contributes only ``_target_`` and its version).
  Class-level forms also apply to dict configs naming the class; an instance method can only
  be evaluated on a model instance. Pydantic's own ``Field(exclude=True)`` is deliberately
  *not* honoured: hiding a field from serialization is a different decision from identity.

A model without ``_target_`` (not a ``VersionedConfig``) is not a schema node: like a plain
nested dict, it is one JSON leaf, or a structured collection if its fields are all schema
nodes. Its own exclusions still apply.

Imported by :mod:`config_ledger.adapters` only once ``pydantic`` is already loaded.

Status (2026-10-17): introduced.
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field

from config_ledger.versioning import (
    METADATA_KEY,
    TARGET_KEY,
    VersionedConfig,
    identity_excluded_fields,
)

EXCA_EXCLUDE = "_exclude_from_cls_uid"
"""exca's per-class declaration of fields excluded from its uid."""

_EXCLUDE_ALL = "."


class _ExecutionMarker:
    def __repr__(self) -> str:
        return "config_ledger.EXECUTION"


EXECUTION = _ExecutionMarker()
"""Pydantic field metadata marking the field as execution control."""


def _is_execution_field(value: Any) -> bool:
    return isinstance(value, dataclasses.Field) and not value.metadata.get(METADATA_KEY, {}).get(
        "identity", True
    )


def _to_pydantic_field(field: dataclasses.Field) -> Any:
    kwargs: dict[str, Any] = {}
    if field.default is not dataclasses.MISSING:
        kwargs["default"] = field.default
    elif field.default_factory is not dataclasses.MISSING:
        kwargs["default_factory"] = field.default_factory
    info = Field(**kwargs)
    info.metadata.append(EXECUTION)
    return info


def _normalize_exca(declared: Any, cls: type[BaseModel]) -> set[str]:
    if declared is None:
        return set()
    where = f"{cls.__qualname__}.{EXCA_EXCLUDE}"
    if isinstance(declared, str):
        raise TypeError(f"{where} must be a list/tuple/set, not a string")
    if not isinstance(declared, (list, tuple, set, frozenset)):
        declared = declared()
    names = set(declared)
    unknown = names - set(cls.model_fields) - {_EXCLUDE_ALL}
    if unknown:
        raise ValueError(f"{where} names unknown fields {sorted(unknown)}")
    return names


def _class_exca(cls: type[BaseModel]) -> set[str]:
    """Class-level ``_exclude_from_cls_uid``; an instance method is skipped (needs an instance)."""
    private = cls.__private_attributes__.get(EXCA_EXCLUDE)
    if private is not None:
        return _normalize_exca(private.get_default(), cls)
    static = inspect.getattr_static(cls, EXCA_EXCLUDE, None)
    if isinstance(static, (classmethod, staticmethod)):
        return _normalize_exca(getattr(cls, EXCA_EXCLUDE), cls)
    if isinstance(static, (list, tuple, set, frozenset, str)):
        return _normalize_exca(static, cls)
    return set()


def _expand(names: set[str], cls: type[BaseModel]) -> set[str]:
    return set(cls.model_fields) if _EXCLUDE_ALL in names else names


class PydanticAdapter:
    # --- ConfigAdapter: model instances -----------------------------------------------

    def handles(self, value: Any) -> bool:
        return isinstance(value, BaseModel)

    def as_mapping(self, value: Any) -> Mapping[str, Any] | None:
        cls = type(value)
        excluded = identity_excluded_fields(cls) | _expand(
            _normalize_exca(getattr(value, EXCA_EXCLUDE, None), cls), cls
        )
        fields = {name: getattr(value, name) for name in cls.model_fields}
        fields.update(value.__pydantic_extra__ or {})
        view = {name: v for name, v in fields.items() if name not in excluded}
        if isinstance(value, VersionedConfig):
            return {TARGET_KEY: getattr(cls, TARGET_KEY), **view}
        return view

    def to_plain(self, value: Any) -> Any:
        return _plain(value)

    # --- SchemaAdapter: model classes ---------------------------------------------------

    def handles_class(self, cls: type) -> bool:
        return isinstance(cls, type) and issubclass(cls, BaseModel)

    def prepare_class(self, cls: type) -> bool:
        # Runs inside VersionedConfig.__init_subclass__, before pydantic collects fields.
        for name, value in list(vars(cls).items()):
            if _is_execution_field(value):
                setattr(cls, name, _to_pydantic_field(value))
        return False

    def schema_defaults(self, cls: type) -> dict[str, Any]:
        return {
            name: field.get_default(call_default_factory=True, validated_data={})
            for name, field in cls.model_fields.items()  # type: ignore[attr-defined]
            if not field.is_required()
        }

    def schema_fields(self, cls: type) -> tuple[str, ...] | None:
        if cls.model_config.get("extra") == "allow":  # type: ignore[attr-defined]
            return None
        return tuple(cls.model_fields)  # type: ignore[attr-defined]

    def excluded_fields(self, cls: type) -> frozenset[str]:
        marked = {name for name, f in cls.model_fields.items() if EXECUTION in f.metadata}  # type: ignore[attr-defined]
        return frozenset(marked | _expand(_class_exca(cls), cls))  # type: ignore[arg-type]


def _plain(value: Any) -> Any:
    """Models (at any depth) as dicts of their mapping view; other leaves left to encode_leaf."""
    if isinstance(value, BaseModel):
        return {k: _plain(v) for k, v in ADAPTER.as_mapping(value).items()}  # type: ignore[union-attr]
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


ADAPTER = SCHEMA = PydanticAdapter()
