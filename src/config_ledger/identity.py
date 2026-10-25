"""From a config tree to a run identity: materialize, flatten, hash.

Three rules decide *what* a run's identity is; all are declared, never inferred from code:

1. **Identity lives in versioned nodes.** A config tree's top-level entries are walked and
   only those that are mappings carrying a ``_target_`` (i.e. instances of a
   :class:`~config_ledger.versioning.VersionedConfig` schema) contribute. Loose top-level
   scalars (``overwrite``, ``output_dir``, ``retry_failed``, the ``hydra`` block, ...) are
   execution control and are invisible to identity. Inside a node, every field counts unless
   the class declared it excluded (``IDENTITY_EXCLUDE`` /
   :func:`~config_ledger.versioning.execution_field`), and Hydra's reserved instantiate keys
   (``_convert_``, ``_recursive_``, ``_partial_``) never count; ``_args_`` raises.
   An identity field's value must be one of:

   * a leaf: a primitive (see "Leaves" below), or a list/tuple/dict of primitives nested to
     any depth, encoded as one JSON string;
   * a schema node (a nested ``_target_`` mapping), recursed into;
   * a *structured collection*: a list/tuple, or a ``_target_``-less mapping, whose entries
     are all schema nodes or None, at least one a node. Each node entry is recursed into at
     ``<field>.<index or key>``; a None entry is the leaf ``""``.

   Any other value holding a schema node (a list mixing nodes and other values, a list of
   lists of nodes, ...) raises ``ValueError``: the node's version would never reach identity.
   A non-versioned dataclass or pydantic model is read as a dict, so the same rules apply.

2. **Every identity-bearing value enters the hash**, defaults included. A field
   the config leaves out is filled from its schema default, so spelling a default out is the
   same identity as omitting it; a missing required field and a key the class does not
   declare both raise. (Classes without a fixed field set, see
   :func:`~config_ledger.versioning.schema_fields`, are taken as written.) Changing a default
   therefore changes the identity of configs that rely on it, which is correct: they now
   compute something else.

3. **Each node records its class's version chain**, resolved from the live class.

Materialization produces two views of the same values:

* the **tree**: nested, plain JSON (``{"method": {"_target_": ..., "config_version": ...,
  "trainer": {...}, "steps": [{...}, None], "name": "x"}}``), stored with every run so that
  :mod:`config_ledger.upgrade` can hand old configs to ``upgrade_config`` as real values;
* the **row**: flat ``{"<RootClass>.<path>.<field>": "<encoded value>"}``, derived from the
  tree, which is what is hashed. The top-level key a node sits under is deliberately *not*
  part of its identity, its class name is.

This module only walks plain mappings itself; reading config objects (a dataclass instance,
an OmegaConf ``DictConfig``, a pydantic model) is delegated to :mod:`config_ledger.adapters`,
so nothing here imports a config framework. A versioned instance passed as the whole config
is itself the one root node.

Leaves are encoded strictly: a value that is not a builtin scalar, ``Enum``, ``PurePath`` or a
list/tuple/dict of those (after its adapter converts it) raises ``TypeError`` rather than
hashing its ``str()``, which would silently give a different identity per repr.

Status (2026-09-10): flattening/hashing ported from an internal project.
2026-10-17: strict leaf encoding; versioned models as the root config.
2026-10-17: full-value hashing only (identity policies and drift verification
removed), default filling, unknown-key rejection, the stored tree.
2026-10-18: dataclass instances are read (through their adapter) instead of rejected.
2026-10-18: Hydra's reserved instantiate keys are skipped; ``_args_`` is rejected.
2026-10-18: structured collections are recursed into; schema nodes anywhere else
in a leaf are rejected.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePath
from typing import Any

import xxhash

from config_ledger.adapters import as_mapping, to_plain
from config_ledger.versioning import (
    ARGS_KEY,
    INSTANTIATE_KEYS,
    TARGET_KEY,
    VERSION_KEY,
    VersionedConfig,
    identity_excluded_fields,
    resolve_target,
    schema_defaults,
    schema_fields,
    target_version,
)

Row = dict[str, str]
Tree = dict[str, Any]


# --- leaf encoding ----------------------------------------------------------------------


_SCALARS = (bool, int, float, str)


def encode_leaf(value: Any) -> str:
    """Stable string form of one leaf value, as it enters the row and the hash.

    Raises ``TypeError`` for a value of any type not listed in the module docstring."""
    value = to_plain(value)
    if value is None:
        return ""
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, _SCALARS):
        return str(value)
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=isinstance(value, dict), default=_json_default)
    raise TypeError(_unencodable(value))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.name
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, PurePath):
        return str(obj)
    plain = to_plain(obj)  # a framework object nested in a builtin container
    if plain is not obj:
        return plain
    raise TypeError(_unencodable(obj))


def plain_leaf(value: Any) -> Any:
    """``value`` as plain JSON (what the tree stores), such that ``encode_leaf`` gives the same
    string for both. Enums become their name, paths ``str``, tuples lists, dict keys ``str``."""
    value = to_plain(value)
    if isinstance(value, Enum):
        return value.name
    if value is None or isinstance(value, _SCALARS):
        return value
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [plain_leaf(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): plain_leaf(v) for k, v in value.items()}
    raise TypeError(_unencodable(value))


def _unencodable(value: Any) -> str:
    return (
        f"cannot encode a {type(value).__name__} as a config leaf; leaves must be None, bool, int, "
        "float, str, Enum, PurePath, or a list/tuple/dict of those"
    )


def is_structured(value: Any) -> bool:
    """A mapping carrying a ``_target_`` is a fixed-shape schema node to recurse into.

    Checked without resolving interpolations, so execution-control blocks that reference
    runtime-only resolvers (``${hydra:runtime.cwd}``) are never touched by identity code.
    """
    return _schema_node(value) is not None


def _structured_collection(
    value: Any,
) -> tuple[bool, list[tuple[str, Mapping[str, Any] | None, bool]]] | None:
    """``value`` as a structured collection, if it is a list/tuple, or a
    ``_target_``-less mapping, whose entries are all schema nodes or None, at least one a node:
    ``(is_list, [(key, node view or None, entry is a schema-class instance), ...])``. Else None."""
    mapping = as_mapping(value)
    if mapping is not None:
        if TARGET_KEY in mapping:
            return None
        is_list, items = False, [(str(k), v) for k, v in mapping.items()]
    else:
        plain = to_plain(value)  # a framework list (ListConfig) as a builtin one
        if not isinstance(plain, (list, tuple)):
            return None
        is_list, items = True, [(str(i), v) for i, v in enumerate(plain)]
    entries = [(key, None if entry is None else _schema_node(entry),
                isinstance(entry, VersionedConfig)) for key, entry in items]
    nodes = sum(view is not None for _, view, _ in entries)
    if not nodes or nodes + sum(entry is None for _, entry in items) < len(items):
        return None
    return is_list, entries


def node_collection(value: Any) -> list[tuple[str, Any]] | None:
    """The ``(key, node or None)`` entries of a structured collection in a materialized tree
    (a list, or a ``_target_``-less dict, of nodes and None, at least one a node), else None."""
    if isinstance(value, list):
        entries = [(str(i), v) for i, v in enumerate(value)]
    elif isinstance(value, dict) and TARGET_KEY not in value:
        entries = [(str(k), v) for k, v in value.items()]
    else:
        return None
    if any(is_node(v) for _, v in entries) and all(v is None or is_node(v) for _, v in entries):
        return entries
    return None


def _holds_node(leaf: Any) -> bool:
    """Whether a plain-JSON leaf contains a ``_target_`` mapping anywhere."""
    if isinstance(leaf, list):
        return any(_holds_node(v) for v in leaf)
    if isinstance(leaf, dict):
        return TARGET_KEY in leaf or any(_holds_node(v) for v in leaf.values())
    return False


def _schema_node(value: Any) -> Mapping[str, Any] | None:
    """``value``'s mapping view (left lazy, via its adapter) if it is a schema node, else None."""
    node = as_mapping(value)
    return node if node is not None and TARGET_KEY in node else None


# --- materialization ----------------------------------------------------------------------


def _materialize_node(node: Mapping[str, Any], where: str, instance: bool = False) -> Tree:
    """``instance``: ``node`` is the view of a schema-class instance (a dataclass or pydantic
    model), which is complete by construction; a field missing from its view was excluded per
    instance (exca's instance-method ``_exclude_from_cls_uid``), so no defaults are filled in."""
    target = str(node[TARGET_KEY])
    cls = resolve_target(target)
    if ARGS_KEY in node:
        raise ValueError(
            f"{where}: {ARGS_KEY} (positional constructor args) is not supported in a schema "
            f"node; set {cls.__qualname__}'s fields by name")
    excluded = identity_excluded_fields(cls) | INSTANTIATE_KEYS
    values: dict[str, Any] = {
        name: value for name, value in node.items()
        if name not in (TARGET_KEY, VERSION_KEY) and name not in excluded
    }
    declared = schema_fields(cls)
    if declared is not None:
        unknown = sorted(set(values) - set(declared))
        if unknown:
            raise ValueError(f"{where}: {cls.__qualname__} declares no field(s) {unknown}")
        missing = [name for name in declared if name not in values and name not in excluded]
        if missing and not instance:
            defaults = schema_defaults(cls)
            required = [name for name in missing if name not in defaults]
            if required:
                raise ValueError(
                    f"{where}: required field(s) {required} of {cls.__qualname__} not set")
            for name in missing:
                # A dataclass-instance default (`field(default_factory=Trainer)`) is read
                # below through its adapter, like any config value.
                values[name] = defaults[name]
    tree: Tree = {TARGET_KEY: target, VERSION_KEY: target_version(target)}
    for name, value in values.items():
        key = f"{where}.{name}"
        sub = _schema_node(value)
        if sub is not None:
            tree[name] = _materialize_node(sub, key, isinstance(value, VersionedConfig))
            continue
        collection = _structured_collection(value)
        if collection is not None:
            is_list, entries = collection
            materialized = {
                k: None if view is None else _materialize_node(view, f"{key}.{k}", is_instance)
                for k, view, is_instance in entries
            }
            tree[name] = list(materialized.values()) if is_list else materialized
            continue
        leaf = plain_leaf(value)
        if _holds_node(leaf):
            raise ValueError(
                f"{key}: a schema node may only be a field's value or an entry of a list/dict "
                "whose entries are all schema nodes or None (not nested deeper, not mixed with "
                "other values); here its version would not enter identity")
        tree[name] = leaf
    return tree


def _root_name(node: Mapping[str, Any]) -> str:
    return str(node[TARGET_KEY]).rsplit(".", 1)[-1]


def materialize_tree(config: Any) -> Tree:
    """Every top-level versioned node of ``config`` as plain JSON, defaults filled in and each
    node's ``config_version`` resolved from the live class.

    ``config`` is a plain mapping or any object a :mod:`config_ledger.adapters` adapter reads
    as one (an OmegaConf ``DictConfig``, a non-versioned dataclass), or a
    :class:`~config_ledger.versioning.VersionedConfig` instance (dataclass or pydantic), which
    is then the single root node, keyed by its class name. A stored tree is itself a valid
    ``config``: its ``config_version`` keys are ignored and re-resolved."""
    if isinstance(config, VersionedConfig):
        node = _schema_node(config)
        if node is None:
            raise TypeError(f"no adapter reads a {type(config).__name__} instance as a config")
        return {_root_name(node): _materialize_node(node, _root_name(node), instance=True)}
    top = as_mapping(config)
    if top is None:
        raise TypeError(f"config must be a mapping, got {type(config).__name__}")
    tree: Tree = {}
    for key, value in top.items():
        if key == "hydra":
            continue
        node = _schema_node(value)
        if node is not None:
            instance = isinstance(value, VersionedConfig)
            tree[str(key)] = _materialize_node(node, _root_name(node), instance)
        elif isinstance(value, VersionedConfig):
            # Never skip a versioned instance as execution control (it would share one hash).
            raise TypeError(f"config[{key!r}]: no adapter reads a {type(value).__name__} instance")
    return tree


def is_node(value: Any) -> bool:
    """A node of a materialized tree (a plain dict carrying ``_target_``)."""
    return isinstance(value, dict) and TARGET_KEY in value


def _flatten_node(node: Tree, prefix: str, row: Row) -> None:
    for name, value in node.items():
        key = f"{prefix}.{name}"
        if is_node(value):
            _flatten_node(value, key, row)
        elif name in (TARGET_KEY, VERSION_KEY):
            row[key] = str(value)
        elif (entries := node_collection(value)) is not None:
            for k, entry in entries:
                if entry is None:
                    row[f"{key}.{k}"] = ""
                else:
                    _flatten_node(entry, f"{key}.{k}", row)
        else:
            row[key] = encode_leaf(value)


def flatten(tree: Tree) -> Row:
    """The flat row of a materialized tree: what is hashed."""
    row: Row = {}
    for node in tree.values():
        _flatten_node(node, _root_name(node), row)
    return row


def materialize(config: Any) -> Row:
    """Every identity-bearing value of ``config`` as one flat row (see :func:`materialize_tree`)."""
    return flatten(materialize_tree(config))


def version_nodes(row: Row) -> dict[tuple[str, str], list[str]]:
    """``(_target_, version chain) -> [row prefixes of the nodes using it]`` for every node of
    a :class:`~config_ledger.versioning.VersionedConfig` class in ``row``. Unversioned targets
    (empty chain) are left out."""
    nodes: dict[tuple[str, str], list[str]] = {}
    suffix = f".{TARGET_KEY}"
    for key, target in row.items():
        if not key.endswith(suffix):
            continue
        prefix = key[: -len(suffix)]
        chain = row.get(f"{prefix}.{VERSION_KEY}", "")
        if chain:
            nodes.setdefault((target, chain), []).append(prefix)
    return nodes


# --- identity ---------------------------------------------------------------------------------


def canonical_json(row: Row) -> str:
    """Serialization the hash is taken over: sorted keys, no whitespace, ASCII only."""
    return json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def hash_row(row: Row) -> str:
    """xxh3_128 hex digest of :func:`canonical_json`."""
    return xxhash.xxh3_128_hexdigest(canonical_json(row).encode("utf-8"))


@dataclass(frozen=True)
class Identity:
    """A run's identity: the materialized tree and the flat row derived from it (hashed in full)."""

    tree: Tree
    materialized: Row

    @property
    def hash(self) -> str:
        return hash_row(self.materialized)

    def version_nodes(self) -> dict[tuple[str, str], list[str]]:
        return version_nodes(self.materialized)


def compute_identity(config: Any) -> Identity:
    """Materialize ``config`` (a plain mapping, an OmegaConf ``DictConfig``, a dataclass or
    pydantic config instance, or a stored tree) and hash every value."""
    tree = materialize_tree(config)
    return Identity(tree=tree, materialized=flatten(tree))
