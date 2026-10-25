"""Carry a stored config tree forward to the live versions of its classes.

Every run stores its materialized *tree* (plain JSON, one dict per node, each with its
``_target_`` and stored ``config_version``; see :mod:`config_ledger.identity`). After a
``VERSION`` bump, :func:`upgrade_tree` walks that tree **bottom-up** and hands each node to its
class's :meth:`~config_ledger.versioning.VersionedConfig.upgrade_config`, so a parent always
sees already-upgraded children (carrying their live ``config_version``), including the
entries of its structured collections. Each
``upgrade_config``:

* returns ``None``: nothing to do, the stored version is the live one;
* returns a new node: the upgrade;
* raises :class:`~config_ledger.versioning.UpgradeImpossible`: no path from this old config.

The upgraded tree is then re-materialized against the live classes, which fills omitted
fields from their defaults, records the live version chains, and rejects unknown fields or
missing required ones. Any failure anywhere fails the whole tree: half-upgraded configs are
never registered.

``config-ledger upgrade`` (:mod:`config_ledger.maintenance`) applies this to every fresh run
under an old version and re-registers the result against the same folder.

Status (2026-10-17): introduced. 2026-10-18: structured collections.
"""

from __future__ import annotations

import copy
from typing import Any

from config_ledger.identity import Tree, is_node, materialize_tree, node_collection
from config_ledger.versioning import (
    TARGET_KEY,
    VERSION_KEY,
    UpgradeImpossible,
    is_versioned,
    resolve_target,
    version_chain,
)


def upgrade_tree(tree: Tree, errors: list[str] | None = None) -> Tree | None:
    """The upgraded, re-materialized ``tree``, or None if any node cannot be upgraded.

    Every failure is appended to ``errors`` (if given) as ``"<node path> (<_target_> @
    <stored version>): <reason>"``; all nodes are attempted, so one call reports every problem.
    """
    found: list[str] = []
    upgraded = {key: _upgrade_node(node, str(key), found) for key, node in tree.items()}
    if not found:
        try:
            return materialize_tree(upgraded)
        except (ValueError, TypeError, ImportError, AttributeError) as exc:
            found.append(f"upgraded config is invalid under the live schema: {exc}")
    if errors is not None:
        errors.extend(found)
    return None


def _upgrade_value(value: Any, where: str, errors: list[str]) -> Any:
    if is_node(value):
        return _upgrade_node(value, where, errors)
    entries = node_collection(value)
    if entries is None:
        return value
    items = {k: None if v is None else _upgrade_node(v, f"{where}.{k}", errors) for k, v in entries}
    return list(items.values()) if isinstance(value, list) else items


def _upgrade_node(node: dict[str, Any], where: str, errors: list[str]) -> dict[str, Any]:
    children = {name: _upgrade_value(value, f"{where}.{name}", errors)
                for name, value in node.items()}
    target = str(node[TARGET_KEY])
    stored = node.get(VERSION_KEY, "")
    label = f"{where} ({target} @ {stored or 'unversioned'})"
    try:
        cls = resolve_target(target)
    except (ImportError, AttributeError) as exc:
        errors.append(f"{label}: class no longer imports: {exc}")
        return children
    if not is_versioned(cls):
        return children
    try:
        new = cls.upgrade_config(copy.deepcopy(children))
    except UpgradeImpossible as exc:
        errors.append(f"{label}: {exc}")
        return children
    if new is None:
        if stored != version_chain(cls):
            errors.append(f"{label}: upgrade_config returned None (no upgrade needed) but the live "
                          f"version is {version_chain(cls)!r}")
        return children
    upgraded = dict(new)
    upgraded.setdefault(TARGET_KEY, target)
    try:  # what the parent sees; re-resolved again on re-materialization
        upgraded[VERSION_KEY] = version_chain(resolve_target(str(upgraded[TARGET_KEY])))
    except (ImportError, AttributeError) as exc:
        errors.append(f"{label}: upgrade_config returned an unimportable _target_: {exc}")
    return upgraded
