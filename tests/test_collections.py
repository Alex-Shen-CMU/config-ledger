"""Structured collections: lists / ``_target_``-less dicts of schema nodes are
recursed into, so each entry's version, defaults and key checks reach identity; schema nodes
anywhere else in a leaf are rejected."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import omegaconf
import pytest

from config_ledger import (
    VersionedConfig,
    compute_identity,
    execution_field,
    materialize,
    materialize_tree,
    upgrade_tree,
    version_chain,
)
from config_ledger.identity import VERSION_KEY, version_nodes


@dataclass
class Step(VersionedConfig):
    VERSION: ClassVar[int] = 1
    lr: float = 0.1
    name: str = "s"


@dataclass
class Needs(VersionedConfig):
    VERSION: ClassVar[int] = 1
    size: int


@dataclass
class Pipeline(VersionedConfig):
    VERSION: ClassVar[int] = 1
    train: dict = field(default_factory=dict)
    steps: list = field(default_factory=list)
    extra: Any = None
    cache: Any = execution_field(default=None)


@dataclass
class Group:  # not versioned: read as a dict
    a: Any = None
    b: Any = None


class Unversioned:  # a third-party class: not a VersionedConfig, no fixed field set
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


STEP = f"{Step.__module__}.{Step.__qualname__}"
PIPELINE = f"{Pipeline.__module__}.{Pipeline.__qualname__}"


def _config(**fields: Any) -> dict:
    return {"pipeline": {"_target_": PIPELINE, **fields}}


def _plain() -> dict:
    return _config(train={"fwd": {"_target_": STEP, "lr": 0.5}, "off": None},
                   steps=[{"_target_": STEP}])


def test_entries_become_nodes_with_their_own_rows():
    row = materialize(_plain())
    assert row["Pipeline.train.fwd._target_"] == STEP
    assert row[f"Pipeline.train.fwd.{VERSION_KEY}"] == "Step:1"
    assert row["Pipeline.train.fwd.lr"] == "0.5"
    assert row["Pipeline.train.off"] == ""
    assert row[f"Pipeline.steps.0.{VERSION_KEY}"] == "Step:1"
    assert "Pipeline.train" not in row and "Pipeline.steps" not in row


def test_defaults_are_filled_inside_entries():
    row = materialize(_plain())
    assert row["Pipeline.steps.0.lr"] == "0.1"
    assert row["Pipeline.steps.0.name"] == "s"


def test_dict_dictconfig_and_dataclass_forms_hash_the_same():
    expected = compute_identity(_plain()).hash
    assert compute_identity(omegaconf.OmegaConf.create(_plain())).hash == expected
    instance = Pipeline(train={"fwd": Step(lr=0.5), "off": None}, steps=[Step()])
    assert compute_identity({"pipeline": instance}).hash == expected
    assert compute_identity(instance).hash == expected


def test_unknown_and_missing_keys_in_an_entry_are_rejected():
    match = r"Pipeline.steps.0: Step declares no field\(s\) \['bogus'\]"
    with pytest.raises(ValueError, match=match):
        materialize(_config(steps=[{"_target_": STEP, "bogus": 1}]))
    needs = f"{Needs.__module__}.{Needs.__qualname__}"
    with pytest.raises(ValueError, match=r"Pipeline.train.x: required field\(s\) \['size'\]"):
        materialize(_config(train={"x": {"_target_": needs}}))


def test_hydra_instantiate_keys_in_an_entry_are_not_identity():
    keyed = _config(steps=[{"_target_": STEP, "_convert_": "all"}])
    assert materialize(keyed) == materialize(_config(steps=[{"_target_": STEP}]))


def test_entry_version_bump_changes_the_hash(monkeypatch):
    before = compute_identity(_plain()).hash
    monkeypatch.setattr(Step, "VERSION", 2)
    assert compute_identity(_plain()).hash != before


def test_version_nodes_list_entry_prefixes():
    nodes = version_nodes(materialize(_plain()))
    assert sorted(nodes[(STEP, "Step:1")]) == ["Pipeline.steps.0", "Pipeline.train.fwd"]


def test_unversioned_entries_are_nodes_without_a_version():
    target = f"{Unversioned.__module__}.{Unversioned.__qualname__}"
    row = materialize(_config(steps=[{"_target_": target, "k": 1}]))
    assert row["Pipeline.steps.0.k"] == "1"
    assert row[f"Pipeline.steps.0.{VERSION_KEY}"] == ""
    assert all(key[0] != target for key in version_nodes(row))


@pytest.mark.parametrize("value, encoded", [
    ([], "[]"),
    ({}, "{}"),
    ({"x": None}, '{"x": null}'),
    ([[1, 2], {"a": [3]}], '[[1, 2], {"a": [3]}]'),
])
def test_collections_without_nodes_stay_leaves(value, encoded):
    assert materialize(_config(extra=value))["Pipeline.extra"] == encoded


def test_tree_keeps_the_container_shape_and_re_materializes_to_the_same_identity():
    identity = compute_identity(_plain())
    tree = identity.tree["pipeline"]
    assert isinstance(tree["steps"], list) and tree["steps"][0][VERSION_KEY] == "Step:1"
    assert tree["train"]["off"] is None
    assert compute_identity(identity.tree) == identity


def test_nonversioned_dataclass_of_nodes_is_a_structured_collection():
    row = materialize(_config(extra=Group(a=Step())))
    assert row == materialize(_config(extra={"a": {"_target_": STEP}, "b": None}))
    assert row[f"Pipeline.extra.a.{VERSION_KEY}"] == "Step:1"


@pytest.mark.parametrize("value", [
    pytest.param([{"_target_": STEP}, 3], id="mixed-list"),
    pytest.param({"a": {"_target_": STEP}, "b": 1}, id="mixed-dict"),
    pytest.param([Step(), "x"], id="mixed-list-of-instances"),
    pytest.param([[{"_target_": STEP}]], id="list-of-lists"),
    pytest.param({"a": [{"_target_": STEP}]}, id="dict-of-lists"),
    pytest.param(Group(a=[Step()], b=1), id="nonversioned-dataclass-nesting-a-node"),
])
def test_nodes_hidden_in_a_leaf_are_rejected(value):
    with pytest.raises(ValueError, match="Pipeline.extra: a schema node may only be"):
        materialize(_config(extra=value))


def test_dictconfig_mixed_list_is_rejected():
    cfg = omegaconf.OmegaConf.create(_config(extra=[{"_target_": STEP}, 3]))
    with pytest.raises(ValueError, match="Pipeline.extra: a schema node may only be"):
        materialize(cfg)


def test_excluded_fields_are_not_checked():
    assert "Pipeline.cache" not in materialize(_config(cache=[{"_target_": STEP}, 3]))


def _step_v1_to_v2(cls, old):
    if old[VERSION_KEY] == version_chain(cls):
        return None
    return {**old, "lr": old["lr"] * 10}


def test_upgrade_tree_upgrades_entries_before_their_parent(monkeypatch):
    tree = materialize_tree(_plain())  # stored while Step is at 1
    seen = []

    def pipeline_upgrade(cls, old):
        seen.append(old)
        return None

    monkeypatch.setattr(Step, "VERSION", 2)
    monkeypatch.setattr(Step, "upgrade_config", classmethod(_step_v1_to_v2))
    monkeypatch.setattr(Pipeline, "upgrade_config", classmethod(pipeline_upgrade))
    errors: list[str] = []
    upgraded = upgrade_tree(tree, errors)
    assert errors == [] and upgraded is not None
    assert upgraded["pipeline"]["train"]["fwd"]["lr"] == 5.0
    assert upgraded["pipeline"]["train"]["off"] is None
    assert upgraded["pipeline"]["steps"][0]["lr"] == 1.0
    assert upgraded["pipeline"]["steps"][0][VERSION_KEY] == "Step:2"
    assert seen[0]["steps"][0][VERSION_KEY] == "Step:2"  # the parent saw upgraded entries


def test_upgrade_tree_reports_failing_entries_by_path(monkeypatch):
    tree = materialize_tree(_plain())
    monkeypatch.setattr(Step, "VERSION", 2)
    errors: list[str] = []
    assert upgrade_tree(tree, errors) is None
    assert any(e.startswith("pipeline.steps.0 (") for e in errors)
    assert any(e.startswith("pipeline.train.fwd (") for e in errors)
