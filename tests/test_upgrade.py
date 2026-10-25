"""upgrade_config / upgrade_tree: bottom-up, all-or-nothing, re-validated."""

import pytest

from config_ledger import (
    UpgradeImpossible,
    compute_identity,
    parse_chain,
    upgrade_tree,
    version_chain,
)
from config_ledger.identity import VERSION_KEY
from tests.conftest import Method, MiningGoldTrainer, Trainer, make_config


def _trainer_v1_to_v2(cls, old):
    """Trainer:2 measures learning rates in units of 1e-2."""
    if old[VERSION_KEY] == version_chain(cls):
        return None
    if parse_chain(old[VERSION_KEY])["Trainer"] == 1:
        return {**old, "learning_rate": old["learning_rate"] * 100}
    raise UpgradeImpossible("unknown Trainer version")


@pytest.fixture
def trainer_v2(monkeypatch):
    monkeypatch.setattr(Trainer, "upgrade_config", classmethod(_trainer_v1_to_v2))
    monkeypatch.setattr(Trainer, "VERSION", 2)


def test_parse_chain():
    assert parse_chain("Trainer:1|MiningGoldTrainer:2") == {"Trainer": 1, "MiningGoldTrainer": 2}
    assert parse_chain("") == {}


def test_default_upgrade_config():
    assert Trainer.upgrade_config({VERSION_KEY: "Trainer:1"}) is None
    with pytest.raises(UpgradeImpossible, match="no upgrade from 'Trainer:0'"):
        Trainer.upgrade_config({VERSION_KEY: "Trainer:0"})


def test_nothing_to_upgrade_returns_the_same_identity():
    identity = compute_identity(make_config(seed=3))
    upgraded = upgrade_tree(identity.tree)
    assert upgraded == identity.tree


def test_upgrade_path_is_applied_and_versions_are_live(trainer_v2):
    tree = compute_identity(make_config(learning_rate=0.05)).tree
    tree["method"]["trainer"][VERSION_KEY] = "Trainer:1"  # as stored before the bump
    upgraded = upgrade_tree(tree)
    assert upgraded["method"]["trainer"]["learning_rate"] == 5.0
    assert upgraded["method"]["trainer"][VERSION_KEY] == "Trainer:2"
    assert upgraded["method"][VERSION_KEY] == "Method:1"


def _stored_v1():
    return compute_identity(make_config()).tree  # materialized while Trainer is at 1


def test_missing_upgrade_path_fails_with_node_path(monkeypatch):
    tree = _stored_v1()
    monkeypatch.setattr(Trainer, "VERSION", 2)
    errors = []
    assert upgrade_tree(tree, errors) is None
    assert errors == [f"method.trainer ({Trainer._target_} @ Trainer:1): "
                      "Trainer defines no upgrade from 'Trainer:1'"]


def test_children_are_upgraded_before_parents(monkeypatch, trainer_v2):
    tree = _stored_v1_at(monkeypatch)
    seen = []

    def method_upgrade(cls, old):
        seen.append(old["trainer"][VERSION_KEY])
        return None if old[VERSION_KEY] == version_chain(cls) else {**old, "name": "upgraded"}

    monkeypatch.setattr(Method, "upgrade_config", classmethod(method_upgrade))
    monkeypatch.setattr(Method, "VERSION", 2)
    upgraded = upgrade_tree(tree)
    assert seen == ["Trainer:2"]
    assert upgraded["method"]["name"] == "upgraded"
    assert upgraded["method"]["trainer"]["learning_rate"] == 5.0


def _stored_v1_at(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(Trainer, "VERSION", 1)
        return compute_identity(make_config(learning_rate=0.05)).tree


def test_every_failing_node_is_reported(monkeypatch):
    gold = make_config(MiningGoldTrainer._target_).method
    tree = compute_identity({"a": make_config().method, "b": gold}).tree
    monkeypatch.setattr(Trainer, "VERSION", 2)
    errors = []
    assert upgrade_tree(tree, errors) is None
    assert [e.split(" ")[0] for e in errors] == ["a.trainer", "b.trainer"]


def test_upgraded_config_is_validated_against_the_live_schema(monkeypatch):
    tree = _stored_v1()
    monkeypatch.setattr(Trainer, "VERSION", 2)
    monkeypatch.setattr(Trainer, "upgrade_config",
                        classmethod(lambda cls, old: {**old, "momentum": 0.9}))
    errors = []
    assert upgrade_tree(tree, errors) is None
    assert "invalid under the live schema" in errors[0] and "momentum" in errors[0]


def test_fields_left_out_take_their_defaults(monkeypatch):
    tree = compute_identity(make_config(seed=9)).tree
    monkeypatch.setattr(Trainer, "VERSION", 2)
    monkeypatch.setattr(Trainer, "upgrade_config",
                        classmethod(lambda cls, old: {k: v for k, v in old.items() if k != "seed"}))
    assert upgrade_tree(tree)["method"]["trainer"]["seed"] == 0


def test_none_for_an_old_version_is_an_error(monkeypatch):
    tree = _stored_v1()
    monkeypatch.setattr(Trainer, "VERSION", 2)
    monkeypatch.setattr(Trainer, "upgrade_config", classmethod(lambda cls, old: None))
    errors = []
    assert upgrade_tree(tree, errors) is None
    assert "returned None" in errors[0]


def test_unimportable_class_fails():
    tree = _stored_v1()
    tree["method"]["trainer"]["_target_"] = "gone.Trainer"
    errors = []
    assert upgrade_tree(tree, errors) is None
    assert "no longer imports" in errors[0]
