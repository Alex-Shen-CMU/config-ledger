from dataclasses import dataclass
from typing import ClassVar

import omegaconf
import pytest

from config_ledger import VersionedConfig, compute_identity, materialize, materialize_tree
from config_ledger.identity import VERSION_KEY, encode_leaf, flatten, plain_leaf, version_nodes
from tests.conftest import Method, MiningGoldTrainer, Trainer, make_config


@dataclass
class Needs(VersionedConfig):
    VERSION: ClassVar[int] = 1
    size: int


def test_materialize_records_target_and_version_for_every_node():
    row = materialize(make_config())
    assert row["Method._target_"] == make_config().method._target_
    assert row[f"Method.{VERSION_KEY}"] == "Method:1"
    assert row[f"Method.trainer.{VERSION_KEY}"] == "Trainer:1"
    assert row["Method.trainer.layers"] == "[8, 8]"


def test_excluded_fields_never_materialize():
    row = materialize(make_config(device="cuda"))
    assert "Method.trainer.device" not in row
    assert "Method.num_workers" not in row


def test_top_level_loose_keys_are_invisible():
    a = compute_identity(make_config())
    b = compute_identity(make_config(overwrite=True, output_dir="/x", hydra={"job": {"name": "n"}}))
    assert a.hash == b.hash


def test_root_key_name_is_not_identity():
    cfg = make_config()
    renamed = omegaconf.OmegaConf.create({"other_key": cfg.method})
    assert compute_identity(cfg).hash == compute_identity(renamed).hash


def test_explicit_default_equals_omitted_default():
    """Spelling a default out in yaml is the same identity as leaving it implicit."""
    explicit = compute_identity(make_config())
    implicit = compute_identity(omegaconf.OmegaConf.create(
        {"method": {"_target_": make_config().method._target_, "trainer": {"_target_": Trainer._target_}}}
    ))
    assert explicit.hash == implicit.hash
    assert implicit.materialized["Method.trainer.learning_rate"] == "0.1"


def test_every_value_enters_the_row_including_defaults():
    row = materialize(make_config())
    assert row["Method.trainer.seed"] == "0"
    assert row["Method.name"] == "fixture"


def test_omitted_node_default_is_filled_from_its_dataclass_default():
    row = materialize({"method": {"_target_": Method._target_}})
    assert row == materialize(make_config())


def test_unknown_field_is_rejected():
    cfg = omegaconf.OmegaConf.to_container(make_config())
    cfg["method"]["trainer"]["learning_rat"] = 0.2  # type: ignore[index]
    match = r"Method.trainer: Trainer declares no field\(s\) \['learning_rat'\]"
    with pytest.raises(ValueError, match=match):
        materialize(cfg)


@dataclass
class DeclaresConvert(VersionedConfig):
    VERSION: ClassVar[int] = 1
    _convert_: str = "all"
    size: int = 1


@pytest.mark.parametrize("key, value", [("_convert_", "all"), ("_recursive_", False), ("_partial_", True)])
def test_hydra_instantiate_keys_are_not_identity(key, value):
    """They control how Hydra builds the node, not what the run is."""
    cfg = omegaconf.OmegaConf.to_container(make_config())
    cfg["method"]["trainer"][key] = value  # type: ignore[index]
    with_key = compute_identity(omegaconf.OmegaConf.create(cfg))
    assert with_key.hash == compute_identity(make_config()).hash
    assert not any(key in column for column in with_key.materialized)
    assert key not in materialize_tree(cfg)["method"]["trainer"]


def test_declared_hydra_instantiate_key_is_not_identity():
    from config_ledger.versions import identity_fields

    plain = materialize({"d": {"_target_": DeclaresConvert._target_}})
    spelled = materialize({"d": {"_target_": DeclaresConvert._target_, "_convert_": "none"}})
    assert plain == spelled == materialize(DeclaresConvert(_convert_="object"))
    assert "DeclaresConvert._convert_" not in plain
    assert identity_fields(DeclaresConvert) == ["size"]


def test_hydra_positional_args_are_rejected():
    cfg = omegaconf.OmegaConf.to_container(make_config())
    cfg["method"]["trainer"]["_args_"] = [0.1]  # type: ignore[index]
    with pytest.raises(ValueError, match=r"Method.trainer: _args_ .* not supported"):
        materialize(cfg)


def test_missing_required_field_is_rejected():
    with pytest.raises(ValueError, match=r"required field\(s\) \['size'\]"):
        materialize({"n": {"_target_": Needs._target_}})
    assert materialize({"n": {"_target_": Needs._target_, "size": 2}})["Needs.size"] == "2"


def test_config_version_is_reserved():
    with pytest.raises(TypeError, match="reserved"):

        @dataclass
        class Bad(VersionedConfig):
            VERSION: ClassVar[int] = 1
            config_version: int = 0


def test_tree_is_plain_json_and_flattens_to_the_row():
    import json

    identity = compute_identity(make_config(layers=(4, 2)))
    tree = identity.tree
    assert tree["method"]["trainer"] == {
        "_target_": Trainer._target_, VERSION_KEY: "Trainer:1",
        "learning_rate": 0.1, "seed": 0, "layers": [4, 2],
    }
    assert json.loads(json.dumps(tree)) == tree
    assert flatten(tree) == identity.materialized


def test_stored_tree_is_a_valid_config_with_the_same_identity():
    identity = compute_identity(make_config(seed=4))
    assert compute_identity(identity.tree).hash == identity.hash


def test_stored_tree_versions_are_re_resolved(monkeypatch):
    tree = materialize_tree(make_config())
    monkeypatch.setattr(Trainer, "VERSION", 3)
    assert materialize(tree)[f"Method.trainer.{VERSION_KEY}"] == "Trainer:3"


def test_plain_leaf_encodes_like_the_original():
    import enum
    from pathlib import Path

    class Color(enum.Enum):
        RED = 1

    for value in (None, True, 3, 0.5, "s", Color.RED, Path("/a"), (1, Path("/x"), Color.RED),
                  {"b": [Color.RED], "a": (1,)}, [{"z": 1, "a": 2}]):
        assert encode_leaf(plain_leaf(value)) == encode_leaf(value)


def test_version_nodes_group_prefixes_by_class_and_chain():
    row = materialize(make_config(MiningGoldTrainer._target_, alpha=0.9))
    assert version_nodes(row) == {
        (Method._target_, "Method:1"): ["Method"],
        (MiningGoldTrainer._target_, "Trainer:1|MiningGoldTrainer:2"): ["Method.trainer"],
    }


def test_changing_a_field_changes_hash():
    assert compute_identity(make_config()).hash != compute_identity(make_config(learning_rate=0.2)).hash


def test_key_order_does_not_matter():
    cfg = make_config()
    shuffled = omegaconf.OmegaConf.create({"method": dict(reversed(list(cfg.method.items())))})
    assert compute_identity(cfg).hash == compute_identity(shuffled).hash


def test_version_bump_on_base_changes_subclass_hash(monkeypatch):
    before = compute_identity(make_config(MiningGoldTrainer._target_, alpha=0.9)).hash
    monkeypatch.setattr(Trainer, "VERSION", 2)
    assert compute_identity(make_config(MiningGoldTrainer._target_, alpha=0.9)).hash != before


def test_encode_leaf_accepts_builtins_enum_and_paths():
    import enum
    from pathlib import Path

    from config_ledger.identity import encode_leaf

    class Color(enum.Enum):
        RED = 1

    assert encode_leaf(None) == ""
    assert encode_leaf(True) == "True"
    assert encode_leaf(3) == "3"
    assert encode_leaf(0.5) == "0.5"
    assert encode_leaf(Color.RED) == "RED"
    assert encode_leaf(Path("/a/b")) == "/a/b"
    assert encode_leaf((1, Path("/x"), Color.RED)) == '[1, "/x", "RED"]'
    assert encode_leaf({"b": 1, "a": [2]}) == '{"a": [2], "b": 1}'


def test_encode_leaf_rejects_unknown_objects():
    import datetime

    import pytest

    from config_ledger.identity import encode_leaf

    class Opaque:
        pass

    for value in (Opaque(), datetime.date(2026, 1, 1), [Opaque()], {"a": {1, 2}}):
        with pytest.raises(TypeError, match="cannot encode"):
            encode_leaf(value)


def test_unknown_object_in_a_node_fails_instead_of_hashing_its_repr():
    import pytest

    cfg = omegaconf.OmegaConf.to_container(make_config())
    cfg["method"]["name"] = object()  # type: ignore[index]
    with pytest.raises(TypeError, match="cannot encode"):
        materialize(cfg)
