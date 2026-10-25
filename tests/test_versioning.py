from dataclasses import dataclass
from typing import ClassVar

import pytest

from config_ledger import VersionedConfig, version_chain
from config_ledger.versioning import identity_excluded_fields, schema_defaults
from tests.conftest import Method, MiningGoldTrainer, Trainer


def test_subclass_without_version_is_rejected():
    with pytest.raises(TypeError, match="does not declare its own"):

        @dataclass
        class NoVersion(VersionedConfig):
            x: int = 0


def test_target_is_injected_as_last_dataclass_field():
    assert Trainer._target_ == f"{Trainer.__module__}.Trainer"
    assert Trainer().__dict__["_target_"].endswith(".Trainer")


def test_version_chain_walks_mro_base_to_leaf():
    assert version_chain(Trainer) == "Trainer:1"
    assert version_chain(MiningGoldTrainer) == "Trainer:1|MiningGoldTrainer:2"


def test_exclusions_from_both_mechanisms():
    assert identity_excluded_fields(Trainer) == {"device"}
    assert identity_excluded_fields(MiningGoldTrainer) == {"device"}
    assert identity_excluded_fields(Method) == {"num_workers"}


def test_schema_defaults_dataclass_and_plain_class():
    assert schema_defaults(Trainer)["layers"] == [8, 8]
    assert schema_defaults(MiningGoldTrainer)["alpha"] == 0.5

    class Plain:
        def __init__(self, a, b=3):
            pass

    assert schema_defaults(Plain) == {"b": 3}
