"""Plain dataclass instances as configs: same rows as their dict / DictConfig forms."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import ClassVar

import omegaconf
import pytest

from config_ledger import VersionedConfig, compute_identity, execution_field, materialize
from config_ledger.adapters import adapter_for
from config_ledger.adapters.dataclass import DataclassAdapter
from tests.conftest import Method, MiningGoldTrainer, Trainer, make_config


@dataclass
class Optimizer:  # not versioned: one JSON leaf
    lr: float = 0.1
    betas: tuple = (0.9, 0.99)
    device: str = execution_field(default="cpu")


@dataclass
class Holder(VersionedConfig):
    VERSION: ClassVar[int] = 1
    optimizer: Optimizer = field(default_factory=Optimizer)
    schedule: list = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Frozen(VersionedConfig):
    VERSION: ClassVar[int] = 1
    x: int = 1


@dataclass
class TopConfig:  # not versioned: walked like a top-level dict
    method: Method = field(default_factory=Method)
    output_dir: str = "/out"
    overwrite: bool = False


def test_dispatch():
    assert isinstance(adapter_for(Trainer()), DataclassAdapter)
    assert isinstance(adapter_for(Optimizer()), DataclassAdapter)
    assert adapter_for(Trainer) is None


@pytest.mark.parametrize(
    "method, cfg",
    [
        (Method(), make_config()),
        (Method(trainer=Trainer(seed=3, layers=[4, 2])), make_config(seed=3, layers=[4, 2])),
        (Method(trainer=MiningGoldTrainer(alpha=0.9)),
         make_config(MiningGoldTrainer._target_, alpha=0.9)),
    ],
)
def test_instance_tree_matches_asdict_structured_and_dictconfig(method, cfg):
    expected = compute_identity(cfg)
    for config in (
        {"method": method},
        {"method": asdict(method)},
        omegaconf.OmegaConf.structured({"method": method}),
        method,  # the versioned instance as the whole config: the single root node
    ):
        identity = compute_identity(config)
        assert identity.materialized == expected.materialized
        assert identity.hash == expected.hash


def test_root_instance_is_keyed_by_class_name():
    assert list(compute_identity(Method()).tree) == ["Method"]


def test_nonversioned_dataclass_is_walked_like_a_top_level_dict():
    top = TopConfig(output_dir="/elsewhere", overwrite=True)
    assert compute_identity(top).hash == compute_identity(make_config()).hash
    assert list(compute_identity(top).tree) == ["method"]


def test_execution_fields_on_instances_never_enter_identity():
    base = compute_identity(Method())
    assert compute_identity(Method(trainer=Trainer(device="cuda"))).hash == base.hash
    assert compute_identity(Method(num_workers=99)).hash == base.hash  # IDENTITY_EXCLUDE
    assert compute_identity(Method(trainer=Trainer(seed=1))).hash != base.hash


def test_nonversioned_leaf_honors_its_own_exclusions():
    row = materialize(Holder(optimizer=Optimizer(device="cuda")))
    assert row["Holder.optimizer"] == '{"betas": [0.9, 0.99], "lr": 0.1}'
    assert materialize(Holder()) == row
    # The default Optimizer() instance is read the same way when the field is omitted.
    assert materialize({"h": {"_target_": Holder._target_}}) == row


def test_dataclasses_nested_in_a_list_leaf():
    row = materialize(Holder(schedule=[Optimizer(lr=1.0), Optimizer(lr=0.5)]))
    as_dicts = [{"lr": 1.0, "betas": [0.9, 0.99]}, {"lr": 0.5, "betas": [0.9, 0.99]}]
    assert row == materialize({"h": {"_target_": Holder._target_, "schedule": as_dicts}})
    tree = compute_identity(Holder(schedule=[Optimizer(lr=1.0)])).tree
    assert tree["Holder"]["schedule"] == [{"lr": 1.0, "betas": [0.9, 0.99]}]


def test_frozen_slots_dataclass():
    as_dict = {"f": {"_target_": Frozen._target_, "x": 2}}
    assert materialize({"f": Frozen(x=2)}) == materialize(as_dict)


def test_target_field_mismatch_raises():
    with pytest.raises(ValueError, match="always stands for its own class"):
        materialize({"t": Trainer(_target_=MiningGoldTrainer._target_)})  # type: ignore[call-arg]


def test_versioned_instance_under_a_key_is_never_skipped():
    # This used to be rejected; it must not fall through as execution control.
    assert materialize({"trainer": Trainer()}) != materialize({})
