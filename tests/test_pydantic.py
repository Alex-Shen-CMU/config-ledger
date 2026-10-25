"""Pydantic models: same rows as the plain-dict form, exclusions from every
declaration, defaults from model fields, and a versioned model as the root config."""

from __future__ import annotations

from dataclasses import asdict, field
from pathlib import Path
from typing import ClassVar

import pytest

pydantic = pytest.importorskip("pydantic")

from pydantic import BaseModel, Field  # noqa: E402
from pydantic.dataclasses import dataclass as pydantic_dataclass  # noqa: E402

from config_ledger import (  # noqa: E402
    VersionedConfig,
    compute_identity,
    execution_field,
    materialize,
)
from config_ledger.adapters import adapter_for  # noqa: E402
from config_ledger.adapters.dataclass import DataclassAdapter  # noqa: E402
from config_ledger.adapters.pydantic import PydanticAdapter  # noqa: E402
from config_ledger.identity import VERSION_KEY  # noqa: E402
from config_ledger.versioning import (  # noqa: E402
    identity_excluded_fields,
    schema_defaults,
    version_chain,
)


class Trainer(VersionedConfig, BaseModel):
    VERSION: ClassVar[int] = 1
    learning_rate: float = 0.1
    seed: int = 0
    layers: list[int] = Field(default_factory=lambda: [8, 8])
    device: str = execution_field(default="cpu")


class MiningGoldTrainer(Trainer):
    VERSION: ClassVar[int] = 2
    alpha: float = 0.5


class Method(BaseModel, VersionedConfig):  # either base order works
    VERSION: ClassVar[int] = 1
    IDENTITY_EXCLUDE: ClassVar = ("num_workers",)
    name: str = "fixture"
    trainer: Trainer = Field(default_factory=Trainer)
    num_workers: int = 4
    out: Path = Path("/tmp/out")


class Optim(BaseModel):  # not versioned: one JSON leaf, like a nested plain dict
    kind: str = "sgd"
    momentum: float = 0.9


class WithLeafModel(VersionedConfig, BaseModel):
    VERSION: ClassVar[int] = 1
    optim: Optim = Field(default_factory=Optim)


class ExcaTuple(VersionedConfig, BaseModel):
    VERSION: ClassVar[int] = 1
    _exclude_from_cls_uid = ("y",)
    x: int = 1
    y: str = "h"


class ExcaClassVar(VersionedConfig, BaseModel):
    VERSION: ClassVar[int] = 1
    _exclude_from_cls_uid: ClassVar = ("y",)
    x: int = 1
    y: str = "h"


class ExcaClassmethod(VersionedConfig, BaseModel):
    VERSION: ClassVar[int] = 1
    x: int = 1
    y: str = "h"

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return ["y"]


class ExcaMethod(VersionedConfig, BaseModel):
    VERSION: ClassVar[int] = 1
    x: int = 1
    y: str = "h"

    def _exclude_from_cls_uid(self) -> list[str]:
        return ["y"] if self.x == 1 else []


class ExcaAll(VersionedConfig, BaseModel):
    VERSION: ClassVar[int] = 1
    _exclude_from_cls_uid = (".",)
    x: int = 1


def _as_dict(model: BaseModel) -> dict:
    """The same config written as a plain dict, as a Hydra yaml would give it."""
    out = {k: _as_dict(v) if isinstance(v, VersionedConfig) else v for k, v in model}
    out = {k: str(v) if isinstance(v, Path) else v for k, v in out.items()}
    if isinstance(model, VersionedConfig):
        out["_target_"] = type(model)._target_
    return {k: (v.model_dump() if isinstance(v, BaseModel) else v) for k, v in out.items()}


# --- schema classes -----------------------------------------------------------------------


def test_target_is_a_class_attribute_not_a_field():
    assert Trainer._target_ == f"{__name__}.Trainer"
    assert MiningGoldTrainer._target_ == f"{__name__}.MiningGoldTrainer"
    assert "_target_" not in Trainer.model_fields
    assert "_target_" not in Trainer().model_dump()
    assert Trainer.__private_attributes__ == {}


def test_version_chain_and_missing_version():
    assert version_chain(MiningGoldTrainer) == "Trainer:1|MiningGoldTrainer:2"
    with pytest.raises(TypeError, match="does not declare its own"):

        class NoVersion(VersionedConfig, BaseModel):
            x: int = 0


def test_execution_field_becomes_a_pydantic_field():
    assert Trainer().device == "cpu"
    assert Trainer(device="cuda").device == "cuda"
    assert identity_excluded_fields(Trainer) == {"device"}
    assert identity_excluded_fields(MiningGoldTrainer) == {"device"}
    assert identity_excluded_fields(Method) == {"num_workers"}


def test_schema_defaults_from_model_fields():
    defaults = schema_defaults(MiningGoldTrainer)
    assert defaults["layers"] == [8, 8]
    assert defaults["alpha"] == 0.5
    assert isinstance(schema_defaults(Method)["trainer"], Trainer)


@pytest.mark.parametrize("cls", [ExcaTuple, ExcaClassVar, ExcaClassmethod])
def test_exca_class_level_exclusions(cls):
    assert identity_excluded_fields(cls) == {"y"}
    assert f"{cls.__name__}.y" not in materialize(cls(y="other"))
    assert materialize(cls(x=2))[f"{cls.__name__}.x"] == "2"


def test_exca_instance_method_is_evaluated_per_instance():
    assert identity_excluded_fields(ExcaMethod) == set()  # needs an instance
    assert "ExcaMethod.y" not in materialize(ExcaMethod())
    assert materialize(ExcaMethod(x=2))["ExcaMethod.y"] == "h"


def test_exca_dot_excludes_every_field_but_keeps_class_identity():
    assert materialize(ExcaAll(x=5)) == {
        "ExcaAll._target_": ExcaAll._target_,
        f"ExcaAll.{VERSION_KEY}": "ExcaAll:1",
    }


def test_exca_declaration_errors():
    with pytest.raises(ValueError, match="unknown fields"):

        class Unknown(VersionedConfig, BaseModel):
            VERSION: ClassVar[int] = 1
            _exclude_from_cls_uid = ("nope",)

        identity_excluded_fields(Unknown)

    class Stringy(VersionedConfig, BaseModel):
        VERSION: ClassVar[int] = 1
        _exclude_from_cls_uid = "x"
        x: int = 1

    with pytest.raises(TypeError, match="not a string"):
        identity_excluded_fields(Stringy)


# --- config objects -------------------------------------------------------------------------


def test_dispatch():
    assert isinstance(adapter_for(Trainer()), PydanticAdapter)
    assert isinstance(adapter_for(Optim()), PydanticAdapter)


@pytest.mark.parametrize(
    "method",
    [
        Method(),
        Method(trainer=Trainer(seed=3, layers=[4, 2]), num_workers=16),
        Method(trainer=MiningGoldTrainer(alpha=0.9, device="cuda")),
    ],
)
def test_model_and_plain_dict_give_same_row_and_hash(method):
    plain = {"method": _as_dict(method), "overwrite": True}
    assert materialize({"method": method}) == materialize(plain)
    assert compute_identity({"method": method}).hash == compute_identity(plain).hash


def test_versioned_model_as_root_config():
    method = Method(trainer=Trainer(seed=3))
    row = materialize(method)
    assert row["Method.trainer.seed"] == "3"
    assert row["Method.out"] == "/tmp/out"
    assert "Method.num_workers" not in row and "Method.trainer.device" not in row
    assert compute_identity(method).hash == compute_identity({"any_key": method}).hash


def test_models_hash_every_identity_field():
    identity = compute_identity(Method(trainer=Trainer(seed=3), num_workers=8))
    row = identity.materialized
    fields = {k: v for k, v in row.items() if not k.endswith(("_target_", VERSION_KEY))}
    assert fields == {"Method.name": "fixture", "Method.out": "/tmp/out",
                      "Method.trainer.learning_rate": "0.1", "Method.trainer.seed": "3",
                      "Method.trainer.layers": "[8, 8]"}
    on_gpu = Method(trainer=Trainer(device="cuda"))
    assert compute_identity(Method()).hash == compute_identity(on_gpu).hash


def test_unversioned_model_is_one_json_leaf():
    row = materialize(WithLeafModel(optim=Optim(momentum=0.5)))
    assert row["WithLeafModel.optim"] == '{"kind": "sgd", "momentum": 0.5}'
    plain = {"w": {"_target_": WithLeafModel._target_, "optim": {"kind": "sgd", "momentum": 0.5}}}
    model = WithLeafModel(optim=Optim(momentum=0.5))
    assert compute_identity(model).hash == compute_identity(plain).hash
    # An omitted leaf model is filled from its default.
    omitted = {"w": {"_target_": WithLeafModel._target_}}
    assert compute_identity(omitted).hash == compute_identity(WithLeafModel()).hash


class StepModel(VersionedConfig, BaseModel):
    VERSION: ClassVar[int] = 1
    lr: float = 0.1


class PipelineModel(VersionedConfig, BaseModel):
    VERSION: ClassVar[int] = 1
    steps: list[StepModel] = Field(default_factory=list)
    named: dict[str, StepModel | None] = Field(default_factory=dict)


def test_model_collections_are_structured(monkeypatch):
    """A list / dict of versioned models is recursed into, like its dict form."""
    model = PipelineModel(steps=[StepModel(lr=0.3)], named={"a": StepModel(), "b": None})
    step, pipeline = (f"{c.__module__}.{c.__qualname__}" for c in (StepModel, PipelineModel))
    plain = {"p": {"_target_": pipeline, "steps": [{"_target_": step, "lr": 0.3}],
                   "named": {"a": {"_target_": step}, "b": None}}}
    row = materialize(model)
    assert row == materialize(plain)
    assert row["PipelineModel.steps.0.lr"] == "0.3"
    assert row[f"PipelineModel.named.a.{VERSION_KEY}"] == "StepModel:1"
    assert row["PipelineModel.named.b"] == ""
    before = compute_identity(model).hash
    monkeypatch.setattr(StepModel, "VERSION", 2)
    assert compute_identity(model).hash != before


def test_version_bump_changes_hash(monkeypatch):
    before = compute_identity(Method(trainer=MiningGoldTrainer())).hash
    monkeypatch.setattr(Trainer, "VERSION", 2)
    assert compute_identity(Method(trainer=MiningGoldTrainer())).hash != before


def test_dict_config_naming_a_model_class_fills_defaults_and_rejects_unknown_keys():
    assert materialize({"t": {"_target_": Trainer._target_}}) == materialize(Trainer())
    with pytest.raises(ValueError, match="declares no field"):
        materialize({"t": {"_target_": Trainer._target_, "nope": 1}})


class Open(VersionedConfig, BaseModel):
    VERSION: ClassVar[int] = 1
    model_config = {"extra": "allow"}
    a: int = 0


def test_model_allowing_extra_fields_takes_extra_keys_as_written():
    row = materialize({"o": {"_target_": Open._target_, "a": 1, "b": 2}})
    assert row["Open.b"] == "2"


# --- pydantic dataclasses ---------------------------------------------------------
# Not BaseModels: read by the stdlib dataclass adapter (plain field reads, after pydantic's
# validation), with `_target_` a real dataclass field as for a stdlib dataclass.


@pydantic_dataclass
class PdOptimizer:  # not versioned: one JSON leaf
    lr: float = 0.1
    device: str = execution_field(default="cpu")


@pydantic_dataclass
class PdTrainer(VersionedConfig):
    VERSION: ClassVar[int] = 1
    IDENTITY_EXCLUDE: ClassVar = ("num_workers",)
    learning_rate: float = 0.1
    num_workers: int = 4
    device: str = execution_field(default="cpu")
    optimizer: PdOptimizer = field(default_factory=PdOptimizer)


def test_pydantic_dataclass_is_read_by_the_dataclass_adapter():
    assert isinstance(adapter_for(PdTrainer()), DataclassAdapter)
    assert identity_excluded_fields(PdTrainer) == {"num_workers", "device"}
    assert schema_defaults(PdTrainer)["learning_rate"] == 0.1


def test_pydantic_dataclass_matches_asdict_and_plain_dict():
    trainer = PdTrainer(learning_rate=0.3, num_workers=9, device="cuda",
                        optimizer=PdOptimizer(lr=0.5, device="cuda"))
    plain = {"_target_": PdTrainer._target_, "learning_rate": 0.3, "optimizer": {"lr": 0.5}}
    row = materialize({"t": trainer})
    assert row == materialize({"t": plain})
    assert row == materialize(trainer)  # the versioned instance as the root node
    assert row["PdTrainer.optimizer"] == '{"lr": 0.5}'  # the leaf's execution field dropped
    # `asdict` keeps the leaf's excluded `device`; only the leaf differs.
    as_dict = materialize({"t": asdict(trainer)})
    assert as_dict["PdTrainer.optimizer"] == '{"device": "cuda", "lr": 0.5}'
    assert {k: v for k, v in as_dict.items() if k != "PdTrainer.optimizer"} == {
        k: v for k, v in row.items() if k != "PdTrainer.optimizer"}


def test_pydantic_dataclass_hashes_validated_values():
    # As for models: validation coerces `1` to `1.0`, which encodes differently from a dict's `1`.
    assert materialize(PdTrainer(learning_rate=1))["PdTrainer.learning_rate"] == "1.0"
    plain = {"t": {"_target_": PdTrainer._target_, "learning_rate": 1}}
    assert materialize(plain)["PdTrainer.learning_rate"] == "1"


def test_pydantic_dataclass_target_mismatch_raises():
    with pytest.raises(ValueError, match="always stands for its own class"):
        materialize({"t": PdTrainer(_target_="other.Cls")})  # type: ignore[call-arg]
