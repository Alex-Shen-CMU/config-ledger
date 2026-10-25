"""Shared fixtures: a small versioned schema and a config tree over it.

Kept dependency-light (omegaconf + xxhash only). Configs are built as DictConfig, the
Hydra path; `test_adapters.py` checks plain dicts give the same rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import omegaconf
import pytest

from config_ledger import (
    Ledger,
    UpgradeImpossible,
    VersionedConfig,
    execution_field,
    parse_chain,
    version_chain,
)
from config_ledger.identity import VERSION_KEY


@dataclass
class Trainer(VersionedConfig):
    VERSION: ClassVar[int] = 1
    learning_rate: float = 0.1
    seed: int = 0
    layers: list = field(default_factory=lambda: [8, 8])
    device: str = execution_field(default="cpu")


@dataclass
class MiningGoldTrainer(Trainer):
    VERSION: ClassVar[int] = 2
    alpha: float = 0.5


@dataclass
class Method(VersionedConfig):
    VERSION: ClassVar[int] = 1
    IDENTITY_EXCLUDE: ClassVar = ("num_workers",)
    name: str = "fixture"
    trainer: Trainer = field(default_factory=Trainer)
    num_workers: int = 4


def make_config(trainer_target: str = Trainer._target_, **fields):
    trainer = {"_target_": trainer_target, "learning_rate": 0.1, "seed": 0, "layers": [8, 8], "device": "cpu"}
    trainer.update({k: v for k, v in fields.items() if k in ("learning_rate", "seed", "layers", "device", "alpha")})
    top = {k: v for k, v in fields.items() if k not in trainer}
    payload = {"method": {"_target_": Method._target_, "name": "fixture", "trainer": trainer, "num_workers": 4}}
    payload.update(top)
    return omegaconf.OmegaConf.create(payload)


@pytest.fixture
def ledger(tmp_path):
    return Ledger(tmp_path / "ledger")


def _lr_times_100(cls, old):
    if old[VERSION_KEY] == version_chain(cls):
        return None
    if parse_chain(old[VERSION_KEY])["Trainer"] == 1:
        return {**old, "learning_rate": old["learning_rate"] * 100}
    raise UpgradeImpossible("unknown")


@pytest.fixture
def bumped(monkeypatch):
    """Trainer goes to version 2 and knows how to upgrade from 1."""

    def bump(upgradable=True):
        if upgradable:
            monkeypatch.setattr(Trainer, "upgrade_config", classmethod(_lr_times_100))
        monkeypatch.setattr(Trainer, "VERSION", 2)

    return bump
