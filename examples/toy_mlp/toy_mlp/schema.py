"""Config schema for the toy pipeline.

Everything that decides *what* a run computes is a field on a ``VersionedConfig`` class.
Everything that only decides *how* it executes (checkpoint cadence, thread count) is an
``execution_field`` and never enters identity. Loose top-level keys in the yaml
(``overwrite``, ``ledger``, output paths) are invisible to identity by construction.

``Model`` shows a version history: when a class's meaning changes, bump ``VERSION`` and teach
``upgrade_config`` how to carry old configs forward, so ``config-ledger upgrade`` can keep the
old results instead of rerunning them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from hydra.core.config_store import ConfigStore

from config_ledger import (
    UpgradeImpossible,
    VersionedConfig,
    execution_field,
    parse_chain,
    version_chain,
)


@dataclass
class Data(VersionedConfig):
    VERSION: ClassVar[int] = 1
    n_samples: int = 512
    n_features: int = 4
    noise: float = 0.1
    seed: int = 0


@dataclass
class Model(VersionedConfig):
    # 2: added `activation`; every version-1 model was tanh.
    VERSION: ClassVar[int] = 2
    hidden: int = 16
    activation: str = "tanh"

    @classmethod
    def upgrade_config(cls, old: dict) -> dict | None:
        if old["config_version"] == version_chain(cls):
            return None
        if parse_chain(old["config_version"])["Model"] == 1:
            return {**old, "activation": "tanh"}
        raise UpgradeImpossible(f"unknown Model version {old['config_version']!r}")


@dataclass
class Trainer(VersionedConfig):
    VERSION: ClassVar[int] = 1
    epochs: int = 200
    learning_rate: float = 0.05
    seed: int = 0
    # Execution control: changing how often we checkpoint does not change the result.
    checkpoint_every: int = execution_field(default=50)


@dataclass
class Evaluation(VersionedConfig):
    VERSION: ClassVar[int] = 1
    n_test: int = 256
    seed: int = 1234


def register() -> None:
    cs = ConfigStore.instance()
    cs.store(group="data", name="base", node=Data)
    cs.store(group="model", name="base", node=Model)
    cs.store(group="trainer", name="base", node=Trainer)
    cs.store(group="evaluation", name="base", node=Evaluation)
