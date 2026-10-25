"""Config schema of the test app. ``sleep_s`` and ``fail_on_value`` only decide *how* a task
runs, so they are execution fields: grid points differing only in them share one identity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from hydra.core.config_store import ConfigStore

from config_ledger import VersionedConfig, execution_field


@dataclass
class Work(VersionedConfig):
    VERSION: ClassVar[int] = 1
    value: int = 0
    sleep_s: float = execution_field(default=0.0)
    # The task whose `value` equals this raises after sleeping.
    fail_on_value: int = execution_field(default=-1)


@dataclass
class Check(VersionedConfig):
    VERSION: ClassVar[int] = 1
    scale: int = 2


@dataclass
class RaceNode(VersionedConfig):
    """Root config of the claim race (no Hydra involved)."""

    VERSION: ClassVar[int] = 1
    index: int = 0


def register() -> None:
    cs = ConfigStore.instance()
    cs.store(group="work", name="base", node=Work)
    cs.store(group="check", name="base", node=Check)
