"""``config-ledger lookup --config-name``: look up Hydra configs and sweeps.

Composes the config the way a run (or, with ``multirun``, the launcher) composes it, from a
config directory, a config name and command line overrides, then looks each config up with
:func:`config_ledger.lookup.lookup`. The ledger root and namespace default to the config's
``ledger:`` block. Read-only, like the core lookup.

A multirun expands to one grid point per job, as Hydra's ``BasicSweeper`` does
(:func:`~config_ledger.hydra.sweeps.expand_grid`): ``hydra.sweeper.params`` (e.g. a
``sweep=...`` group choice) plus sweep-style overrides. Each job is composed by the same
:func:`~config_ledger.hydra.sweeps.job_composer` the launcher uses.

Apps that register schema configs into ``ConfigStore`` from Python must run that code first:
``setup`` takes ``MODULE`` (imported) or ``MODULE:FUNC`` (imported, then ``FUNC()`` called).

Status (2026-10-18): introduced.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode

from config_ledger.hydra import ledger_settings
from config_ledger.hydra.sweeps import expand_grid, job_composer
from config_ledger.lookup import LookupResult, lookup
from config_ledger.store import Ledger


@dataclass(frozen=True)
class GridPoint:
    """One composed config's overrides and lookup result."""

    overrides: list[str]
    result: LookupResult


def run_setup(specs: Sequence[str]) -> None:
    """Import each ``MODULE``, calling ``FUNC`` for ``MODULE:FUNC``."""
    for spec in specs:
        module_name, _, func = spec.partition(":")
        module = importlib.import_module(module_name)
        if func:
            getattr(module, func)()


def lookup_hydra(
    config_dir: str | os.PathLike[str],
    config_name: str,
    overrides: Sequence[str] = (),
    multirun: bool = False,
    ledger_root: str | os.PathLike[str] | None = None,
    namespace: str | None = None,
) -> list[GridPoint]:
    """One :class:`GridPoint` for a single run, one per job for a multirun."""
    hydra_config = HydraConfig.instance()
    previous = hydra_config.cfg
    with initialize_config_dir(config_dir=str(Path(config_dir).resolve()), version_base=None):
        loader = GlobalHydra.instance().config_loader()
        mode = RunMode.MULTIRUN if multirun else RunMode.RUN
        master = loader.load_configuration(config_name, list(overrides), run_mode=mode)
        # `${hydra:...}` (e.g. runtime.cwd in the ledger block) needs HydraConfig, as in the
        # launcher's driver; restored afterwards, as `initialize_config_dir` restores GlobalHydra.
        hydra_config.set_config(master)
        try:
            settings = ledger_settings(master) if ledger_root is None or namespace is None else {}
            ledger = Ledger(ledger_root if ledger_root is not None else settings["root"])
            ns = namespace if namespace is not None else settings["namespace"]
            if not multirun:
                return [GridPoint(list(overrides), lookup(ledger, ns, master))]
            compose = job_composer(loader, master)
            return [GridPoint(job, lookup(ledger, ns, compose(job)))
                    for job in expand_grid(loader, master, overrides)]
        finally:
            hydra_config.cfg = previous
