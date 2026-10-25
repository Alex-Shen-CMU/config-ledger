"""Hydra integration (install with ``config-ledger[hydra]``).

* :func:`ledger_from_config` -- build a :class:`~config_ledger.Ledger` from a top-level
  ``ledger:`` block in the composed config. That block is a loose top-level key, so it is
  invisible to run identity by construction.
* :mod:`config_ledger.hydra.sweeps` -- per-sweep manifest CSVs: written once by the launcher
  at submit time, read back to compile results. A sweep is both a launcher and a query.
* :mod:`config_ledger.hydra.launcher` -- ``SlurmSubmitOnlyLauncher`` / ``LocalSubmitOnlyLauncher``,
  submitit-based launchers that write the sweep manifest and (for Slurm) return immediately
  so stages can be chained with ``--dependency`` instead of a polling driver.
* :mod:`config_ledger.hydra.run_status` -- :class:`SweepRunStatus`, a run status registered by
  namespace, so the launchers can skip grid points that need no launch (``skip_finished``).
* :mod:`config_ledger.hydra.dump` -- write ``ConfigStore`` nodes to a tree of static YAML that
  Hydra loads back to the same config (``config-ledger dump-configs``).

Config convention (all top-level, all execution-control)::

    ledger:
      root: ${hydra:runtime.cwd}/.ledger   # sidecars + sweeps live here
      namespace: train                     # one per producing script/stage
      sweep_group: sweep                   # config group whose choice names the sweep
      parent_sweep: null                   # optional: file a scoped submission under this
                                           # existing sweep (usually `+ledger.parent_sweep=`)
      outputs: ${hydra:runtime.cwd}/out/train  # optional: parent dir of this namespace's output
                                               # folders; needed by hydra.launcher.skip_finished
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from config_ledger.store import Ledger

LEDGER_KEY = "ledger"


def ledger_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    block = config.get(LEDGER_KEY)
    if block is None or "root" not in block or "namespace" not in block:
        raise KeyError(
            f"config needs a top-level `{LEDGER_KEY}:` block with at least `root` and `namespace` "
            "(see config_ledger.hydra docstring)"
        )
    return {
        "root": str(block["root"]),
        "namespace": str(block["namespace"]),
        "sweep_group": str(block.get("sweep_group", "sweep")),
        "parent_sweep": str(block["parent_sweep"]) if block.get("parent_sweep") else None,
        "outputs": None if block.get("outputs") is None else str(block["outputs"]),
    }


def ledger_from_config(config: Mapping[str, Any]) -> tuple[Ledger, str]:
    """``(Ledger, namespace)`` from the composed config's ``ledger:`` block."""
    settings = ledger_settings(config)
    return Ledger(settings["root"]), settings["namespace"]
