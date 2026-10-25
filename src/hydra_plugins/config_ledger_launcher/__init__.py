"""Hydra plugin discovery shim.

Hydra only discovers (and only instantiates) launcher plugins that live inside a top-level
``hydra_plugins`` namespace package. Importing this package registers the
``hydra/launcher=config_ledger_slurm`` and ``hydra/launcher=config_ledger_local`` options.
"""

from hydra_plugins.config_ledger_launcher.submitit_launcher import (  # noqa: F401
    LocalSubmitOnlyLauncher,
    SlurmSubmitOnlyLauncher,
)

__all__ = ["LocalSubmitOnlyLauncher", "SlurmSubmitOnlyLauncher"]
