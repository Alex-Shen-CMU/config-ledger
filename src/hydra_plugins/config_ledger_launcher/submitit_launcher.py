"""Concrete launcher classes, placed here because Hydra only instantiates launchers whose
module lives under the top-level ``hydra_plugins`` namespace. All logic is in
:mod:`config_ledger.hydra.launcher`."""

from config_ledger.hydra.launcher import LocalSubmitOnlyBase, SlurmSubmitOnlyBase


class SlurmSubmitOnlyLauncher(SlurmSubmitOnlyBase):
    pass


class LocalSubmitOnlyLauncher(LocalSubmitOnlyBase):
    pass
