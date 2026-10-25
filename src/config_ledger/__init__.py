"""config-ledger: content-addressed, versioned run identity for experiment configs.

Public API (see each module's docstring for the reasoning):

* :class:`VersionedConfig`, :func:`execution_field` -- declare identity on config classes;
  ``upgrade_config`` / :class:`UpgradeImpossible` carry old configs to a new version.
* :func:`compute_identity`, :func:`materialize` -- config -> tree, row, hash (every value).
* :class:`Ledger` -- hash -> output folder, NFS-safe. :mod:`config_ledger.lookup` -- read-only
  config -> folder with version diagnostics (``config-ledger lookup``).
* :mod:`config_ledger.versions` -- the per-namespace version index and stale-run discovery
  (:class:`StaleRunsError`).
* :mod:`config_ledger.upgrade` / :mod:`config_ledger.maintenance` -- upgrade, mark stale,
  purge, rebuild; driven by the ``config-ledger`` CLI.
* :class:`RunStatus`, :class:`RunDecision` -- declare, from a run's artifacts, whether it should
  be launched; the Hydra launchers can use it to skip finished grid points.
* :mod:`config_ledger.adapters` -- how config objects are read: plain dataclass instances
  (core, stdlib only), OmegaConf ``DictConfig`` (extra ``config-ledger[omegaconf]``), pydantic
  models (extra ``config-ledger[pydantic]``); the core itself is plain-dict.
* :mod:`config_ledger.hydra` (extra ``config-ledger[hydra]``) -- sweep manifests and the
  submit-only Slurm launcher.
"""

from config_ledger.identity import (
    Identity,
    compute_identity,
    hash_row,
    materialize,
    materialize_tree,
)
from config_ledger.run_status import RunDecision, RunStatus
from config_ledger.store import Entry, Ledger
from config_ledger.upgrade import upgrade_tree
from config_ledger.versioning import (
    UpgradeImpossible,
    VersionedConfig,
    execution_field,
    parse_chain,
    version_chain,
)
from config_ledger.versions import (
    NewerVersionError,
    SchemaChangedError,
    StaleRunsError,
    skip_version_check,
)

__all__ = [
    "Entry",
    "Identity",
    "Ledger",
    "NewerVersionError",
    "RunDecision",
    "RunStatus",
    "SchemaChangedError",
    "StaleRunsError",
    "UpgradeImpossible",
    "VersionedConfig",
    "compute_identity",
    "execution_field",
    "hash_row",
    "materialize",
    "materialize_tree",
    "parse_chain",
    "skip_version_check",
    "upgrade_tree",
    "version_chain",
]

__version__ = "0.1.0"
