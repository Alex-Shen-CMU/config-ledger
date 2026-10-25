"""Skip grid points whose run status says "don't launch".

Declare a :class:`~config_ledger.run_status.RunStatus` for a namespace by subclassing
:class:`SweepRunStatus` with a ``namespace`` keyword, typically at the top of the script that
does the run::

    class TrainStatus(SweepRunStatus, namespace="train"):
        @classmethod
        def check_run_status(cls, config, output_folder): ...

and opt in with the launcher param ``hydra.launcher.skip_finished=true``. ``python train.py -m``
runs the script's top level before the launcher starts, so the class is registered by the time
the driver decides what to skip. The registry is keyed by the namespace string
(``ledger.namespace``); namespaces themselves stay plain strings.

The driver does not know where a grid point's outputs live (only the script knows the parent
directory), so skipping needs ``ledger.outputs``: the directory the namespace's output folders
are created in, i.e. the ``parent_dir`` the script passes to
:meth:`~config_ledger.store.Ledger.resolve_output_folder`. It is read from each job's own
composed config. A grid point is launched without calling the check when its identity is not
in the ledger or its output folder does not exist; otherwise the check decides.

Classes defined in ``__main__`` reach submitit tasks pickled by value, which rebuilds them
without the ``namespace`` keyword, so a task never registers anything. Re-registering the same
module and qualified name (a re-import) replaces the entry; two different classes claiming one
namespace is an error.

Status (2026-10-18): introduced.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

from config_ledger.hydra import ledger_settings
from config_ledger.run_status import RunDecision, RunStatus
from config_ledger.store import Ledger

log = logging.getLogger(__name__)

_REGISTRY: dict[str, type[SweepRunStatus]] = {}


class SweepRunStatus(RunStatus):
    """A :class:`~config_ledger.run_status.RunStatus` the launchers find by namespace."""

    namespace: ClassVar[str | None] = None

    def __init_subclass__(cls, namespace: str | None = None, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if namespace is None:
            return
        if not isinstance(namespace, str) or not namespace:
            raise TypeError(f"{cls.__qualname__}: namespace must be a non-empty string, "
                            f"not {namespace!r}")
        existing = _REGISTRY.get(namespace)
        if existing is not None and _qualified(existing) != _qualified(cls):
            raise TypeError(f"namespace {namespace!r} already has a run status, "
                            f"{_qualified(existing)}; cannot also register {_qualified(cls)}")
        cls.namespace = namespace
        _REGISTRY[namespace] = cls


def _qualified(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def registered_run_status(namespace: str) -> type[SweepRunStatus] | None:
    """The class registered for ``namespace``, or None."""
    return _REGISTRY.get(namespace)


def run_status_for(namespace: str) -> type[SweepRunStatus]:
    """The class registered for ``namespace``; raises if none is, or it is still abstract."""
    status = registered_run_status(namespace)
    if status is None:
        raise LookupError(
            f"skip_finished is set but no run status is registered for namespace {namespace!r}; "
            f"declare `class ...(SweepRunStatus, namespace={namespace!r})` in the script, "
            "or drop hydra.launcher.skip_finished"
        )
    if inspect.isabstract(status):
        raise TypeError(f"{_qualified(status)} does not override check_run_status")
    return status


def plan_launches(
    ledger: Ledger,
    namespace: str,
    job_overrides: Sequence[Sequence[str]],
    compose_job_config: Callable[[list[str]], Mapping[str, Any]],
    status: type[RunStatus],
) -> list[RunDecision]:
    """One :class:`~config_ledger.run_status.RunDecision` per grid point, in grid order.

    Read-only: identities are looked up, not claimed. Raises if a job config has no
    ``ledger.outputs`` or cannot be composed, so a requested skip never silently degrades
    into launching everything.
    """
    decisions = []
    for overrides in job_overrides:
        job_config = compose_job_config(list(overrides))
        outputs = ledger_settings(job_config)["outputs"]
        if outputs is None:
            raise KeyError("skip_finished needs `ledger.outputs`: the directory this namespace's "
                           "output folders are created in (see config_ledger.hydra.run_status)")
        entry = ledger.find(namespace, ledger.identity(job_config))
        if entry is None:
            decisions.append(RunDecision(True, "not in the ledger yet"))
            continue
        folder = Path(outputs) / entry.folder_id
        if not folder.is_dir():
            decisions.append(RunDecision(True, f"no output folder yet ({folder})"))
            continue
        decisions.append(RunDecision(*status.check_run_status(job_config, folder)))
    return decisions
