"""Declared run status: should this run be launched, and why.

A sidecar means "this identity owns this folder", never "the run finished"; whether a run is
done is judged from the artifacts it writes. :class:`RunStatus` gives that judgement one home
instead of every stage re-implementing it::

    class TrainStatus(RunStatus):
        @classmethod
        def check_run_status(cls, config, output_folder):
            if (Path(output_folder) / "summary.json").exists():
                return RunDecision(False, "summary.json exists")
            return RunDecision(True, "no summary.json yet")

The answer is "launch or not", not "complete or not", so a stage can say "partially done,
launch to resume". Namespaces stay plain strings and know nothing about this class; using it
is up to the caller. A stage typically calls it itself (``if not decision.launch: return``),
and the Hydra launchers can call it before submitting to skip grid points
(:class:`config_ledger.hydra.run_status.SweepRunStatus`).

The method is a classmethod and the classes are never instantiated, so ``@abstractmethod``
is not enforced at class creation; the base raises ``NotImplementedError`` when called, and
the Hydra registry refuses a class that does not override it.

Status (2026-10-18): introduced.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, NamedTuple

__all__ = ["RunDecision", "RunStatus"]


class RunDecision(NamedTuple):
    """Whether to launch a run, and a human-readable reason (logged by the launchers)."""

    launch: bool
    reason: str


class RunStatus(ABC):
    """Judges, from a run's config and its output folder's artifacts, whether to launch it."""

    @classmethod
    @abstractmethod
    def check_run_status(cls, config: Any, output_folder: str | os.PathLike[str]) -> RunDecision:
        """Decide whether the run with ``config``, owning ``output_folder``, should be launched.

        Judge from the artifacts in ``output_folder``, never from the ledger's sidecar or run
        manifest existing: those are written when a run starts, not when it finishes.
        """
        raise NotImplementedError(f"{cls.__qualname__} must override check_run_status")
