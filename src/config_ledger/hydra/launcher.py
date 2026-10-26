"""Submitit launchers that record the sweep manifest and (optionally) do not wait.

Why a launcher that returns immediately: hydra-submitit-launcher's stock ``launch()`` submits
the array and then blocks on ``job.results()`` for every task, which polls ``sacct``. On
clusters where ``sacct`` is flaky from compute nodes the poll can hang forever *after* the
array has finished, so a driver that chains stages with ``&&`` never proceeds. The Slurm
launcher here submits, prints ``CONFIG_LEDGER_ARRAY_JOB_ID=<id>`` on stdout, and returns; chain
the next stage with Slurm's own dependency mechanism::

    TRAIN=$(python train.py -m hydra/launcher=config_ledger_slurm ... | grep -oP 'ARRAY_JOB_ID=\\K\\d+')
    python evaluate.py -m hydra/launcher=config_ledger_slurm \\
        +hydra.launcher.additional_parameters.dependency=afterany:$TRAIN ...

Both launchers check for stale runs once for the whole grid and write the sweep manifest
(:mod:`config_ledger.hydra.sweeps`) before submitting, since the driver is the one process that
sees the whole grid. The check (:meth:`~config_ledger.store.Ledger.check_configs`) covers the
union of every config class any job uses; if it fails, nothing is claimed or submitted. Tasks
then skip their own per-run check (:func:`~config_ledger.versions.skip_version_check`). If the
driver could not compose the grid, the tasks check individually instead.

``__call__`` (the per-task entry) re-raises a failed task's exception. Hydra's ``run_job``
catches exceptions and stashes them on the ``JobReturn``, relying on a driver that inspects
``.return_value``; a non-waiting driver never does, so without this a task could raise,
produce nothing, and still be reported COMPLETED by Slurm.

``skip_finished: true`` asks the namespace's registered
:class:`~config_ledger.hydra.run_status.SweepRunStatus` whether each grid point should be
launched, after the manifest is written (the manifest always lists the whole grid) and before
submitting. Skips are logged with their reason; launched points keep their job index, so
``hydra.job.num`` and the sweep subdirectories do not shift. If every point is skipped nothing
is submitted and no ``CONFIG_LEDGER_ARRAY_JOB_ID`` line is printed; that is success, not an
error, so add the dependency only for a non-empty id (``grep ... || true``, then
``${TRAIN:+"+hydra.launcher.additional_parameters.dependency=afterany:$TRAIN"}``).

``manifest_only: true`` runs the stale-run check, writes the manifest for the whole
grid and returns without submitting anything (no executor is built, so no Slurm access is
needed). Use it to record a big sweep's full grid before running it piecewise: each scoped part
is then submitted with ``+ledger.parent_sweep=<parent>``, which writes no CSV of its own and
aborts unless every one of its folder ids is already in the parent's manifest
(:func:`~config_ledger.hydra.sweeps.check_parent_sweep`). The two cannot be combined.

Tasks see the script's directory on ``sys.path``, as under ``python script.py``, so a task
script can import sibling modules. Submitit workers run ``python -m submitit.core._submit``,
whose ``sys.path[0]`` is the job's cwd, and the pickled task function is unpickled (importing
the modules it references) before any code of ours runs on the worker. So the driver's
``sys.path[0]`` (the script's directory, or the cwd under ``python -m``) is prepended to
``PYTHONPATH`` while the jobs are submitted. Both executors pass on the submitting environment
(the local one spawns subprocesses, and ``sbatch`` exports it unless ``--export`` says
otherwise). The driver's own environment is restored afterwards. Set
``script_dir_to_pythonpath: false`` to opt out. Nothing is added under ``python -P``.

The local launcher defaults to ``wait_for_completion: true`` so an off-cluster example can
run a sweep and then aggregate it in the same script; the same config with
``hydra/launcher=config_ledger_slurm`` runs the real thing on a cluster.

Status (2026-09-10): ported from an internal project's submit-only Slurm launcher and
generalized to the local executor. 2026-10-17: sweep-wide stale-run check.
2026-10-18: ``skip_finished``. 2026-10-18: ``manifest_only`` and
``ledger.parent_sweep``. 2026-10-26: ``script_dir_to_pythonpath``.
"""

from __future__ import annotations

import logging
import os
import sys
import traceback
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.core.singleton import Singleton
from hydra.core.utils import JobReturn, JobStatus, filter_overrides
from hydra_plugins.hydra_submitit_launcher.config import BaseQueueConf, LocalQueueConf, SlurmQueueConf
from hydra_plugins.hydra_submitit_launcher.submitit_launcher import (
    BaseSubmititLauncher,
    LocalLauncher,
    SlurmLauncher,
)
from omegaconf import OmegaConf

from config_ledger.hydra import ledger_from_config, ledger_settings
from config_ledger.hydra.run_status import plan_launches, run_status_for
from config_ledger.hydra.sweeps import (
    check_parent_sweep,
    check_sweep_versions,
    job_composer,
    write_sweep_manifest,
)
from config_ledger.versions import skip_version_check

log = logging.getLogger(__name__)

ARRAY_JOB_ID_MARKER = "CONFIG_LEDGER_ARRAY_JOB_ID"
_LAUNCHER_ONLY_KEYS = {"wait_for_completion", "skip_finished", "manifest_only",
                       "script_dir_to_pythonpath"}


@contextmanager
def script_dir_on_pythonpath() -> Iterator[None]:
    """Prepend the driver's script directory (``sys.path[0]``) to ``PYTHONPATH`` for the
    duration, so processes spawned meanwhile import from it as the driver does."""
    if getattr(sys.flags, "safe_path", False) or not sys.path:
        yield
        return
    script_dir = os.path.abspath(sys.path[0] or os.curdir)
    old = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = os.pathsep.join([script_dir, old]) if old else script_dir
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = old


class _SubmitOnlyMixin(BaseSubmititLauncher):
    # Set by the driver before submission (and pickled with the launcher into every task) once
    # the whole grid passed the stale-run check.
    _versions_checked: bool = False

    def __call__(self, sweep_overrides: list[str], job_dir_key: str, job_num: int, job_id: str,
                 singleton_state: dict[type, Any]) -> JobReturn:
        args = (sweep_overrides, job_dir_key, job_num, job_id, singleton_state)
        if self._versions_checked:
            with skip_version_check():
                ret = super().__call__(*args)
        else:
            ret = super().__call__(*args)
        if ret.status == JobStatus.FAILED:
            exc = ret._return_value
            log.error("task_function raised:\n%s",
                      "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
            raise exc
        return ret

    def launch(self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int) -> Sequence[JobReturn]:
        import submitit

        assert self.config is not None
        assert len(job_overrides) > 0
        params = dict(self.params)
        wait = bool(params.pop("wait_for_completion", False))
        skip_finished = bool(params.pop("skip_finished", False))
        manifest_only = bool(params.pop("manifest_only", False))
        script_dir_to_pythonpath = bool(params.pop("script_dir_to_pythonpath", True))

        sweep_dir = Path(str(self.config.hydra.sweep.dir))
        sweep_dir.mkdir(parents=True, exist_ok=True)
        if "mode" in self.config.hydra.sweep:
            os.chmod(sweep_dir, mode=int(str(self.config.hydra.sweep.mode), 8))
        log.info("Submitit '%s' sweep output dir : %s", self._EXECUTOR, sweep_dir)

        job_params = []
        for idx, overrides in enumerate(job_overrides):
            idx = initial_job_idx + idx
            log.info("\t#%d : %s", idx, " ".join(filter_overrides(overrides)))
            job_params.append((list(overrides), "hydra.sweep.dir", idx, f"job_id_for_{idx}", Singleton.get_state()))

        # The driver is the one process that sees the whole grid: check it for stale runs once
        # (raising aborts the submission) and record it before submitting. Each job config is
        # composed once and shared by both. A scoped submission (ledger.parent_sweep) records
        # nothing; it is checked against the parent's manifest instead (raising aborts it).
        # HydraConfig is normally only set inside a job; set it here so `${hydra:...}`
        # interpolations in the ledger block (e.g. runtime.cwd) resolve in the driver too.
        HydraConfig.instance().set_config(self.config)
        settings = ledger_settings(self.config)
        if manifest_only and settings["parent_sweep"]:
            raise ValueError(
                "hydra.launcher.manifest_only writes a sweep's own full-grid manifest, and "
                f"ledger.parent_sweep={settings['parent_sweep']!r} files a scoped submission under "
                "an existing one; set one or the other."
            )
        ledger, namespace = ledger_from_config(self.config)
        compose = job_composer(self.hydra_context.config_loader, self.config)
        self._versions_checked = check_sweep_versions(ledger, namespace, job_overrides, compose)
        path = None
        if settings["parent_sweep"]:
            check_parent_sweep(ledger, namespace, settings["parent_sweep"], job_overrides, compose)
        else:
            path = write_sweep_manifest(
                ledger, namespace, self.config, job_overrides, compose,
                sweep_group=settings["sweep_group"],
            )

        # A grid point that is not submitted (every point under manifest_only, skipped points
        # otherwise) comes back as COMPLETED with no return value, so there is one JobReturn per
        # grid point, in grid order.
        returns = [JobReturn(overrides=list(overrides), status=JobStatus.COMPLETED)
                   for overrides in job_overrides]
        if manifest_only:
            # Recording the grid is this mode's whole job, so a failure to write is an error here.
            if path is None:
                raise RuntimeError("manifest_only: failed to write the sweep manifest (see the "
                                   "log above); nothing was submitted.")
            log.info("Manifest-only: recorded %d grid point(s) in %s; submitted nothing.",
                     len(job_params), path)
            return returns

        launch = [True] * len(job_params)
        if skip_finished:
            # Raises (aborting the submission) rather than launching everything if it cannot decide.
            decisions = plan_launches(ledger, namespace, job_overrides, compose,
                                      run_status_for(namespace))
            for (_, _, idx, _, _), decision in zip(job_params, decisions, strict=True):
                log.info("\t#%d : %s (%s)", idx, "launch" if decision.launch else "skip",
                         decision.reason)
            launch = [decision.launch for decision in decisions]
            log.info("Skipped %d of %d grid point(s).", launch.count(False), len(launch))
        submitted = [i for i, launched in enumerate(launch) if launched]
        if not submitted:
            log.info("All %d grid point(s) skipped; nothing submitted.", len(job_params))
            return returns

        init_params = {"folder": params["submitit_folder"]}
        specific_init_keys = {"max_num_timeout"}
        init_params.update({f"{self._EXECUTOR}_{k}": v for k, v in params.items() if k in specific_init_keys})
        init_keys = specific_init_keys | {"submitit_folder"}
        executor = submitit.AutoExecutor(cluster=self._EXECUTOR, **init_params)
        baseparams = set(OmegaConf.structured(BaseQueueConf).keys())
        executor.update_parameters(**{
            k if k in baseparams else f"{self._EXECUTOR}_{k}": v
            for k, v in params.items() if k not in init_keys
        })
        with script_dir_on_pythonpath() if script_dir_to_pythonpath else nullcontext():
            jobs = executor.map_array(self, *zip(*(job_params[i] for i in submitted), strict=True))
        array_job_id = jobs[0].job_id.split("_")[0]
        print(f"{ARRAY_JOB_ID_MARKER}={array_job_id}", flush=True)
        if wait:
            log.info("Submitted %d job(s) as %s; waiting for completion.", len(jobs), array_job_id)
            for i, job in zip(submitted, jobs, strict=True):
                returns[i] = job.results()[0]
            return returns
        log.info("Submitted %d job(s) as %s; not waiting (submit-only launcher).", len(jobs), array_job_id)
        return returns


# Hydra refuses to instantiate a launcher whose class module is not under the top-level
# `hydra_plugins` namespace, so the concrete classes live in
# `hydra_plugins.config_ledger_launcher.submitit_launcher` and only the logic lives here.
PLUGIN_MODULE = "hydra_plugins.config_ledger_launcher.submitit_launcher"


class SlurmSubmitOnlyBase(_SubmitOnlyMixin, SlurmLauncher):
    """Slurm array via submitit; returns right after submission unless ``wait_for_completion``."""


class LocalSubmitOnlyBase(_SubmitOnlyMixin, LocalLauncher):
    """Local subprocesses via submitit; waits by default so a script can aggregate afterwards."""


@dataclass
class SlurmSubmitOnlyConf(SlurmQueueConf):
    _target_: str = f"{PLUGIN_MODULE}.SlurmSubmitOnlyLauncher"
    wait_for_completion: bool = False
    skip_finished: bool = False
    manifest_only: bool = False
    script_dir_to_pythonpath: bool = True


@dataclass
class LocalSubmitOnlyConf(LocalQueueConf):
    _target_: str = f"{PLUGIN_MODULE}.LocalSubmitOnlyLauncher"
    wait_for_completion: bool = True
    skip_finished: bool = False
    manifest_only: bool = False
    script_dir_to_pythonpath: bool = True


ConfigStore.instance().store(group="hydra/launcher", name="config_ledger_slurm", node=SlurmSubmitOnlyConf,
                             provider="config-ledger")
ConfigStore.instance().store(group="hydra/launcher", name="config_ledger_local", node=LocalSubmitOnlyConf,
                             provider="config-ledger")
