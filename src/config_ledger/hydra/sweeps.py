"""Per-sweep manifest CSVs: the sweep as launcher *and* query.

Every multirun writes one timestamped CSV::

    <ledger root>/sweeps/<namespace>/<sweep name>/<timestamp>.csv

with a column per ``hydra.sweeper.params`` key (swept axes and constant singletons alike)
plus the ``folder_id`` each grid point's results live under. With the launcher's
``overrides_to_manifest`` option, every other override in the job overrides (command-line
sweeps such as ``data_generator.seed=0,1,2``, constants and config group choices) gets a column
too, after the sweeper params in first-seen order. Without it, a command-line sweep records
only ``folder_id``. Compiling a sweep's results
is then "read one CSV and look up folders", never "recompose every combo's config".

The CSV is written by the *driver* at submission time, in one shot: it is the only process
that sees the whole materialized grid, it avoids one-timestamp-per-task clutter, and it
sidesteps concurrent-append contention on NFS. Folder ids are resolved eagerly, so a grid
point that later fails still has a row. A row says "this is where these results would
live", not "these results exist"; completion is judged from artifacts.

A big sweep is often run piecewise: one scoped sweeper config per seed or architecture, or a
temporary backfill config. Those rows belong to the *parent* sweep, so:

* the launcher's ``manifest_only`` mode writes the parent's full-grid CSV and submits nothing;
* each scoped submission sets ``ledger.parent_sweep=<parent>``. It then writes no CSV of its
  own, and :func:`check_parent_sweep` aborts it unless the parent already has a manifest
  that lists every one of its folder ids.

Reader side: :func:`load_sweep` concatenates every CSV for a sweep (a resubmitted sweep
re-registers the same folder ids, so duplicates are expected and dropped) and
:class:`SweepTable` turns it into a grid: pin constant columns with filters, name the axes,
and get ``(axis values...) -> folder_id``. Any column that still varies and is neither an
axis nor pinned is an error, not a silent collapse.

Status (2026-09-10): generic half of an internal project's sweep manifest writer; the
project-specific grid logic was deliberately not ported. 2026-10-17:
:func:`check_sweep_versions`.
2026-10-18: :func:`check_parent_sweep`; microsecond, exclusive-create CSV names.
2026-10-26: ``include_overrides`` (launcher option ``overrides_to_manifest``).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config_ledger.store import Ledger

log = logging.getLogger(__name__)

FOLDER_ID_COL = "folder_id"
UNNAMED_SWEEP = "unnamed_sweep"


def sweeps_dir(ledger: Ledger) -> Path:
    return ledger.root / "sweeps"


def sweep_dir(ledger: Ledger, namespace: str, sweep_name: str) -> Path:
    return sweeps_dir(ledger) / namespace / sweep_name


def sweep_name(config: Mapping[str, Any], sweep_group: str = "sweep") -> str:
    """The chosen option of the ``sweep_group`` config group, else ``sweep_name`` if the
    config carries one, else ``"unnamed_sweep"``."""
    try:
        choices = config["hydra"]["runtime"]["choices"]
        if choices.get(sweep_group):
            return str(choices[sweep_group])
    except (KeyError, TypeError):
        pass
    return str(config.get("sweep_name", UNNAMED_SWEEP))


def overrides_to_dict(overrides: Iterable[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for override in overrides:
        key, separator, value = override.partition("=")
        if separator:
            parsed[key.lstrip("+~")] = value
    return parsed


def build_rows(
    ledger: Ledger,
    namespace: str,
    config: Mapping[str, Any],
    job_overrides: Sequence[Sequence[str]],
    compose_job_config: Callable[[list[str]], Mapping[str, Any]],
    include_overrides: bool = False,
) -> list[dict[str, str]]:
    """One row per grid point: sweeper param values plus the resolved folder id.

    ``compose_job_config(overrides)`` must return the config exactly as the task itself will
    see it, so the identity claimed here is the one the task will look up.

    With ``include_overrides``, every override key in any job's overrides that is not a
    sweeper param also gets a column, in first-seen order; a job without that key gets ``""``.
    """
    try:
        sweeper_params = dict(config["hydra"]["sweeper"]["params"] or {})
    except (KeyError, TypeError):
        sweeper_params = {}
    param_keys = [key.lstrip("+~") for key in sweeper_params]
    applied_per_job = [overrides_to_dict(overrides) for overrides in job_overrides]
    extra_keys = []
    if include_overrides:
        known = set(param_keys)
        extra_keys = [key for key in dict.fromkeys(k for applied in applied_per_job for k in applied)
                      if key not in known]
    rows = []
    for overrides, applied in zip(job_overrides, applied_per_job, strict=True):
        row = {
            key: applied.get(key, str(sweeper_params[raw]).strip())
            for raw, key in zip(sweeper_params, param_keys)
        }
        row.update({key: applied.get(key, "") for key in extra_keys})
        job_config = compose_job_config(list(overrides))
        row[FOLDER_ID_COL] = ledger.get_or_create(namespace, ledger.identity(job_config)).folder_id
        rows.append(row)
    return rows


def job_composer(config_loader: Any, master_config: Any) -> Callable[[Sequence[str]], Any]:
    """``overrides -> job config``, composed as Hydra composes each multirun job and cached, so
    the launcher's check and manifest (and ``config-ledger lookup -m``) see one config per job."""
    composed: dict[tuple[str, ...], Any] = {}

    def compose(overrides: Sequence[str]) -> Any:
        key = tuple(overrides)
        if key not in composed:
            composed[key] = config_loader.load_sweep_config(master_config, list(overrides))
        return composed[key]

    return compose


def expand_grid(config_loader: Any, master_config: Any,
                arguments: Sequence[str]) -> list[list[str]]:
    """The job overrides of a multirun, expanded as Hydra's ``BasicSweeper`` expands them:
    ``hydra.sweeper.params`` of the composed config, then the command line ``arguments``."""
    from hydra._internal.core_plugins.basic_sweeper import BasicSweeper
    from hydra.core.override_parser.overrides_parser import OverridesParser

    sweeper = master_config.hydra.sweeper
    if not str(sweeper.get("_target_", "")).endswith(BasicSweeper.__name__):
        raise ValueError(f"only Hydra's BasicSweeper grids can be expanded, not {sweeper._target_}")
    params = [f"{key}={value}" for key, value in (sweeper.get("params") or {}).items()]
    parsed = OverridesParser.create(config_loader=config_loader).parse_overrides(
        [*params, *arguments])
    (batch,) = BasicSweeper.split_arguments(parsed, None)
    return batch


def check_sweep_versions(
    ledger: Ledger,
    namespace: str,
    job_overrides: Sequence[Sequence[str]],
    compose_job_config: Callable[[list[str]], Mapping[str, Any]],
) -> bool:
    """Stale-run check for a whole grid at once, before anything is claimed.

    Raises :class:`~config_ledger.versions.StaleRunsError` (or another version error) to abort
    the submission. Returns True if the grid was checked, so tasks may skip their own check, or
    False if the grid could not be composed here, so tasks must check themselves.
    """
    try:
        configs = [compose_job_config(list(overrides)) for overrides in job_overrides]
    except Exception:
        log.exception("Could not compose the sweep in the driver; each task will check "
                      "versions itself.")
        return False
    ledger.check_configs(namespace, configs)
    return True


class ParentSweepManifestMissingError(RuntimeError):
    """``ledger.parent_sweep`` names a sweep with no manifest CSV yet."""


class ParentSweepMismatchError(ValueError):
    """A scoped submission has grid points whose folder ids are not in the parent's manifest."""


def check_parent_sweep(
    ledger: Ledger,
    namespace: str,
    parent: str,
    job_overrides: Sequence[Sequence[str]],
    compose_job_config: Callable[[list[str]], Mapping[str, Any]],
) -> None:
    """Check a scoped submission against its parent sweep's manifest, before anything is claimed.

    Raises :class:`ParentSweepManifestMissingError` if ``parent`` has no manifest CSV (rather
    than filing the rows under a name with no full-grid manifest; in an earlier implementation a
    silent fallback picked up a stale one), and :class:`ParentSweepMismatchError` if any grid
    point's folder id is not among the parent's. A grid point whose config resolves to
    something the parent's grid does not (e.g. an outdated seed scheme) has a different
    identity, hence a different or no folder id. The lookup is read-only, so a failed check
    claims nothing. Unlike :func:`write_sweep_manifest`, this raises on any error.
    """
    if not list_manifest_csvs(ledger, namespace, parent):
        raise ParentSweepManifestMissingError(
            f"ledger.parent_sweep={parent!r}, but {sweep_dir(ledger, namespace, parent)} has no "
            "manifest CSV. Write the parent's full-grid manifest first, by running the parent "
            "sweep with hydra.launcher.manifest_only=true."
        )
    parent_ids = set(load_sweep(ledger, namespace, parent).df[FOLDER_ID_COL])
    missing = []
    for overrides in job_overrides:
        entry = ledger.find(namespace, ledger.identity(compose_job_config(list(overrides))))
        if entry is None or entry.folder_id not in parent_ids:
            missing.append(" ".join(overrides))
    if missing:
        shown = "\n  ".join(missing[:10]) + ("\n  ..." if len(missing) > 10 else "")
        raise ParentSweepMismatchError(
            f"{len(missing)} of {len(job_overrides)} grid point(s) are not in parent sweep "
            f"{parent!r}'s manifest, so their configs differ from every grid point the parent "
            f"recorded:\n  {shown}"
        )
    log.info("All %d grid point(s) are in parent sweep %r's manifest; not writing a CSV for "
             "this scoped submission.", len(job_overrides), parent)


def write_sweep_manifest(
    ledger: Ledger,
    namespace: str,
    config: Mapping[str, Any],
    job_overrides: Sequence[Sequence[str]],
    compose_job_config: Callable[[list[str]], Mapping[str, Any]],
    sweep_group: str = "sweep",
    include_overrides: bool = False,
) -> Path | None:
    """Write the sweep CSV and return its path, or None for an empty grid.

    ``include_overrides`` is passed to :func:`build_rows`.

    Never raises: a sweep manifest is bookkeeping, and failing to write one must not take
    down an otherwise valid submission.
    """
    try:
        rows = build_rows(ledger, namespace, config, job_overrides, compose_job_config,
                          include_overrides=include_overrides)
        if not rows:
            return None
        directory = sweep_dir(ledger, namespace, sweep_name(config, sweep_group))
        directory.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(rows)
        # Microseconds plus exclusive create, so concurrent submissions of one sweep never
        # overwrite each other's CSV. Names still sort chronologically.
        while True:
            path = directory / f"{datetime.now().strftime('%Y%m%dT%H%M%S_%f')}.csv"
            try:
                with path.open("x", newline="") as f:
                    frame.to_csv(f, index=False)
                break
            except FileExistsError:
                continue
        log.info("Wrote sweep manifest for %d grid point(s) to %s", len(rows), path)
        return path
    except Exception:
        log.exception("Failed to write the sweep manifest; continuing with submission anyway.")
        return None


def list_manifest_csvs(ledger: Ledger, namespace: str, sweep: str) -> list[Path]:
    directory = sweep_dir(ledger, namespace, sweep)
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix == ".csv")


def load_sweep(ledger: Ledger, namespace: str, sweep: str,
               csv_paths: Sequence[str | os.PathLike[str]] | None = None) -> SweepTable:
    paths = [Path(p) for p in csv_paths] if csv_paths else list_manifest_csvs(ledger, namespace, sweep)
    if not paths:
        raise FileNotFoundError(
            f"No sweep manifest CSVs for namespace={namespace!r}, sweep={sweep!r} under "
            f"{sweep_dir(ledger, namespace, sweep)}. Manifests are written by the config-ledger "
            "launchers (config_ledger.hydra.launcher); a sweep run under another launcher has none."
        )
    frames = [pd.read_csv(p, dtype=str, keep_default_na=False) for p in paths]
    # CSVs written with and without ``overrides_to_manifest`` have different columns; a cell
    # missing from a CSV reads as "", like a job that lacked the override.
    df = (pd.concat(frames, ignore_index=True).fillna("")
          .drop_duplicates().reset_index(drop=True))
    return SweepTable(df)


class SweepTable:
    """A loaded sweep manifest with grid helpers. ``df`` is one row per grid point."""

    def __init__(self, df: pd.DataFrame) -> None:
        if FOLDER_ID_COL not in df.columns:
            raise ValueError(f"sweep manifest has no {FOLDER_ID_COL!r} column")
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    @property
    def param_columns(self) -> list[str]:
        return [c for c in self.df.columns if c != FOLDER_ID_COL]

    def axes(self) -> list[str]:
        """Columns that take more than one value: the sweep's varying axes, in CSV order."""
        return [c for c in self.param_columns if self.df[c].nunique() > 1]

    def pin(self, filters: Mapping[str, str]) -> SweepTable:
        """Keep only rows whose columns equal the given raw CSV string values."""
        missing = sorted(set(filters) - set(self.df.columns))
        if missing:
            raise KeyError(f"filter column(s) not in the sweep manifest: {missing}")
        df = self.df
        for col, value in filters.items():
            df = df[df[col] == value]
        if df.empty:
            raise ValueError(f"no sweep manifest rows left after applying filters: {dict(filters)}")
        return SweepTable(df.reset_index(drop=True))

    def grid(self, axes: Sequence[str]) -> dict[tuple[str, ...], str]:
        """``(axis values...) -> folder_id`` over exactly ``axes``.

        Every other column must be single-valued; otherwise two grid points would share a key
        and results would silently collapse, so it raises naming the offending columns.
        """
        missing = sorted(set(axes) - set(self.df.columns))
        if missing:
            raise KeyError(f"axis column(s) not in the sweep manifest: {missing}")
        stray = {c: sorted(self.df[c].unique()) for c in self.param_columns
                 if c not in axes and self.df[c].nunique() > 1}
        if stray:
            raise ValueError(
                f"columns vary across the manifest but are neither axes nor pinned: {stray}. "
                "Pin them with .pin({...}) to select one slice, or add them as axes."
            )
        grid: dict[tuple[str, ...], str] = {}
        for _, row in self.df.iterrows():
            key = tuple(str(row[a]) for a in axes)
            folder_id = str(row[FOLDER_ID_COL])
            if grid.get(key, folder_id) != folder_id:
                raise ValueError(f"grid point {key} maps to two folder ids ({grid[key]}, {folder_id})")
            grid[key] = folder_id
        return grid

    def values(self, axis: str) -> list[str]:
        """Distinct values of one column, in first-seen order."""
        return list(dict.fromkeys(self.df[axis].astype(str)))
