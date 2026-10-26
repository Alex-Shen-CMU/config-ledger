# config-ledger

Content-addressed, versioned run identity for experiment configs on shared filesystems and
Slurm clusters. No database, no daemon.

`config-ledger` answers one question for a pipeline that runs thousands of overlapping,
concurrent, preemptible jobs on an NFS-backed cluster: **"has this exact configuration
already been run, and if so, where are its results?"** It does so with:

- **Declared identity.** Fields on a `VersionedConfig` dataclass are identity; fields marked
  `execution_field` (devices, cache dirs, checkpoint cadence) and loose top-level config keys
  (`overwrite`, output paths) are not. Nothing is inferred from source code.
- **Declared, inheritance-propagated versions.** Every config class carries a `VERSION`;
  the hash records the whole chain (`Trainer:1|MiningGoldTrainer:2`), so bumping a base class
  invalidates its subclasses. A rename or a comment never invalidates anything.
- **Every value is hashed; nothing goes stale silently.** Omitted fields are filled from
  schema defaults and unknown keys are rejected. Each namespace indexes its runs by every
  `(class, version)` they use, so after a `VERSION` bump new runs are refused until the old
  ones are dealt with: upgraded in place through the class's `upgrade_config`
  (`config-ledger upgrade`; a folder can be reachable by several hashes), or marked stale and
  purged. No full scan of old results is needed.
- **NFS-safe, lock-free allocation.** One JSON sidecar per identity, claimed with an atomic
  `os.link()`. Concurrent Slurm array tasks that race on the same config resolve to one
  folder; the loser adopts the winner's id. Nothing to lock, nothing to corrupt.
- **Sweeps as launcher and query.** The Hydra launcher writes one CSV per multirun mapping
  every grid point to its folder id, so compiling a sweep is reading one file. A completed
  sidecar never means a completed run: completion is judged from the artifacts.

## Install

```bash
pip install -e .               # core: xxhash only; configs are plain dicts or dataclasses
pip install -e ".[omegaconf]"  # + omegaconf, to pass a DictConfig directly
pip install -e ".[pydantic]"   # + pydantic v2, to declare configs as pydantic models
pip install -e ".[hydra]"      # + omegaconf, hydra-core, hydra-submitit-launcher, submitit, pandas
pip install -e ".[dev]"     # + example deps, pytest, ruff, pyright
```

Python 3.10+.

## Sixty-second tour

```python
from dataclasses import dataclass, field
from typing import ClassVar
from config_ledger import VersionedConfig, execution_field, Ledger

@dataclass
class Trainer(VersionedConfig):
    VERSION: ClassVar[int] = 1
    learning_rate: float = 0.05
    seed: int = 0
    device: str = execution_field(default="cpu")   # never enters identity

cfg = {   # a plain dict; an OmegaConf DictConfig (e.g. from @hydra.main) works the same
    "trainer": {"_target_": Trainer._target_, "learning_rate": 0.1, "seed": 0, "device": "cuda"},
    "overwrite": True,                              # loose top-level key: never enters identity
}

ledger = Ledger("/shared/project/.ledger")
folder, entry = ledger.resolve_output_folder("train", "/shared/project/runs", cfg)
# folder == /shared/project/runs/<uuid4>; the same cfg from any node, any time, resolves here.
```

A loose top-level value that should be identity (a dataset selector, say) goes in a
`VersionedConfig` of its own. There is deliberately no allow-list for loose keys: the wrapper
gives the value a schema default, a `VERSION` to bump and an `upgrade_config`, which a loose
key never has. With Hydra, the override becomes `data.name=cifar10` instead of `data=cifar10`.

```python
@dataclass
class Data(VersionedConfig):
    VERSION: ClassVar[int] = 1
    name: str = "mnist"

cfg["data"] = {"_target_": Data._target_, "name": "imagenet"}   # now part of identity
```

Without any config framework installed, dataclass instances can be passed directly too:
a versioned instance alone, or a plain dataclass whose versioned fields are the identity
roots (its other fields are execution control, like loose top-level dict keys):

```python
@dataclass
class Config:
    trainer: Trainer = field(default_factory=Trainer)
    output_dir: str = "/shared/project/runs"         # not identity

cfg = Config(trainer=Trainer(learning_rate=0.1, device="cuda"))
folder, entry = ledger.resolve_output_folder("train", cfg.output_dir, cfg)  # same folder as above
```

Pydantic models mix in `VersionedConfig` the same way and can be passed directly, alone or
inside a dict (exca's `_exclude_from_cls_uid` is honoured too):

```python
from pydantic import BaseModel

class Trainer(VersionedConfig, BaseModel):
    VERSION: ClassVar[int] = 1
    IDENTITY_EXCLUDE: ClassVar = ("num_workers",)   # must be annotated ClassVar under pydantic
    learning_rate: float = 0.05
    num_workers: int = 4
    device: str = execution_field(default="cpu")

folder, entry = ledger.resolve_output_folder("train", "/shared/project/runs", Trainer(learning_rate=0.1))
```

With Hydra, add a top-level `ledger:` block and use the launchers:

```yaml
ledger: {root: ${hydra:runtime.cwd}/outputs/ledger, namespace: train}
defaults: [{override hydra/launcher: config_ledger_slurm}]   # or config_ledger_local
```

```bash
python train.py -m sweep=hidden_sizes          # writes outputs/ledger/sweeps/train/hidden_sizes/<ts>.csv
config-ledger versions --ledger outputs/ledger  # runs per (config class, version) and status
```

To launch only the grid points that still need running, declare the stage's completion rule
once with a `RunStatus` and set `hydra.launcher.skip_finished=true`. The launcher asks the
class registered for the namespace about every grid point whose folder exists, logs each skip
with its reason, and submits the rest. The rule is judged from the artifacts. A run that has a
ledger entry but no artifacts is still launched. Skipping needs `ledger.outputs`, the
directory the namespace's run folders are created in:

```python
from config_ledger import RunDecision
from config_ledger.hydra.run_status import SweepRunStatus

class TrainStatus(SweepRunStatus, namespace="train"):   # top of train.py
    @classmethod
    def check_run_status(cls, config, output_folder):
        if (Path(output_folder) / "summary.json").exists():
            return RunDecision(False, "already complete")
        return RunDecision(True, "no summary.json yet")
```

```yaml
ledger: {root: ..., namespace: train, outputs: ${hydra:runtime.cwd}/outputs/train}
```

If every point is skipped, nothing is submitted, the launcher exits 0 and no
`CONFIG_LEDGER_ARRAY_JOB_ID` is printed. A chained stage should only add the dependency when
there is an id:

```bash
TRAIN=$(python train.py -m hydra.launcher.skip_finished=true ... | grep -oP 'ARRAY_JOB_ID=\K\d+' || true)
python evaluate.py -m ... ${TRAIN:+"+hydra.launcher.additional_parameters.dependency=afterany:$TRAIN"}
```

`RunStatus` itself is in the core package and needs no Hydra; the script can call
`TrainStatus.check_run_status(cfg, folder)` itself too, as `examples/toy_mlp/train.py` does.

To run a big sweep piecewise (one seed or architecture at a time, or a temporary backfill
config), first record its full grid without running anything, then file every scoped part
under it:

```bash
python train.py -m sweep=full_grid hydra.launcher.manifest_only=true   # writes the CSV, submits nothing
python train.py -m sweep=seed0_tmp +ledger.parent_sweep=full_grid      # runs; writes no CSV of its own
```

A part with `ledger.parent_sweep` aborts before claiming anything if the parent has no
manifest yet (`ParentSweepManifestMissingError`), or if any of its grid points' folder ids is
not in the parent's manifest (`ParentSweepMismatchError`, listing the points). That usually
means the scoped config resolves a point differently from the parent.

The manifest has a column per `hydra.sweeper.params` key. A sweep given on the command line
(`python train.py -m data_generator.seed=0,1,2`) is not a sweeper param, so by default its
manifest only records `folder_id`. Set `hydra.launcher.overrides_to_manifest=true` to give every
job override its own column as well. That covers command-line sweeps, constants and config
group choices.

To ask which folder already owns a config, without allocating one, use `config-ledger lookup`.
It prints the folder id, or the path with `--parent`. It exits 0 when a folder is found, 1 when
none is, and 2 on errors:

```bash
# a composed config: .json, or .yaml with the [omegaconf] extra (e.g. a run's .hydra/config.yaml)
config-ledger lookup --ledger outputs/ledger --namespace train run/.hydra/config.yaml --json
# compose with Hydra; ledger root/namespace come from the `ledger:` block; -m expands a sweep
config-ledger lookup --config-dir conf --config-name train --setup toy_mlp.schema:register \
    -m sweep=learning_rates                     # one line per grid point: folder id or MISSING
```

On a miss after a version bump, it warns that the run may exist under the old version and
points at `config-ledger upgrade`.

If your configs are dataclasses registered with `ConfigStore` and you keep static YAML copies
of them in the config directory, `config-ledger dump-configs` regenerates those copies. It
writes every entry the `--setup` function registers to `<out>/<group>/<name>.yaml` and never
touches other files there. `--check` writes nothing and exits 1 if a file is missing or out of
date, which is useful in CI:

```bash
config-ledger dump-configs --setup my_app.configs:register --out conf [--check]
```

From Python, call `config_ledger.hydra.dump.dump_config_store(entries, "conf")` with
`(group, name)` pairs, or with `capture_registrations(register)`.

`examples/toy_mlp/` is a complete three-stage pipeline (train, evaluate, aggregate) on a
numpy MLP that runs in about a minute with the local executor; `bash examples/toy_mlp/run_example.sh`.

## Concepts

| Term | Meaning |
|---|---|
| **materialized row** | Flat `{"Root.path.field": "value"}` of every identity-bearing value in a config (defaults filled in), plus each node's `_target_` and `config_version`. This is what is hashed. |
| **materialized tree** | The same values as nested plain JSON, stored with every run so that `upgrade_config` gets real values. |
| **namespace** | The stage that owns the outputs (`train`, `evaluate`). The same config may own one folder per namespace. Always passed explicitly. |
| **sidecar** | `<root>/<namespace>/<hash[:2]>/<hash>.json`, insert-only. Existence means "this identity owns this folder", not "the run finished". |
| **run manifest** | `run_manifest.json` inside each output folder listing every hash that resolves to it, with the row and tree of each; `config-ledger rebuild` regenerates the ledger from these. |
| **sweep manifest** | `<root>/sweeps/<namespace>/<sweep>/<timestamp>.csv`: every sweeper param (and, with `overrides_to_manifest`, every other job override) plus `folder_id`, one row per grid point. |
| **version index** | `<root>/<namespace>/_versions/<_target_>/<version>/{fresh,stale,migrated}/<folder_id>.json`: every run under every `(class, version)` it uses. Status belongs to that one config of the run. |

## When a schema changes

| Change | Old runs | What to do |
|---|---|---|
| Rename a variable, refactor, comment | still resolve | nothing |
| Change a default | still resolve when the old value is requested; configs relying on the new default are new runs | nothing |
| Add / remove a field | new runs raise `SchemaChangedError` | bump `VERSION` (and see the next row) |
| Change the meaning of a class (bump `VERSION`) | new runs raise `StaleRunsError` until old runs are resolved | `config-ledger upgrade --apply` (needs `upgrade_config`), or `config-ledger mark-stale --apply` then optionally `purge-stale --apply` |

```python
from config_ledger import UpgradeImpossible, parse_chain, version_chain

@dataclass
class Trainer(VersionedConfig):
    VERSION: ClassVar[int] = 2          # 2: learning rates are now in units of 1e-2
    learning_rate: float = 5.0

    @classmethod
    def upgrade_config(cls, old: dict) -> dict | None:
        if old["config_version"] == version_chain(cls):
            return None                                   # nothing to upgrade
        if parse_chain(old["config_version"])["Trainer"] == 1:
            return {**old, "learning_rate": old["learning_rate"] * 100}
        raise UpgradeImpossible("no path from this version")
```

All maintenance commands are dry runs unless `--apply` is given, and take `--namespace`,
`--target` (`_target_` or class name) and `--version` (chain or integer) filters:
`config-ledger versions | mark-stale | upgrade [--include-stale] | purge-stale`, plus
`config-ledger rebuild --root <outputs>` to regenerate sidecars and the version index from run
manifests (after losing or moving the ledger or the outputs; conflicts are reported, never
overwritten).

## Related work

- **[exca](https://github.com/facebookresearch/exca)** (Meta): pydantic configs, uid-based
  filesystem cache, submitit. Has declared uid exclusion and a `version` field. config-ledger
  differs in inheritance-propagated versions, upgrade/aliasing instead of orphaning results
  on a bump, sidecar-is-not-completion semantics for resumable jobs, sweep manifests, and
  OmegaConf/Hydra-first configs.
- **[signac](https://github.com/glotzerlab/signac)**: state point -> job id -> directory on a
  shared filesystem. No schema, no versioning; a changed state point renames the directory.
- **[AiiDA](https://aiida.readthedocs.io/)** `CACHE_VERSION` and `_hash_ignored_attributes`:
  same idea, database-backed, not inheritance-aware.
- **[SEML](https://github.com/TUM-DAML/seml)**, **[MLXP](https://github.com/inria-thoth/mlxp)**:
  Slurm sweep managers backed by MongoDB / TinyDB.

See `docs/literature_review.md` for the full survey and `docs/design.md` for the reasoning.

## Development

```bash
pip install -e ".[dev]"
pytest                       # unit tests; `pytest -m e2e` also runs the toy example
ruff check src tests examples
```

### Testing on a Slurm cluster

`pytest -m slurm` submits real jobs through `hydra/launcher=config_ledger_slurm` and checks
the claims the design rests on. Five tests cover these points:

- the driver prints `CONFIG_LEDGER_ARRAY_JOB_ID=<id>` and returns before any task has finished;
- grid points that share an identity share one folder;
- a stage submitted with `+hydra.launcher.additional_parameters.dependency=afterok:<id>` is held
  until the first stage is done;
- a task that raises fails in Slurm (`FAILED`) instead of being reported `COMPLETED`;
- many tasks claiming the same identities at the same instant in a fresh ledger end up with
  exactly one sidecar and one folder per identity, which is the `os.link()` race. This test
  submits a submitit array directly, because the launcher's driver claims every grid point
  before submitting.

The cluster must provide:

- `sbatch`, `squeue` and `scancel` on the host running pytest (`sacct` is optional);
- a directory on a filesystem shared by that host and every compute node, at the same path;
- an environment with `pip install -e ".[dev]"` that compute nodes can run (tasks use the same
  `python` as the driver).

The run is configured with environment variables:

```bash
export CONFIG_LEDGER_SLURM_DIR=~/scratch/config-ledger-tests   # required, shared filesystem
export CONFIG_LEDGER_SLURM_PARTITION=all                       # optional
export CONFIG_LEDGER_SLURM_JOB_PREFIX=cl_test_                 # optional job-name prefix
export CONFIG_LEDGER_SLURM_RACE_NODES=n01,n02,n03              # optional: one race array per node
# optional: CONFIG_LEDGER_SLURM_RACE_TASKS (32), _RACE_IDENTITIES (200), _TIMEOUT_MIN (30)
pytest -m slurm -s           # -s prints each test's directory and the race statistics
```

Without `CONFIG_LEDGER_SLURM_RACE_NODES`, the scheduler may pack the whole race onto one node,
so it never crosses NFS clients. The race needs every task running at once and fails if the
cluster cannot co-schedule them. Outputs stay under `CONFIG_LEDGER_SLURM_DIR` for inspection.
Jobs still queued when a test ends are cancelled.

## License

MIT, Alex Shen.
