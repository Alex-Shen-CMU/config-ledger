---
name: config-ledger
description: Run identity and the results store (config-ledger, https://github.com/Alex-Shen-CMU/config-ledger) - how configs map to output folders, sweep manifest CSVs, VERSION bumps, upgrade/mark-stale, piecewise sweeps, finding a run's folder. Use before adding/removing a config field, bumping a VERSION, migrating old runs, or when completed results look missing.
---

# config-ledger: run identity and the results store

config-ledger's `README.md` is the full API and `docs/design.md` the rationale. `config-ledger --help` lists the CLI. Below, `<root>` is the ledger root (the `ledger.root` key of a Hydra config, or the path passed to `Ledger(...)`), `$LEDGER` is the same path in shell commands, and `<namespace>` is the stage that owns the outputs (e.g. `train`, `evaluate`).

## Identity

- A run's identity is the hash of every top-level versioned node of its config: each node's fields (schema defaults filled in for omitted fields), its class's version chain, and nested nodes, dicts and lists of nodes (a `None`/`null` slot counts as `""`).
- **Loose top-level keys are never identity** (e.g. `overwrite`, `output_dir`, the `ledger:` block). Neither are fields declared with `execution_field(...)` or listed in a class's `IDENTITY_EXCLUDE`. Anything that changes results must live inside a `VersionedConfig` node. To make a loose value identity, wrap it in a `VersionedConfig` of its own.
- Hydra's `_convert_` / `_recursive_` / `_partial_` aren't identity; `_args_` is rejected.
- The same config can own one folder per namespace. Namespaces are always passed explicitly. They are never inferred from the script or `hydra.job.name`.
- Every config class inherits `VersionedConfig` and declares its own `VERSION: ClassVar[int]` (enforced at import). Versions come from the live class's MRO at hash time, so bumping a base class invalidates its subclasses.

## Layout

Keep one shared ledger root per cluster, outside any checkout, so every worktree and node sees the same runs.

| Path | Holds |
|---|---|
| `<root>/<namespace>/<hash[:2]>/<hash>.json` | Sidecar: this config owns this folder. Insert-only. |
| `<root>/<namespace>/_versions/<_target_>/<version>/{fresh,stale,migrated}/` | Version index: every run under every `(class, version)` it uses. |
| `<root>/sweeps/<namespace>/<sweep>/<timestamp>.csv` | Sweep manifest: one row per grid point, every sweeper param plus `folder_id` (every job override too with `hydra.launcher.overrides_to_manifest=true`). `<sweep>` is the choice of the `ledger.sweep_group` config group (default `sweep`). Written at submission by the config-ledger launchers only. |
| `<outputs>/<uuid>/` | Opaque output folders, each with `run_manifest.json` (every hash resolving to it). `config-ledger rebuild --root <outputs>` regenerates the ledger from these. |

**A sidecar or CSV row doesn't mean the run finished.** Folder ids are claimed eagerly. Judge completion from the run's own artifacts. Declare that rule once in a `RunStatus` subclass registered for the namespace, which `hydra.launcher.skip_finished=true` also uses.

## Finding results

- Sweep results: read the sweep's CSV (`config_ledger.hydra.sweeps.load_sweep`, or any CSV reader) and join on `folder_id`.
- One config: `config-ledger lookup --config-dir <conf> --config-name <entrypoint> [--setup <module>:<fn>] [overrides] [-m]` prints the folder id, or one line per grid point with `MISSING` for misses. For an already composed config: `config-ledger lookup --ledger $LEDGER --namespace <namespace> <config.yaml|.json>`. It's read-only and never allocates. It exits 0 when a folder is found, 1 when none is and 2 on error.

## Changing a config class

| Change | What to do |
|---|---|
| Rename, refactor, comment | Nothing. |
| Change a field's default | Nothing. Old runs still resolve when they set the old value; configs relying on the new default are new runs. |
| Add or remove a field | Bump `VERSION`, else new runs raise `SchemaChangedError`. Then do the next row. |
| Bump `VERSION` | Every submission using the class raises `StaleRunsError` until the old runs are resolved. **Immediately**, before submitting anything: `config-ledger versions --ledger $LEDGER`, then either `config-ledger upgrade --ledger $LEDGER [--apply]` (carries old runs forward; needs `upgrade_config` on the class) or `config-ledger mark-stale --ledger $LEDGER --target <Class> [--apply]` (old runs stop counting, aren't deleted; `purge-stale` deletes them). |

- All maintenance commands are dry runs without `--apply`. They take `--namespace`, `--target` and `--version` filters.
- **The ledger is shared state.** A half-done bump blocks every other session's submissions using that class. Never `mark-stale` or `purge-stale` runs that other work may depend on without asking the user.
- If the project keeps YAML copies of its config classes, write new fields into them too (`config-ledger dump-configs --setup <module>:<fn> --out <conf>` regenerates them; `--check` verifies).

## Piecewise sweeps

A sweep's CSV directory is named after its sweep-group choice. A temporary or scoped sweep config would otherwise file rows under its own name, and downstream readers would read the parent's stale CSV. Always:

1. **Record the parent's full grid, running nothing**: `python <script>.py -m <sweep_group>=<parent> hydra.launcher.manifest_only=true`. Skip this if the parent already has a CSV. Each namespace has its own manifests, so record each stage.
2. **Submit every part with `+ledger.parent_sweep=<parent>`.** It writes no CSV of its own. It aborts before claiming anything if the parent has no manifest (`ParentSweepManifestMissingError`) or if a point's folder id isn't in it (`ParentSweepMismatchError`). That usually means the scoped config resolves a point differently from the parent.

## Debugging reference

Look up by what you see. Read the linked issue (its description ends with a Resolution section) only when you hit it.

| Symptoms | Cause | Issue |
|---|---|---|
