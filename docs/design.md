# Design notes

## Thesis

Every existing tool either hashes the config literally (signac, SEML, DVC) or hashes source
code (joblib, Nextflow, MLXP's git snapshots). Neither is right for a research codebase under
constant refactor: literal hashing means every new field orphans every old run, and source
hashing means every rename does. The right unit of invalidation is *a declared statement
by the author that the meaning changed*, attached to the class that changed, and propagated
to everything that inherits from it.

## Identity: every value, defaults filled in

Two candidate rules were weighed (2026-09-10):

1. **All values enter the hash.** Changing a default has no side effects on old runs; adding a
   field requires migrating every old run.
2. **Only non-default values enter the hash** (exca's rule). Adding a field that was
   implicitly there all along does not stale old runs; changing a default moves the meaning
   of "no override" underneath old hashes.

Rule (2) was adopted first, with a guard: sidecars stored the full materialized row and every
hit was compared against it (`IdentityDriftError`). A 2026-10-17 revision replaced it with
rule (1). The guard only made a changed default *loud*; recovering still needed a full scan of
every run manifest (`rehash`), and a changed default kept breaking old hashes. Under rule (1)
a changed default is a non-event: old runs keep the identity of the value they actually used,
and new configs relying on the new default are new runs. Adding or removing a field is the
remaining hazard, and two declared mechanisms now cover it: the version bump that the schema
guard enforces, and `upgrade_config` (see "Stale runs" below). The policy layer
(`DefaultsExcluded` / `FullValues`, `--policy`, `ledger.policy`) and drift verification are
gone.

Two rules keep "the same config" one identity:

- **Omitted fields are filled from the schema default** of the node's `_target_` class, so a
  yaml that spells a default out and one that omits it hash the same. A missing required
  field raises.
- **Undeclared keys raise.** A key the class does not declare would enter the hash but not the
  field set the schema guard records. Classes without a fixed field set (a pydantic model with
  `extra="allow"`, a plain class taking `**kwargs`) are taken as written.

An instance of a schema class (a pydantic model) is complete by construction and is not
filled. A field its view leaves out was excluded per instance (exca's instance-method
`_exclude_from_cls_uid`). Known limitation: re-materializing such a model's *stored tree*
during an upgrade fills that field back in, since the instance is gone by then.

`config_version` is a reserved field name: it is where each node records its version chain.

Hydra's reserved instantiate keys control how a node is built, not what the run is.
`_convert_`, `_recursive_` and `_partial_` are skipped in every node, in core and whether or
not omegaconf is loaded, and also when a dataclass declares one as a field, so moving
`_convert_: all` from yaml into the class keeps every run's id. `_args_` raises: its positional
values would otherwise be dropped from the hash.

## Stale runs: the version index

A version bump used to orphan old runs silently. They stayed on disk, unreachable, and finding
them meant walking every run manifest. Now each namespace keeps an index of which runs use
which `(class, version chain)`:

```
<root>/<namespace>/_versions/<_target_>/<version slug>/
    meta.json                    target, raw chain, identity fields (schema guard)
    {fresh,stale,migrated}/<folder_id>.json
```

- **Status belongs to one config of a run, not to the run.** A run can be `stale` in
  `Trainer:1` and `fresh` in `Data:1`. `purge-stale` reports which config caused a run to be
  purged.
- **Status is the subdirectory**, so changing it is one `os.rename` (atomic on NFS) and "does
  an old version still have fresh runs?" is one `listdir`. Entries are claimed with
  `os.link()` like sidecars, lazily, by the first run that uses a version.
- **Per namespace**, so a stale `train` stage does not block `evaluate`.
- **Discovery.** `resolve_output_folder` refuses to start (`StaleRunsError`) while any
  non-live version of one of its classes still has `fresh` entries, before anything is
  claimed. `stale` entries under old versions are allowed; `migrated` ones are ignored. A
  stored version newer than the code raises `NewerVersionError` (an old checkout). Read-only
  lookups (`lookup_output_folder`) are never blocked.
- **Sweeps check once.** The launcher composes the grid (each job config once, shared with the
  sweep manifest), checks the union of every class any job uses, and aborts the submission
  on failure. Tasks then skip their own check (`skip_version_check()`, set in the launcher's
  per-task `__call__`, gated on a flag pickled with the launcher). If the driver cannot
  compose the grid, tasks check individually. This lives in the `[hydra]` extra; the core
  exposes `Ledger.check_configs` for any other driver.
- **Schema guard.** With every value hashed, adding a field without bumping `VERSION` would
  change every hash with no stale signal. `meta.json` of the live version records the identity
  field names, and a later run under the same version with a different field set raises
  `SchemaChangedError`. It compares declared fields, never source code, so it stays within
  the thesis.

Resolution is the user's call, through the CLI (dry run unless `--apply`):

| Command | Effect |
|---|---|
| `versions` | `(class, version) -> fresh/stale/migrated` counts, live/older/newer marked |
| `mark-stale` | `fresh` -> `stale` under old versions; refuses the live version (rerun with `overwrite` instead) |
| `upgrade` | runs with a fresh config under an old version: upgrade the stored tree, register the new hash against the same folder, old entries -> `migrated`, record under live versions. Skips runs with `stale` configs unless `--include-stale`. A new hash already owned by another folder is a `CONFLICT`. |
| `purge-stale` | delete output folder, sidecars and all index entries of runs with a stale config; sweep CSVs stay (history) |
| `rebuild` | regenerate sidecars and index entries from run manifests (replaces `index backfill`, which replaced `rehash`) |

Selection: `--namespace`, `--target` (`_target_` or class name), `--version` (full chain or the
leaf's integer), all repeatable.

### Rebuilding the ledger

The run manifests are the durable copy of the ledger. `config-ledger rebuild --root <outputs>`
regenerates both halves from them: after losing the ledger root, moving it, or moving the
outputs.

- **Every hash keeps its row.** The manifest stores `configs: {hash: {config, tree}}`, not
  just the latest row, so the sidecar of an upgraded-from hash can be rebuilt.
- **Rows are re-hashed as stored, never re-materialized.** Materializing re-resolves
  `config_version` from the live class, which would give an old run the new version's hash.
  So no project class needs to import, which suits recovering after the code has moved on. A
  row that doesn't hash to its listed hash fails the whole manifest.
- **Reconcile, never clobber.** Missing sidecars are claimed with `os.link()`. A sidecar that
  already points at another folder, or a hash that two manifests claim, is a `CONFLICT`, and
  that run isn't indexed. A folder whose name isn't its `folder_id` (renamed, so lookups miss
  it) or a `folder_id` found at two paths is `FAILED`. Exit 1 on either.
- **Index statuses.** Versions of the latest hash come back `fresh`. Versions only older hashes
  used come back `migrated`, since the run was upgraded from them. `stale` is only recorded by
  the index layout and comes back `fresh`, so the stale check flags it again for the user to
  re-mark. Existing entries keep their status. If their `output_folder` isn't where the
  manifest was found (outputs moved), they are relocated, so `purge-stale` deletes the right
  folder.
- **Orphans** (sidecars or index entries of a folder with no manifest under the scanned roots)
  are reported only, since the roots may cover only part of the outputs.

### Lookups

`config-ledger lookup` answers "which folder owns this config?" and never writes: no sidecar,
no index entry, no folder. It is not blocked by the stale check either. Since full-value hashing, a hit can't
disagree with the config, so the old `IdentityDriftError` has no counterpart. The failure mode
is now a *miss after a version bump*: the live hash is new, and the run sits under the old
version until it is upgraded. `VersionIndex.diagnose` is the read-only counterpart of `check`.
It returns warnings instead of raising:

- on a miss, older versions of the config's classes that have `fresh` or `stale` runs ("this
  config may have run under the old version");
- always, versions newer than the code (an old checkout) and schema changes.

It doesn't try to match the config against old runs, which would need `upgrade_config` on
every stored tree.

Inputs follow the dependency grouping. The core reads `.json`, plus `.yaml` through the
`[omegaconf]` extra, so a YAML file hashes as the `DictConfig` a run sees. A Hydra dump
(`.hydra/config.yaml`) keeps `${hydra:...}` in top-level execution-control keys, which can't
resolve outside a job. Those keys are kept as raw strings, since they are never identity, so
the loader only works around the top-level xfail described under Hydra gotchas.
`--config-name` composes with Hydra (`config_ledger.hydra.lookup`, imported lazily). `-m`
expands the grid as `BasicSweeper` does (`sweeps.expand_grid`) and composes each job with the
launcher's own `sweeps.job_composer`, so lookup and launcher see the same job configs.
`--setup MODULE[:FUNC]` runs an app's `ConfigStore` registration first. The current directory
and `--pythonpath` entries go on `sys.path`, so `_target_` classes import. Exit codes are 0 if
found, 1 if missing (for a sweep, any grid point missing), and 2 on errors.

### Upgrades

`VersionedConfig.upgrade_config(old) -> dict | None` is a classmethod. It returns `None` when
no upgrade is needed (stored chain == live chain), a new node when an upgrade is possible, or
raises `UpgradeImpossible(reason)`. The default knows no paths. `old` comes from the **stored
tree**: every run (sidecar and run manifest) stores its materialized config as nested plain
JSON next to the flat row, because the row's encoded strings cannot be turned back into
typed values. `upgrade_tree` runs bottom-up, so parents see upgraded children, then
re-materializes the result against the live classes (defaults filled, unknown keys
rejected). Every failing node is reported; any failure fails the whole tree.

## What is identity, declared three ways

| Mechanism | Scope | Use |
|---|---|---|
| Node is a `VersionedConfig` (has `_target_`) | subtree | everything inside is identity unless excluded |
| Structured collection in a node (list / dict of nodes and `None`) | each entry | entries are nodes, as above |
| `execution_field(...)` / `IDENTITY_EXCLUDE` (pydantic also: exca's `_exclude_from_cls_uid`) | one field | devices, worker counts, cache dirs, cadence |
| Loose top-level key (no `_target_`) | whole key | `overwrite`, output paths, the `ledger:` and `hydra:` blocks |
| Hydra instantiate key in a node (`_convert_`, `_recursive_`, `_partial_`) | one key | always excluded, not declared per class |

The top-level rule is inherited from the source project and is the part the literature
review identified as needing a real API; the field-level mechanisms are that API. A versioned
pydantic model passed as the whole config is itself the single root node, so
exca-style "one model is the config" callers need no wrapper dict.

## Leaves are encoded strictly

A leaf must be `None`, `bool`, `int`, `float`, `str`, an `Enum` (encoded by name), a
`PurePath` (encoded as `str`), or a list/tuple/dict of those, after its adapter converts it.
Anything else raises `TypeError` (since 2026-10-17). Before, `encode_leaf` fell back to
`str(value)`, so an object no adapter recognised hashed by its repr: a different, silently
wrong identity, e.g. a pydantic model hashed in a process that never loaded its adapter.
Note that numpy scalars other than `float64` (which subclasses `float`) are now rejected.

## Field values: leaves, nodes, structured collections

Before structured collections, a field was recursed into only if its value was itself a `_target_` mapping;
anything else was one JSON leaf. A `Mapping[str, TrainingStep | None]` pipeline therefore
hashed as one string: its steps got no `config_version`, no default filling and no key
checks, so bumping a step's `VERSION` silently reused stale results. Now every identity field
of every schema node (excluded fields are not checked) must hold one of:

| Value | Identity |
|---|---|
| a primitive, or a list/tuple/dict of primitives nested to any depth | one leaf (JSON for containers), as before |
| a schema node (`_target_` mapping) | recursed into |
| a *structured collection*: a list/tuple or `_target_`-less dict whose entries are all schema nodes or `None`, at least one a node | each node entry recursed into at `<field>.<index or key>`; `None` is the leaf `""` |

Anything else that holds a schema node raises `ValueError`, rather than hiding the node's
version in a leaf: a list mixing nodes and other values, a list of lists of nodes, a dict of
lists of nodes. Nested structured collections are rejected rather than recursed for now;
allowing them later changes no existing hash.

- Values are checked, not annotations: yaml and `DictConfig` configs carry no types, and
  `Any` / union annotations would need per-framework typing introspection.
- A non-versioned dataclass or pydantic model is read as a dict, so it is a leaf
  when its fields are primitives and a structured collection when they are all nodes.
- Entries whose `_target_` is not a `VersionedConfig` (a third-party class) are nodes with an
  empty `config_version`, as direct fields already were.
- The stored tree keeps the container's shape (a list or dict of nodes and `None`);
  `flatten` and `upgrade_tree` recognise collections there by the same rule, and
  `upgrade_tree` upgrades entries before their parent.
- A list and a dict keyed `"0"`, `"1"`, ... give the same rows. The schema fixes a field's
  container type, so this is not treated as a collision.
- Top-level config entries are unaffected: non-node values there stay execution control.
- Hashes change for configs with such collections. No ledger held one yet, so nothing was
  migrated; a stored tree still re-materializes into the new rows if one is ever needed.

A schema default that is a dataclass instance (`trainer: Trainer = field(default_factory=Trainer)`)
is read through the dataclass adapter, like a dataclass instance passed in the config.

## Config adapters: the seam between frameworks and the core

The core (`identity`, `store`, `versions`, `upgrade`, `maintenance`, `hash_row`,
`canonical_json`) is plain-dict and imports no config framework; its only dependency is
`xxhash`. Reading a framework's
config objects goes through `config_ledger.adapters` (since 2026-10-17). An adapter
answers three questions about one value of a config tree:

| Method | Returns | Contract |
|---|---|---|
| `handles(value)` | `bool` | Cheap; must not resolve interpolations or import the framework. |
| `as_mapping(value)` | `Mapping[str, Any]` or `None` | A view of a node's fields; `None` for that framework's leaves (e.g. lists). May be lazy: only identity nodes are iterated, and yielded values are dispatched again. A node is a schema node iff the view has `_target_`. |
| `to_plain(value)` | builtins | A list/dict *leaf* as builtin containers with interpolations resolved, ready for `encode_leaf`. |

An adapter is correct when the row it produces equals the row of the same config written
as a plain nested dict (`tests/test_adapters.py` checks this for OmegaConf).

Two decisions about the seam:

- **Adapters read config objects, not schema classes.** `VersionedConfig` is not subclassed
  per framework: Hydra users and plain-dict users declare the same schema and differ only
  in the object passed at run time. Schema introspection is a separate seam: a framework
  whose schema classes are not dataclasses also exposes a `SchemaAdapter`
  (`handles_class` / `prepare_class` / `schema_fields` / `schema_defaults` /
  `excluded_fields`), which
  `versioning` consults from `VersionedConfig.__init_subclass__`, `schema_fields()`,
  `schema_defaults()` and `identity_excluded_fields()`. Only pydantic has one.
- **Discovery is internal and import-free.** The adapter list is fixed; we decided against a
  public `register_adapter()` (an in-tree entry suffices; revisit when a third-party adapter
  needs one). A framework adapter is consulted only once its framework is already in
  `sys.modules`, because no object of that framework can exist otherwise. So the core never
  imports an optional dependency; the OmegaConf adapter ships in the `[omegaconf]` extra
  (pulled in by `[hydra]`), the pydantic one in `[pydantic]`.

### Pydantic

`class Trainer(VersionedConfig, BaseModel)`, either base order. Differences from dataclasses:

- `_target_` is a plain class attribute, not a field (pydantic would make a `_`-prefixed
  annotation a private attribute); the adapter adds it to a versioned model's mapping view.
- `execution_field()` is converted to a pydantic field with an `EXECUTION` metadata marker in
  `prepare_class`; only `default` / `default_factory` carry over.
- Class-level declarations must be annotated: `IDENTITY_EXCLUDE: ClassVar = (...)`.
- exca's `_exclude_from_cls_uid` (tuple/list/set, classmethod, or instance method; `"."` =
  all fields, the node keeping only `_target_` and version) is honoured. Class-level forms
  also apply to dict configs naming the class; an instance method only to model instances.
  Pydantic's `Field(exclude=True)` is deliberately not identity exclusion.
- A model without `_target_` is one JSON leaf, like a nested plain dict (and, like one, a
  structured collection if its fields are all schema nodes).
- Defaults come from `model_fields` (`default_factory` called). Pydantic does not validate
  defaults, so `lr: float = 1` passed explicitly as `lr=1` validates to `1.0`, encodes as
  `"1.0"` against the default's `"1"`, and enters the hash; declare defaults in their
  validated type.

### Plain dataclasses

The zero-framework path: `config_ledger.adapters.dataclass` reads dataclass instances using
only the stdlib. It is always active (no `sys.modules` gate) and consulted after the framework
adapters, before the builtin mapping adapter. A tree of instances gives the same row and hash
as its `dataclasses.asdict` form and as `OmegaConf.structured(...)` of it, with one exception
(the third point below).

- A `VersionedConfig` instance is a schema node. Its target is always its own class's import
  path; an instance whose `_target_` field holds another value (`Trainer(_target_="x")`)
  raises `ValueError`. Passed as the whole config, it is the single root node, as for
  pydantic.
- A non-versioned dataclass passed as the whole config (`@dataclass class Config: trainer:
  Trainer; output_dir: str`) is walked like a top-level dict: versioned fields are identity
  roots, everything else execution control.
- A non-versioned dataclass as a field value is one JSON leaf, *minus* the fields its class
  excludes (`execution_field` / `IDENTITY_EXCLUDE`), as for a non-versioned pydantic model.
  If its fields are all schema nodes (or `None`), it is a structured collection instead.
  `asdict` keeps those fields, so here the two forms hash differently.
- A pydantic dataclass (`pydantic.dataclasses.dataclass`) is not a `BaseModel`, so it is read
  by this adapter. The values read are the validated ones, so the same default caveat as for
  models applies: `lr: float = 1` passed as `lr=1` hashes `"1.0"`.
- Before the dataclass adapter, `materialize()` raised `TypeError` on any dataclass instance. Silently
  mishandling one was the outcome to avoid: a top-level `VersionedConfig` instance would
  have been skipped as execution control (all such configs sharing one hash), and one inside
  a node would have hashed its repr. Identities of dict / DictConfig configs are unchanged.

## Why uuids for folders, not the hash

If the folder were named by the hash, a version bump would require *moving* results (signac's
rename) and a folder could never be reachable by two identities. Opaque ids plus a hash ->
id map make migration an insert: `config-ledger upgrade` registers the upgraded hash against
the existing folder. The map is what the `os.link()` claim keeps consistent.

## Why files and `os.link()`

The ledger lives on the same NFS export the jobs write to. SQLite and LMDB depend on POSIX
byte-range locks / mmap, which are unreliable there (we have seen exactly that
corruption in practice). `os.link()` is atomic on NFS, fails with `EEXIST` for the loser, and
needs no lock file, no timeout, no stale-lock recovery. Sidecars are only ever inserted; the
version index changes an entry's status with `os.rename`, which is atomic on NFS too.

## Hydra gotchas found while building the example

- Hydra instantiates a launcher only if its class module is under the top-level
  `hydra_plugins` namespace. Logic lives in `config_ledger.hydra.launcher`; the concrete
  classes are one-line subclasses in `hydra_plugins/config_ledger_launcher/submitit_launcher.py`.
- The `${hydra:...}` resolver requires `HydraConfig` to be set, which only happens inside a
  job. The launcher sets it on the driver before reading the `ledger:` block; example configs
  keep `hydra.run.dir` / `hydra.sweep.dir` relative for the same reason.
- `is_structured()` must not resolve interpolations, or identity code would trip over
  execution-control blocks that reference runtime-only resolvers. This protects nested
  blocks (`ledger:`, `hydra:`) only: iterating the top level of a `DictConfig` resolves
  top-level *scalar* values, so `outputs: ${hydra:runtime.cwd}/out` at the top level fails
  outside a job (recorded as an xfail in `tests/test_adapters.py`).
