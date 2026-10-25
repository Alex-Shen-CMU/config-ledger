# Literature review: does `config-ledger` already exist?

Date: 2026-09-10. Extends an earlier eight-search survey
with ~16 further searches and page reads targeted at the corners that survey flagged as
most likely to hide prior art: HPC/bioinformatics workflow provenance, materials-science
data management, and internal lab frameworks that were later packaged.

## What we are looking for

The proposed package combines four things. A competitor has to cover all four to make
this project redundant:

1. **Content-addressed run identity** derived from a nested config tree.
2. **A declared identity-vs-execution-control split** (which fields count) plus a
   **declared, class-level semantic version** that propagates through inheritance, so
   that "I changed what this class means" invalidates results without hashing source.
3. **Lock-free, database-free correctness on a shared filesystem (NFS) under Slurm-array
   concurrency**, with an explicit "old results reachable under a new identity"
   migration story instead of "new version = new folder, old data orphaned".
4. **Slurm sweep orchestration** that records the grid → run-directory map at submit time.

## Candidates examined

### exca (Meta, 2024–, MIT) — closest overall match

[GitHub](https://github.com/facebookresearch/exca) · [docs](https://facebookresearch.github.io/exca/)

Pydantic-model configs, `@infra.apply` decorator, uid-based filesystem cache, submitit
for Slurm/local execution. It has *declared* identity control:
`_exclude_from_cls_uid` on a class, `exclude_from_cache_uid=(...)` on the decorator, and
all `infra` (resource) parameters are excluded from the uid except a `version` field.
Bumping `version` "will change the current class uid and therefore lead to the creation
of a new cache folder", and the docs warn that "cache of classes depending on the class
with a new version will not be usable anymore".

Where it differs from what we need:

| Aspect | exca | config-ledger |
|---|---|---|
| Identity of a new field with a default | uid built from **non-default** public attributes, so adding a defaulted field does not change identity (a deliberate, reasonable answer to the "new fields" question) | every field counts; new fields need a migration decision. Worth adopting exca's rule as an *option* |
| Version | one integer on the `infra` sub-object; a bump on a base class is **not** propagated to subclasses' identity | MRO-walked `Base:1\|Leaf:2` chain resolved from the live class |
| Old results after a bump | new folder; old cache orphaned unless you hand-copy | `rehash` re-registers existing folders under the new hash; several hashes → one folder |
| Concurrency | per-process `*-info.jsonl` cache files; deterministic key so a race is benign; nothing documented about NFS | `os.link()` claim with explicit loser-adopts-winner; folder ids are uuids so a hash→folder *map* is needed and the claim primitive is what keeps it consistent |
| Config framework | pydantic only | OmegaConf/Hydra (planned: framework-agnostic core) |
| Sweep manifest | none (sweeps are user code) | CSV of grid → folder id written by the launcher |
| Completion semantics | cache hit == result exists (cache stores the return value) | sidecar means "owns folder", completion judged from artifacts; suits long, resumable, preemptible training |

exca is the real prior art for items 1 and 2 and should be cited prominently. It is not
built for the long-running/preemptible/resumable job model (its cache stores a
function's return value), it has no inheritance-aware versioning, and no migration tool.


**Follow-up (user question, 2026-09-10): does exca do yaml configs and sweeps?**

- *YAML / config-logic decoupling:* partially. `ConfDict.from_yaml()` / `to_yaml()` load
  and dump a config, `ConfDict.from_args()` parses `--optimizer.lr=0.01`-style CLI
  overrides, and `ConfDict.flat()` gives dotted keys. There is no composition layer
  (no config groups, defaults lists, interpolation, or `_target_`-style polymorphism):
  the yaml must mirror one pydantic model, and the pydantic class *is* the schema, with
  the decorated method living on the same class. Logic and config are one object by
  design.
- *Sweeps:* launching only. `TaskInfra.job_array()` collects tasks in a context and
  submits them as one submitit/Slurm array (`slurm_array_parallelism=max_workers`,
  in-memory uid dedup via `allow_repeated_tasks=False`). No grid/multirun syntax, and
  **no manifest of the batch is written** — each task writes its own `job.pkl`. The
  query side is `iter_cached()`, which walks the class's whole cache folder and yields
  every reconstructed config that has a `job.pkl`; it takes no filters and has no notion
  of "the runs that belonged to this sweep". So "a sweep is both a launcher and a
  query" is not something exca offers.

### signac (Glotzer lab, 2016–, BSD) — closest on items 1 and 3

[GitHub](https://github.com/glotzerlab/signac) · [SciPy 2018 paper](https://pdfs.semanticscholar.org/dd7b/d60ac9c6b6247038e6a1a3094fe4126db044.pdf)

A *state point* (JSON dict) is hashed to a *job id* which names a workspace directory;
HPC/shared-FS is the native environment, no database. The state point / job document
split is exactly the identity / execution-control split, but decided per-key by the
user at `open_job()` time, not declared on a schema. No class or code versioning of any
kind; the docs say a changed state point means "the job's workspace directory is
renamed" and warn to "take great care" — i.e. migration is a move, not an alias.
Concurrency: creating a job is a `makedirs` on a deterministic id (benign race), but
"read and write operations to the job-/ and project-document are not protected against
race-conditions". signac's `signac-flow` judges completion by user-declared
post-conditions on artifacts, the same rule config-ledger uses.

### AiiDA (2.6+) — closest on item 2, but database-backed

[Caching docs](https://aiida.readthedocs.io/projects/aiida-core/en/latest/topics/provenance/caching.html)

Node hash from attributes minus `_hash_ignored_attributes`, plus a class-level
`CACHE_VERSION` integer that developers bump "for significant implementation changes".
Notably AiiDA *removed* the package `__version__` from the hash in 2.6 because it
invalidated everything on every release — independent confirmation that a coarse
declared version beats automatic code hashing. `CACHE_VERSION` is a plain class
attribute, so an override on a subclass silently drops the base class's bump (the S1
failure the MRO chain prevents). Requires PostgreSQL + RabbitMQ; not usable for our
audience.

### SEML (TUM-DAML) — Slurm sweeps with config-hash dedup, MongoDB-backed

[GitHub](https://github.com/TUM-DAML/seml). Queues sacred experiments into MongoDB with a
`config_hash` ("checks whether experiments with the same configuration are already in
the database collection"), then submits Slurm arrays. Covers items 1 and 4, but the
dedup lives in a database service, hashes the config literally, and has no versioning.

### MLXP (Inria, ACM REP 2024)

[GitHub](https://github.com/inria-thoth/mlxp) · [paper](https://arxiv.org/abs/2402.13831).
Hydra-based launcher, per-run directories, TinyDB result querying, `mlxpsub` for
Slurm/PBS/OAR/LSF. Code versioning is *automatic git-commit snapshotting*, the approach
the earlier survey argued against for identity. No content addressing; runs are numbered.

### OxyMake (2026, Rust)

[arXiv 2606.20989](https://arxiv.org/abs/2606.20989). "A result's identity is a pure
function of what its rule declares: command text, inputs, parameters, environment";
single binary, no service. A workflow engine for command-line rules, not a Python config
library; identity includes command text (source-hashing flavor). Validates the
"content-addressed, no daemon" direction and is worth citing.

### Others checked and why they do not apply

- **hydra-cached** — Hydra `instantiate` with (sub)configs as cache keys; in-memory with
  optional pickle persistence; no versioning, no concurrency story, small/unmaintained.
- **DVC run-cache, Dagster `code_version`, joblib, redun (`hash_includes`), Nextflow,
  Snakemake `--rerun-triggers`** — covered in the earlier survey; all either hash source/params
  literally or propagate versions along a DAG rather than a class hierarchy.
- **pyiron / pyiron_core** — hash-based caching of workflow graphs into HDF5; materials
  science workflow engine, not a config library.
- **Sacred, MLflow, W&B, Aim, Guild AI, ClearML** — tracking; none define identity from
  config (Aim's `run_hash` is a random id you may reuse).
- **slurm_sweep, MMF sweep script, pyexperiment, SmartSim, ExperimentManager,
  exp-manager** — launchers or run-folder bookkeeping with numbered/timestamped runs.
- **noWorkflow, Sumatra, YesWorkflow, HyProv/ProvDeploy** — retrospective provenance
  capture; answer "what ran", not "has this already run".
- **hashfs** — content-addressed *file* store (hash of bytes), unrelated to configs.

## Verdict

No existing package provides the combination. Two are close enough that the
package's thesis must be stated relative to them:

- **exca** already offers declared uid exclusion, a declared version, filesystem caching
  and submitit integration for pydantic users. config-ledger's differentiation is
  (a) inheritance-propagated versions, (b) the rehash/alias migration model where a
  version bump never orphans results, (c) the "sidecar ≠ completion" model for long
  resumable jobs, (d) the explicit NFS-safe claim primitive, and (e) framework-agnostic
  configs (OmegaConf/Hydra first, dataclass/pydantic adapters later).
- **signac** already offers content-addressed job directories on shared filesystems for
  HPC. config-ledger's differentiation is schema-declared identity, versioning, and
  aliasing instead of directory renames.

Design consequences adopted from the review:

1. Offer exca's "defaulted fields do not enter identity" as a selectable policy; it is
   the cleanest answer to "do new config fields deserve a new identity automatically".
2. Keep the class-level version integer (matches AiiDA's `CACHE_VERSION` and exca's
   `version`, so the concept is familiar), but propagate it through the MRO.
3. Cite AiiDA 2.6's removal of package version from the hash as evidence for coarse
   declared versions over automatic hashing.

Recommendation: **proceed to the scaffold**, with exca and signac named in the README's
"related work" from day one.

## Sources

- https://github.com/facebookresearch/exca
- https://facebookresearch.github.io/exca/infra/howto.html
- https://facebookresearch.github.io/exca/infra/explanation.html
- https://github.com/glotzerlab/signac
- https://signac.readthedocs.io/en/latest/jobs.html
- https://aiida.readthedocs.io/projects/aiida-core/en/latest/topics/provenance/caching.html
- https://github.com/TUM-DAML/seml
- https://github.com/inria-thoth/mlxp and https://arxiv.org/abs/2402.13831
- https://arxiv.org/abs/2606.20989
- https://github.com/ArneBinder/hydra-cached
- https://github.com/pyiron/pyiron_core
- https://github.com/quadbio/slurm_sweep
- https://github.com/duerrp/pyexperiment
- https://github.com/CrayLabs/SmartSim
- https://github.com/dgilland/hashfs
