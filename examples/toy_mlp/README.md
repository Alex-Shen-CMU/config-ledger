# Toy example: three-stage numpy MLP pipeline

```
train.py      -> outputs/train/<uuid>/{checkpoint.npz, summary.json, run_manifest.json}
evaluate.py   -> outputs/evaluate/<uuid>/{metrics.json, run_manifest.json}
aggregate.py  -> table from outputs/ledger/sweeps/evaluate/<sweep>/*.csv
```

`bash run_example.sh` runs two overlapping training sweeps (10 distinct runs out of 12
grid points; the two shared points are trained once), evaluates both, aggregates each from
its sweep manifest, and ends with a `config-ledger versions` report.

Things to try afterwards:

- Re-run `python train.py -m sweep=hidden_sizes`: every task logs "already complete". Add
  `hydra.launcher.skip_finished=true` and the launcher asks `TrainStatus` (declared at the top
  of `train.py`) before submitting: "All 6 grid point(s) skipped; nothing submitted".
- Kill a run mid-way and re-run it: it resumes from `checkpoint.npz`.
- Set `trainer.checkpoint_every=10`: same identities, it is an `execution_field`.
- Bump `Trainer.VERSION` to 2 in `toy_mlp/schema.py` and run `python train.py`: it refuses
  with `StaleRunsError`, because the 10 runs under `Trainer:1` are still "fresh". Then either
  - `python -m config_ledger.cli mark-stale --ledger outputs/ledger --apply` to declare them
    outdated (`purge-stale --apply` deletes them), or
  - give `Trainer` an `upgrade_config` like `Model`'s and run
    `python -m config_ledger.cli upgrade --ledger outputs/ledger --apply`: the old folders now
    answer for the upgraded configs, and the sweep is a no-op again.
- Add a field to `Trainer` without bumping `VERSION`: the next run raises
  `SchemaChangedError`, since the new field would silently change every hash.
- Change `Trainer.learning_rate`'s default to 0.1 and run `python train.py`: a new run, since
  the old one was trained at 0.05 and every value is part of the identity.

On a Slurm cluster add `hydra/launcher=config_ledger_slurm` to each `-m` invocation and chain
stages with `--dependency`; see `config_ledger/hydra/launcher.py`.
