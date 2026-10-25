#!/usr/bin/env bash
# End-to-end demo. From the repo root, after `pip install -e ".[example]"`:
#     bash examples/toy_mlp/run_example.sh
# Off-cluster this uses the local submitit executor (hydra/launcher=config_ledger_local);
# on a Slurm cluster append `hydra/launcher=config_ledger_slurm` to the -m lines and chain
# stages with --dependency as described in config_ledger/hydra/launcher.py.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}

echo "== 1. train sweep (6 runs)";            $PY train.py -m sweep=hidden_sizes
echo "== 2. overlapping sweep: 2 of 6 already trained, not launched"
$PY train.py -m sweep=learning_rates hydra.launcher.skip_finished=true
echo "== 3. evaluate both sweeps";            $PY evaluate.py -m sweep=hidden_sizes; $PY evaluate.py -m sweep=learning_rates
echo "== 4. aggregate from the sweep manifest"
$PY aggregate.py --sweep hidden_sizes
$PY aggregate.py --sweep learning_rates
echo "== 5. runs per (config class, version); bump a VERSION and see \`config-ledger upgrade\`"
$PY -m config_ledger.cli versions --ledger outputs/ledger
