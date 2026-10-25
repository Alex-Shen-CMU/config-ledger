"""Tiny two-stage Hydra app driven by ``tests/slurm`` through the real Slurm launcher.

Kept free of numpy and of the toy example so the integration tests control timing
(``work.sleep_s``) and failure (``work.fail_on_value``) exactly. ``tests/slurm/app`` must be on
``PYTHONPATH`` both for the driver and for the array tasks (sbatch exports the environment).
"""

DONE = "done.json"
SEEN = "seen"
FAIL_SENTINEL = "config-ledger-slurm-test: deliberate task failure"
