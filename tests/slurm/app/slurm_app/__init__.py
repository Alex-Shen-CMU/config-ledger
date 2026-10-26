"""Tiny two-stage Hydra app driven by ``tests/slurm`` through the real Slurm launcher.

Kept free of numpy and of the toy example so the integration tests control timing
(``work.sleep_s``) and failure (``work.fail_on_value``) exactly. The launcher puts
``tests/slurm/app`` (the scripts' directory) on the array tasks' path; tasks submitted directly
with submitit (the race test) need it on ``PYTHONPATH`` (sbatch exports the environment).
"""

DONE = "done.json"
SEEN = "seen"
FAIL_SENTINEL = "config-ledger-slurm-test: deliberate task failure"
