"""Stage 1: claim this config's folder, optionally sleep or fail, then write ``done.json``.

Every task also writes ``seen/<array task id>.json`` into its folder, so the tests can check
that grid points sharing an identity all ran in the same folder.
"""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig
from slurm_app import DONE, FAIL_SENTINEL, SEEN, schema

from config_ledger.hydra import ledger_from_config

schema.register()


@hydra.main(config_path="conf", config_name="stage1", version_base="1.3")
def main(cfg: DictConfig) -> None:
    started = time.time()
    ledger, namespace = ledger_from_config(cfg)
    folder, entry = ledger.resolve_output_folder(namespace, Path(cfg.outputs) / namespace, cfg)
    time.sleep(cfg.work.sleep_s)
    if cfg.work.value == cfg.work.fail_on_value:
        raise RuntimeError(f"{FAIL_SENTINEL} (value={cfg.work.value})")
    record = {"folder_id": entry.folder_id, "hash": entry.hash, "host": socket.gethostname(),
              "task": os.environ.get("SLURM_ARRAY_TASK_ID", "local"), "started": started,
              "finished": time.time()}
    (folder / SEEN).mkdir(exist_ok=True)
    (folder / SEEN / f"{record['task']}.json").write_text(json.dumps(record))
    (folder / DONE).write_text(json.dumps({"value": cfg.work.value, **record}))


if __name__ == "__main__":
    main()
