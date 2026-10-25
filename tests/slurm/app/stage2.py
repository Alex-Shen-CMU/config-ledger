"""Stage 2: read-only lookup of stage 1's folder for the same ``work`` node; fails unless
stage 1 already finished there, so it only succeeds if Slurm held it until stage 1 was done."""

from __future__ import annotations

import json
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from slurm_app import DONE, schema

from config_ledger.hydra import ledger_from_config

schema.register()


@hydra.main(config_path="conf", config_name="stage2", version_base="1.3")
def main(cfg: DictConfig) -> None:
    started = time.time()
    ledger, namespace = ledger_from_config(cfg)
    upstream = ledger.lookup_output_folder("stage1", Path(cfg.outputs) / "stage1",
                                           OmegaConf.create({"work": cfg.work}))
    if upstream is None or not (upstream / DONE).exists():
        raise FileNotFoundError(f"stage 1 has not finished for work.value={cfg.work.value}")
    stage1 = json.loads((upstream / DONE).read_text())
    folder, entry = ledger.resolve_output_folder(namespace, Path(cfg.outputs) / namespace, cfg)
    (folder / DONE).write_text(json.dumps({
        "value": cfg.work.value, "started": started, "stage1_finished": stage1["finished"],
        "stage1_folder": upstream.name, "result": cfg.work.value * cfg.check.scale,
    }))


if __name__ == "__main__":
    main()
