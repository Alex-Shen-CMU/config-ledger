"""Stage 1: train one toy MLP per config; content-addressed output folder.

    python train.py                                   # one run, default config
    python train.py model.hidden=32                   # a different identity
    python train.py -m sweep=hidden_sizes             # the sweep named by conf/sweep/hidden_sizes.yaml
    python train.py -m sweep=hidden_sizes hydra.launcher.skip_finished=true   # only unfinished

Re-running a config that already has ``summary.json`` is a no-op (completion is judged from
the artifact, not from the ledger); a run killed mid-way resumes from its checkpoint.
:class:`TrainStatus` declares that rule once: the script calls it, and with ``skip_finished``
the launcher calls it before submitting so finished grid points are never launched.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from config_ledger import RunDecision
from config_ledger.hydra import ledger_from_config, ledger_settings
from config_ledger.hydra.run_status import SweepRunStatus
from toy_mlp import pipeline, schema

log = logging.getLogger(__name__)
schema.register()

TRAIN_NODES = ("data", "model", "trainer")


class TrainStatus(SweepRunStatus, namespace="train"):
    @classmethod
    def check_run_status(cls, config, output_folder):
        if (Path(output_folder) / pipeline.SUMMARY).exists() and not config.overwrite:
            return RunDecision(False, "already complete (pass overwrite=true to redo)")
        return RunDecision(True, "no summary.json yet")


@hydra.main(config_path="conf", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    ledger, namespace = ledger_from_config(cfg)
    folder, entry = ledger.resolve_output_folder(namespace, ledger_settings(cfg)["outputs"], cfg)
    log.info("identity %s -> %s", entry.hash, folder)
    decision = TrainStatus.check_run_status(cfg, folder)
    if not decision.launch:
        log.info("skipping: %s", decision.reason)
        return
    x, y = pipeline.make_data(cfg.data.n_samples, cfg.data.n_features, cfg.data.noise, cfg.data.seed)
    summary = pipeline.train(
        folder, x, y, hidden=cfg.model.hidden, activation=cfg.model.activation, epochs=cfg.trainer.epochs,
        learning_rate=cfg.trainer.learning_rate, seed=cfg.trainer.seed,
        checkpoint_every=cfg.trainer.checkpoint_every,
    )
    log.info("done: %s", summary)


if __name__ == "__main__":
    main()
