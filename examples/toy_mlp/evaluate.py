"""Stage 2: evaluate a trained model on fresh test data.

The eval config carries the *same* ``data``/``model``/``trainer`` nodes as training plus an
``evaluation`` node. The trained folder is found by a read-only ledger lookup on the
training subset of the identity; the metrics land in this stage's own content-addressed
folder (namespace ``evaluate``), so a config evaluated twice is evaluated once.

    python evaluate.py -m sweep=hidden_sizes
    python evaluate.py -m sweep=hidden_sizes hydra.launcher.skip_finished=true
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from config_ledger import RunDecision
from config_ledger.hydra import ledger_from_config, ledger_settings
from config_ledger.hydra.run_status import SweepRunStatus
from toy_mlp import pipeline, schema
from train import TRAIN_NODES

log = logging.getLogger(__name__)
schema.register()


class EvaluateStatus(SweepRunStatus, namespace="evaluate"):
    @classmethod
    def check_run_status(cls, config, output_folder):
        if (Path(output_folder) / pipeline.METRICS).exists() and not config.overwrite:
            return RunDecision(False, "already evaluated")
        return RunDecision(True, "no metrics.json yet")


@hydra.main(config_path="conf", config_name="evaluate", version_base="1.3")
def main(cfg: DictConfig) -> None:
    ledger, namespace = ledger_from_config(cfg)
    train_cfg = OmegaConf.create({k: cfg[k] for k in TRAIN_NODES})
    trained = ledger.lookup_output_folder("train", Path(cfg.outputs) / "train", train_cfg)
    if trained is None or not (trained / pipeline.SUMMARY).exists():
        raise FileNotFoundError(
            "no completed training run for this config; run train.py with the same overrides first "
            "(or, if a config class's VERSION changed since, "
            "`config-ledger upgrade --namespace train`)"
        )
    folder, entry = ledger.resolve_output_folder(namespace, ledger_settings(cfg)["outputs"], cfg)
    decision = EvaluateStatus.check_run_status(cfg, folder)
    if not decision.launch:
        log.info("skipping: %s", decision.reason)
        return
    x, y = pipeline.make_data(cfg.evaluation.n_test, cfg.data.n_features, cfg.data.noise, cfg.evaluation.seed)
    pred = pipeline.predict(pipeline.load_params(trained), x, cfg.model.activation)
    metrics = {"test_mse": float(np.mean((pred - y) ** 2)), "trained_folder": trained.name}
    (folder / pipeline.METRICS).write_text(json.dumps(metrics, indent=2))
    log.info("identity %s -> %s: %s", entry.hash, folder, metrics)


if __name__ == "__main__":
    main()
