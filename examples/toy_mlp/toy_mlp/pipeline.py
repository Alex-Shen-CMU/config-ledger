"""The compute: synthetic regression data, a one-hidden-layer numpy MLP, resumable training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CHECKPOINT = "checkpoint.npz"
SUMMARY = "summary.json"
METRICS = "metrics.json"


def make_data(n_samples: int, n_features: int, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_samples, n_features))
    y = np.sin(x[:, :1]) + 0.5 * x[:, 1:2] ** 2 + noise * rng.normal(size=(n_samples, 1))
    return x, y


def _act(name: str):
    return {"tanh": (np.tanh, lambda a: 1 - a**2), "relu": (lambda z: np.maximum(z, 0), lambda a: (a > 0) * 1.0)}[name]


def init_params(n_features: int, hidden: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "w1": rng.normal(scale=1 / np.sqrt(n_features), size=(n_features, hidden)),
        "b1": np.zeros(hidden),
        "w2": rng.normal(scale=1 / np.sqrt(hidden), size=(hidden, 1)),
        "b2": np.zeros(1),
    }


def predict(params: dict[str, np.ndarray], x: np.ndarray, activation: str) -> np.ndarray:
    f, _ = _act(activation)
    return f(x @ params["w1"] + params["b1"]) @ params["w2"] + params["b2"]


def train(folder: Path, x: np.ndarray, y: np.ndarray, *, hidden: int, activation: str, epochs: int,
          learning_rate: float, seed: int, checkpoint_every: int) -> dict:
    """Full-batch gradient descent, resuming from ``checkpoint.npz`` if present.

    Writes ``summary.json`` when done; its presence is what "this run is complete" means.
    """
    f, df = _act(activation)
    ckpt = folder / CHECKPOINT
    if ckpt.exists():
        saved = np.load(ckpt)
        params = {k: saved[k] for k in ("w1", "b1", "w2", "b2")}
        start, losses = int(saved["epoch"]), list(saved["losses"])
    else:
        params, start, losses = init_params(x.shape[1], hidden, seed), 0, []
    for epoch in range(start, epochs):
        h = f(x @ params["w1"] + params["b1"])
        out = h @ params["w2"] + params["b2"]
        err = out - y
        losses.append(float(np.mean(err**2)))
        g_out = 2 * err / len(x)
        g_w2, g_b2 = h.T @ g_out, g_out.sum(0)
        g_h = (g_out @ params["w2"].T) * df(h)
        g_w1, g_b1 = x.T @ g_h, g_h.sum(0)
        for k, g in (("w1", g_w1), ("b1", g_b1), ("w2", g_w2), ("b2", g_b2)):
            params[k] = params[k] - learning_rate * g
        if (epoch + 1) % checkpoint_every == 0:
            np.savez(ckpt, epoch=epoch + 1, losses=np.array(losses), **params)
    np.savez(ckpt, epoch=epochs, losses=np.array(losses), **params)
    summary = {"final_train_loss": losses[-1], "epochs": epochs}
    (folder / SUMMARY).write_text(json.dumps(summary, indent=2))
    return summary


def load_params(folder: Path) -> dict[str, np.ndarray]:
    saved = np.load(folder / CHECKPOINT)
    return {k: saved[k] for k in ("w1", "b1", "w2", "b2")}
