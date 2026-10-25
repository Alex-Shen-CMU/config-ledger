"""`config-ledger lookup --config-name`: compose with Hydra, single runs and sweeps."""

import json
import os
import sys
import textwrap

import pytest

pytest.importorskip("hydra")
from hydra.core.config_store import ConfigStore  # noqa: E402

from config_ledger.cli import main  # noqa: E402
from tests.conftest import make_config  # noqa: E402

LR = "method.trainer.learning_rate"

TRAIN = textwrap.dedent("""\
    defaults:
      - _self_
      - sweep: null
    method:
      _target_: tests.conftest.Method
      name: fixture
      num_workers: 4
    outputs: ${hydra:runtime.cwd}/outputs
    ledger:
      root: ${hydra:runtime.cwd}/ledger
      namespace: train
    """)
SWEEP = textwrap.dedent("""\
    # @package hydra
    sweeper:
      params:
        method.trainer.learning_rate: 0.1, 0.2
        method.trainer.seed: 0
    """)
SETUP = textwrap.dedent("""\
    from hydra.core.config_store import ConfigStore

    def register():
        ConfigStore.instance().store(name="trainer", node={"method": {"trainer": {
            "_target_": "tests.conftest.Trainer", "learning_rate": 0.1, "seed": 0,
            "layers": [8, 8], "device": "cpu"}}})
    """)


@pytest.fixture
def project(tmp_path, monkeypatch, ledger):
    """A Hydra app in tmp_path whose `ledger:` block points at the `ledger` fixture's root.
    `trainer` comes from a ConfigStore node registered by `app_setup:register`."""
    conf = tmp_path / "conf"
    (conf / "sweep").mkdir(parents=True)
    (conf / "train.yaml").write_text(TRAIN.replace("_self_", "trainer\n  - _self_", 1))
    (conf / "sweep" / "lr.yaml").write_text(SWEEP)
    (tmp_path / "app_setup.py").write_text(SETUP)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    sys.modules.pop("app_setup", None)
    yield tmp_path
    ConfigStore.instance().repo.pop("trainer.yaml", None)  # registered by app_setup


def _files(root):
    return sorted(os.path.join(d, f) for d, _, fs in os.walk(root) if "__pycache__" not in d
                  for f in fs)


def _cli(*args):
    return main(["lookup", "--config-dir", "conf", "--config-name", "train",
                 "--setup", "app_setup:register", *args])


def _run(ledger, tmp_path, **fields):
    return ledger.resolve_output_folder("train", tmp_path / "out", make_config(**fields))


def test_single_config_reads_ledger_block(project, ledger, capsys):
    _, entry = _run(ledger, project)
    assert _cli() == 0
    assert capsys.readouterr().out.strip() == entry.folder_id
    assert _cli(f"{LR}=0.5") == 1
    assert "not found" in capsys.readouterr().err


def test_ledger_and_namespace_flags_override_the_block(project, ledger, capsys):
    _run(ledger, project)
    assert _cli("--namespace", "evaluate") == 1
    other = project / "other"
    assert _cli("--ledger", str(other)) == 1
    assert not other.exists()


def test_sweep_group_grid(project, ledger, capsys):
    _, entry = _run(ledger, project, learning_rate=0.2)
    before = _files(project)
    assert _cli("-m", "sweep=lr") == 1  # one of two grid points is missing
    out = capsys.readouterr().out.splitlines()
    assert [line.split()[0] for line in out] == ["MISSING", entry.folder_id]
    assert out[1].split()[1:] == [f"{LR}=0.2", "method.trainer.seed=0", "sweep=lr"]
    assert _files(project) == before, "lookup must not write"
    _run(ledger, project, learning_rate=0.1)
    assert _cli("-m", "sweep=lr") == 0


def test_command_line_sweep_json(project, ledger, capsys):
    _, entry = _run(ledger, project, seed=1)
    assert _cli("-m", f"{LR}=0.1", "method.trainer.seed=0,1", "--json", "--parent", "out") == 1
    payload = json.loads(capsys.readouterr().out)
    assert [p["overrides"] for p in payload] == [
        [f"{LR}=0.1", "method.trainer.seed=0"], [f"{LR}=0.1", "method.trainer.seed=1"]]
    assert [p["folder_id"] for p in payload] == [None, entry.folder_id]
    assert payload[1]["output_folder"] == os.path.join("out", entry.folder_id)


def test_sweep_override_without_multirun_is_an_error(project, capsys):
    assert _cli("method.trainer.seed=0,1") == 2
    assert "multirun" in capsys.readouterr().err.lower()


def test_without_setup_configstore_nodes_are_missing(project, capsys):
    assert main(["lookup", "--config-dir", "conf", "--config-name", "train"]) == 2
    assert "trainer" in capsys.readouterr().err


def test_bump_warning_is_reported_once_per_sweep(project, ledger, capsys, monkeypatch):
    from tests.conftest import Trainer

    _run(ledger, project)
    monkeypatch.setattr(Trainer, "VERSION", 2)
    assert _cli("-m", "sweep=lr") == 1
    err = capsys.readouterr().err
    assert err.count("warning: ") == 1, "deduplicated across grid points"
    assert "Trainer @ Trainer:1 (live: Trainer:2)" in err
