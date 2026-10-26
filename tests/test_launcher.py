import os
import sys

import pytest

pytest.importorskip("hydra")
pytest.importorskip("hydra_plugins.hydra_submitit_launcher")
from config_ledger.hydra.launcher import script_dir_on_pythonpath  # noqa: E402


def test_script_dir_is_prepended_then_restored(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path[1:]])
    monkeypatch.setenv("PYTHONPATH", "/elsewhere")
    with script_dir_on_pythonpath():
        assert os.environ["PYTHONPATH"] == os.pathsep.join([str(tmp_path), "/elsewhere"])
    assert os.environ["PYTHONPATH"] == "/elsewhere"


def test_unset_pythonpath_is_unset_again(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path[1:]])
    monkeypatch.delenv("PYTHONPATH", raising=False)
    with script_dir_on_pythonpath():
        assert os.environ["PYTHONPATH"] == str(tmp_path)
    assert "PYTHONPATH" not in os.environ


def test_empty_path_entry_means_cwd(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "path", ["", *sys.path[1:]])
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    with script_dir_on_pythonpath():
        assert os.environ["PYTHONPATH"] == str(tmp_path)


def test_restored_when_submission_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path[1:]])
    monkeypatch.setenv("PYTHONPATH", "/elsewhere")
    with pytest.raises(RuntimeError), script_dir_on_pythonpath():
        raise RuntimeError
    assert os.environ["PYTHONPATH"] == "/elsewhere"
