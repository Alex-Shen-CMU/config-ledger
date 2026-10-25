"""`config-ledger lookup`: config -> folder, read-only, with version diagnostics."""

import json
import os
import sys

import omegaconf
import pytest

from config_ledger.cli import main
from config_ledger.lookup import load_config_file, lookup
from tests.conftest import Trainer, make_config


@pytest.fixture(autouse=True)
def _restore_sys_path(monkeypatch):
    """The CLI puts cwd and --pythonpath on sys.path."""
    monkeypatch.setattr(sys, "path", list(sys.path))


def _run(ledger, tmp_path, **fields):
    return ledger.resolve_output_folder("train", tmp_path / "out", make_config(**fields))


def _files(root):
    return sorted(os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs)


def _write(tmp_path, config, name="config.json"):
    path = tmp_path / name
    if name.endswith(".json"):
        path.write_text(json.dumps(omegaconf.OmegaConf.to_container(config)))
    else:
        omegaconf.OmegaConf.save(config, path)
    return path


def _cli(ledger, *args):
    return main(["lookup", "--ledger", str(ledger.root), "--namespace", "train", *map(str, args)])


def test_hit_and_miss(ledger, tmp_path):
    _, entry = _run(ledger, tmp_path)
    hit = lookup(ledger, "train", make_config())
    assert (hit.found, hit.entry, hit.warnings) == (True, entry, [])
    miss = lookup(ledger, "train", make_config(seed=7))
    assert (miss.found, miss.entry, miss.warnings) == (False, None, [])


def test_lookup_never_writes(ledger, tmp_path, capsys):
    _run(ledger, tmp_path)
    before = _files(tmp_path)
    assert _cli(ledger, _write(tmp_path, make_config(seed=7))) == 1
    assert _files(tmp_path) == sorted([*before, str(tmp_path / "config.json")])


@pytest.mark.parametrize("name", ["config.json", "config.yaml", "config.yml"])
def test_cli_prints_folder_id_or_path(ledger, tmp_path, capsys, name):
    _, entry = _run(ledger, tmp_path)
    path = _write(tmp_path, make_config(), name)
    assert _cli(ledger, path) == 0
    assert capsys.readouterr().out.strip() == entry.folder_id
    assert _cli(ledger, path, "--parent", tmp_path / "out") == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "out" / entry.folder_id)


def test_cli_miss_exits_1(ledger, tmp_path, capsys):
    _run(ledger, tmp_path)
    assert _cli(ledger, _write(tmp_path, make_config(seed=7))) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "not found: no folder owns this config in namespace 'train'" in captured.err


def test_cli_json(ledger, tmp_path, capsys):
    _, entry = _run(ledger, tmp_path)
    assert _cli(ledger, _write(tmp_path, make_config()), "--json", "--parent", "out") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"namespace": "train", "hash": entry.hash, "found": True,
                       "folder_id": entry.folder_id, "created": entry.created,
                       "output_folder": os.path.join("out", entry.folder_id), "warnings": []}
    assert _cli(ledger, _write(tmp_path, make_config(seed=7)), "--json") == 1
    payload = json.loads(capsys.readouterr().out)
    assert (payload["found"], payload["folder_id"], payload["output_folder"]) == (False, None, None)


@pytest.mark.parametrize("args, message", [
    ((), "one CONFIG file"),
    (("a.json", "b.json"), "one CONFIG file"),
    (("c.toml",), "unsupported config file 'c.toml'"),
    (("c.json", "-m"), "need --config-name"),
])
def test_cli_usage_errors_exit_2(ledger, tmp_path, capsys, monkeypatch, args, message):
    (tmp_path / "c.toml").write_text("")
    monkeypatch.chdir(tmp_path)
    assert _cli(ledger, *args) == 2
    assert message in capsys.readouterr().err


def test_bump_miss_points_at_old_version(ledger, tmp_path, monkeypatch, capsys):
    """After a version bump the live hash is new: say the run may exist under the old version."""
    _run(ledger, tmp_path)
    monkeypatch.setattr(Trainer, "VERSION", 2)
    result = lookup(ledger, "train", make_config())
    assert not result.found
    (warning,) = result.warnings
    assert "Trainer @ Trainer:1 (live: Trainer:2) has 1 fresh and 0 stale run(s)" in warning
    assert "config-ledger upgrade" in warning
    assert _cli(ledger, _write(tmp_path, make_config())) == 1
    assert "warning: " in capsys.readouterr().err


def test_old_version_is_not_reported_on_a_hit(ledger, tmp_path, monkeypatch):
    _run(ledger, tmp_path)
    monkeypatch.setattr(Trainer, "VERSION", 2)
    (old,) = ledger.versions("train").entries(target=Trainer._target_)
    ledger.versions("train").move(old, "stale")
    _run(ledger, tmp_path)
    assert lookup(ledger, "train", make_config()).warnings == []


def test_newer_version_warns_even_on_a_hit(ledger, tmp_path, monkeypatch):
    _run(ledger, tmp_path)
    monkeypatch.setattr(Trainer, "VERSION", 2)
    (old,) = ledger.versions("train").entries(target=Trainer._target_)
    ledger.versions("train").move(old, "stale")
    _run(ledger, tmp_path)
    monkeypatch.setattr(Trainer, "VERSION", 1)  # back to an old checkout
    result = lookup(ledger, "train", make_config())
    assert result.found
    (warning,) = result.warnings
    assert "Trainer @ Trainer:2 is newer than this code (Trainer:1)" in warning


def test_schema_change_warns(ledger, tmp_path, monkeypatch):
    import config_ledger.versions as versions

    _run(ledger, tmp_path)
    real = versions.schema_fields
    monkeypatch.setattr(versions, "schema_fields",
                        lambda cls: (*real(cls), "momentum") if cls is Trainer else real(cls))
    (warning,) = lookup(ledger, "train", make_config()).warnings
    assert "added ['momentum']" in warning


def test_hydra_dump_with_runtime_interpolations(ledger, tmp_path, capsys):
    """A `.hydra/config.yaml` keeps `${hydra:...}` in top-level execution-control keys, which
    cannot resolve outside a job; nested interpolations in identity nodes still resolve."""
    _, entry = _run(ledger, tmp_path, learning_rate=0.3)
    config = make_config()
    config.lr = 0.3
    config.method.trainer.learning_rate = "${lr}"
    config.outputs = "${hydra:runtime.cwd}/outputs"
    path = _write(tmp_path, config, "config.yaml")
    assert lookup(ledger, "train", load_config_file(path)).entry == entry
    assert _cli(ledger, path) == 0
    assert capsys.readouterr().out.strip() == entry.folder_id


def test_yaml_unresolvable_nested_interpolation_still_raises(tmp_path, ledger):
    config = make_config()
    config.method.trainer.learning_rate = "${no_such_resolver:x}"
    path = _write(tmp_path, config, "config.yaml")
    with pytest.raises(omegaconf.errors.UnsupportedInterpolationType):
        lookup(ledger, "train", load_config_file(path))


def test_cwd_and_pythonpath_are_importable(ledger, tmp_path, capsys, monkeypatch):
    """`_target_` classes of the project import from the current directory or --pythonpath."""
    pkg = tmp_path / "proj"
    pkg.mkdir()
    (pkg / "proj_schema.py").write_text(
        "from dataclasses import dataclass\n"
        "from typing import ClassVar\n"
        "from config_ledger import VersionedConfig\n"
        "@dataclass\n"
        "class Opt(VersionedConfig):\n"
        "    VERSION: ClassVar[int] = 1\n"
        "    lr: float = 0.1\n")
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"opt": {"_target_": "proj_schema.Opt"}}))
    assert _cli(ledger, path) == 2  # not importable from tmp_path
    assert "proj_schema" in capsys.readouterr().err
    assert _cli(ledger, path, "--pythonpath", pkg) == 1
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != str(pkg)])
    sys.modules.pop("proj_schema", None)
    monkeypatch.chdir(pkg)
    assert _cli(ledger, path) == 1
