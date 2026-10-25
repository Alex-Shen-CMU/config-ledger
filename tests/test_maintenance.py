"""CLI and maintenance operations on the version index."""

import json
import shutil

import pytest

from config_ledger import StaleRunsError, version_chain
from config_ledger import maintenance as m
from config_ledger.cli import main
from config_ledger.identity import VERSION_KEY
from config_ledger.store import RUN_MANIFEST_FILENAME
from config_ledger.versions import FRESH, MIGRATED, STALE
from tests.conftest import Method, MiningGoldTrainer, Trainer, make_config


def _run(ledger, tmp_path, **fields):
    return ledger.resolve_output_folder("train", tmp_path / "out", make_config(**fields))


def _statuses(ledger, folder_id):
    return {(e.target.rsplit(".", 1)[-1], e.version_chain): e.status
            for e in ledger.versions("train").folder_entries(folder_id)}


def _cli(ledger, *args):
    return main([*args[:1], "--ledger", str(ledger.root), *args[1:]])


# --- versions report ------------------------------------------------------------------------


def test_versions_report(ledger, tmp_path, bumped, capsys):
    _run(ledger, tmp_path)
    bumped()
    assert _cli(ledger, "versions") == 0
    out = capsys.readouterr().out
    assert f"older             1      0      0  {Trainer._target_} @ Trainer:1" in out
    assert f"live              1      0      0  {Method._target_} @ Method:1" in out
    assert "1 old version(s) still have fresh runs" in out


# --- mark-stale -------------------------------------------------------------------------


def test_mark_stale_dry_run_then_apply(ledger, tmp_path, bumped, capsys):
    _, entry = _run(ledger, tmp_path)
    bumped()
    assert _cli(ledger, "mark-stale") == 0
    assert "WOULD-MARK" in capsys.readouterr().out
    assert _statuses(ledger, entry.folder_id)[("Trainer", "Trainer:1")] == FRESH
    assert _cli(ledger, "mark-stale", "--apply") == 0
    # Status belongs to the config: only the Trainer:1 entry moves, Method:1 stays fresh.
    assert _statuses(ledger, entry.folder_id) == {
        ("Trainer", "Trainer:1"): STALE, ("Method", "Method:1"): FRESH,
    }
    _run(ledger, tmp_path)  # no longer blocked


def test_mark_stale_refuses_the_live_version(ledger, tmp_path, capsys):
    _, entry = _run(ledger, tmp_path)
    assert _cli(ledger, "mark-stale", "--target", "Trainer", "--version", "1", "--apply") == 1
    assert "live version; rerun it with overwrite=true" in capsys.readouterr().out
    assert _statuses(ledger, entry.folder_id)[("Trainer", "Trainer:1")] == FRESH


def test_selection_by_class_name_and_version(ledger, tmp_path, monkeypatch):
    _, plain = _run(ledger, tmp_path)
    _, gold = _run(ledger, tmp_path, trainer_target=MiningGoldTrainer._target_, alpha=0.7)
    monkeypatch.setattr(Trainer, "VERSION", 2)
    marked = m.mark_stale(ledger, m.Selection(targets=["MiningGoldTrainer"]), apply=True)
    assert [o.folder_id for o in marked] == [gold.folder_id]
    selection = m.Selection(targets=[Trainer._target_], versions=["Trainer:1"])
    marked = m.mark_stale(ledger, selection, apply=True)
    assert [o.folder_id for o in marked] == [plain.folder_id]


# --- upgrade ------------------------------------------------------------------------------


def test_bump_error_upgrade_rerun_is_a_noop(ledger, tmp_path, bumped, capsys):
    """bump -> new runs refused -> upgrade -> the same config lands in the old folder."""
    folder, old = _run(ledger, tmp_path, learning_rate=0.05)
    (folder / "result.txt").write_text("done")
    bumped()
    upgraded_cfg = make_config(learning_rate=5.0)  # what the old run means under Trainer:2
    with pytest.raises(StaleRunsError):
        ledger.resolve_output_folder("train", tmp_path / "out", upgraded_cfg)

    assert _cli(ledger, "upgrade") == 0
    assert "WOULD-UPGRADE" in capsys.readouterr().out
    assert _cli(ledger, "upgrade", "--apply") == 0
    assert "UPGRADED" in capsys.readouterr().out

    assert _statuses(ledger, old.folder_id) == {
        ("Trainer", "Trainer:1"): MIGRATED, ("Trainer", "Trainer:2"): FRESH,
        ("Method", "Method:1"): FRESH,
    }
    again, entry = ledger.resolve_output_folder("train", tmp_path / "out", upgraded_cfg)
    assert again == folder and (again / "result.txt").read_text() == "done"
    manifest = json.loads((folder / RUN_MANIFEST_FILENAME).read_text())
    assert manifest["config_hashes"] == [old.hash, entry.hash]
    assert manifest["tree"]["method"]["trainer"]["learning_rate"] == 5.0
    assert m.upgrade(ledger) == []  # nothing left to do


def test_upgrade_failure_is_logged_and_status_unchanged(ledger, tmp_path, bumped, capsys):
    _, entry = _run(ledger, tmp_path)
    bumped(upgradable=False)
    assert _cli(ledger, "upgrade", "--apply") == 0
    out = capsys.readouterr().out
    assert "FAILED" in out and "Trainer defines no upgrade from 'Trainer:1'" in out
    assert "because: " + Trainer._target_ + " @ Trainer:1 [fresh] at Method.trainer" in out
    assert _statuses(ledger, entry.folder_id)[("Trainer", "Trainer:1")] == FRESH


def test_upgrade_conflict_when_the_new_config_was_already_run(ledger, tmp_path, bumped):
    _, old = _run(ledger, tmp_path, learning_rate=0.05)
    bumped()
    index = ledger.versions("train")
    (old_entry,) = index.entries(target=Trainer._target_, chain="Trainer:1")
    stale = index.move(old_entry, STALE)  # out of the way, so the v2 config can run first
    _, rerun = _run(ledger, tmp_path, learning_rate=5.0)
    index.move(stale, FRESH)
    (outcome,) = m.upgrade(ledger, apply=True)
    assert outcome.status == m.CONFLICT and rerun.folder_id in outcome.detail
    assert _statuses(ledger, old.folder_id)[("Trainer", "Trainer:1")] == FRESH


def test_upgrade_skips_runs_with_stale_configs_unless_asked(ledger, tmp_path, bumped, monkeypatch):
    _, entry = _run(ledger, tmp_path, learning_rate=0.05)
    bumped()
    monkeypatch.setattr(Method, "VERSION", 2)
    monkeypatch.setattr(Method, "upgrade_config", classmethod(lambda cls, old: (
        None if old[VERSION_KEY] == version_chain(cls) else dict(old))))
    m.mark_stale(ledger, m.Selection(targets=["Method"]), apply=True)
    (outcome,) = m.upgrade(ledger, apply=True)
    assert outcome.status == m.SKIPPED and "Method:1 [stale]" in outcome.detail
    (outcome,) = m.upgrade(ledger, apply=True, include_stale=True)
    assert outcome.status == m.UPGRADED
    assert _statuses(ledger, entry.folder_id) == {
        ("Trainer", "Trainer:1"): MIGRATED, ("Method", "Method:1"): MIGRATED,
        ("Trainer", "Trainer:2"): FRESH, ("Method", "Method:2"): FRESH,
    }


def test_upgrade_cli_include_stale_flag(ledger, tmp_path, bumped, capsys):
    _run(ledger, tmp_path)
    bumped()
    assert _cli(ledger, "upgrade", "--include-stale") == 0
    assert "WOULD-UPGRADE" in capsys.readouterr().out


# --- purge-stale ------------------------------------------------------------------------


def test_purge_stale_dry_run_names_the_cause_then_deletes_all(ledger, tmp_path, bumped, capsys):
    folder, entry = _run(ledger, tmp_path)
    _, keep = _run(ledger, tmp_path, trainer_target=MiningGoldTrainer._target_, alpha=0.7)
    bumped(upgradable=False)
    m.mark_stale(ledger, m.Selection(targets=["Trainer"]), apply=True)

    assert _cli(ledger, "purge-stale") == 0
    out = capsys.readouterr().out
    assert f"WOULD-PURGE   train/{entry.folder_id}" in out
    assert f"because: {Trainer._target_} @ Trainer:1 [stale] at Method.trainer" in out
    assert folder.exists()

    assert _cli(ledger, "purge-stale", "--apply") == 0
    assert not folder.exists()
    assert not ledger.sidecar_path("train", entry.hash).exists()
    # Every entry goes, including the run's still-fresh Method:1 entry.
    assert ledger.versions("train").folder_entries(entry.folder_id) == []
    assert ledger.versions("train").folder_entries(keep.folder_id) != []


def test_purge_finishes_when_the_folder_is_already_gone(ledger, tmp_path, bumped):
    folder, entry = _run(ledger, tmp_path)
    bumped()
    m.mark_stale(ledger, apply=True)
    shutil.rmtree(folder)
    (outcome,) = m.purge_stale(ledger, apply=True)
    assert "removed 1 sidecar(s)" in outcome.detail
    assert not ledger.sidecar_path("train", entry.hash).exists()
