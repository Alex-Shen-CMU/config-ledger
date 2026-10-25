"""The version index: layout, lazy creation, stale discovery, schema guard."""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from config_ledger import NewerVersionError, SchemaChangedError, StaleRunsError, skip_version_check
from config_ledger.versions import FRESH, MIGRATED, STALE, compare_chains, version_slug
from tests.conftest import Method, MiningGoldTrainer, Trainer, make_config


def _run(ledger, tmp_path, namespace="train", **fields):
    parent = tmp_path / "out" / namespace
    return ledger.resolve_output_folder(namespace, parent, make_config(**fields))


def test_run_is_recorded_once_per_class_and_version(ledger, tmp_path):
    folder, entry = _run(ledger, tmp_path)
    index = ledger.versions("train")
    assert {(v.target, v.version_chain): v.counts[FRESH] for v in index.versions()} == {
        (Method._target_, "Method:1"): 1,
        (Trainer._target_, "Trainer:1"): 1,
    }
    (trainer,) = index.entries(target=Trainer._target_)
    assert trainer.folder_id == entry.folder_id
    assert (trainer.status, trainer.nodes) == (FRESH, ("Method.trainer",))
    assert trainer.output_folder == str(folder.resolve())
    meta = json.loads((index.version_dir(Trainer._target_, "Trainer:1") / "meta.json").read_text())
    assert meta["fields"] == ["layers", "learning_rate", "seed"]  # `device` is execution control


def test_layout_uses_target_and_version_slug(ledger, tmp_path):
    _, entry = _run(ledger, tmp_path, trainer_target=MiningGoldTrainer._target_, alpha=0.7)
    chain = "Trainer:1|MiningGoldTrainer:2"
    assert version_slug(chain) == "Trainer-1+MiningGoldTrainer-2"
    target_dir = ledger.root / "train" / "_versions" / MiningGoldTrainer._target_
    path = target_dir / version_slug(chain) / "fresh"
    assert (path / f"{entry.folder_id}.json").exists()


def test_recording_is_idempotent_and_race_safe(ledger, tmp_path):
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda _: _run(ledger, tmp_path), range(16)))
    assert [v.counts[FRESH] for v in ledger.versions("train").versions()] == [1, 1]


def test_bump_blocks_new_runs_until_resolved(ledger, tmp_path, monkeypatch):
    _run(ledger, tmp_path)
    monkeypatch.setattr(Trainer, "VERSION", 2)
    match = r"Trainer @ Trainer:1: 1 fresh run\(s\) \(live: Trainer:2\)"
    with pytest.raises(StaleRunsError, match=match):
        _run(ledger, tmp_path, seed=5)
    assert not (ledger.root / "train" / "_versions" / Trainer._target_ / "Trainer-2").exists(), \
        "nothing may be claimed before the check"
    (old,) = ledger.versions("train").entries(target=Trainer._target_)
    ledger.versions("train").move(old, STALE)
    _run(ledger, tmp_path, seed=5)  # stale runs under an old version are allowed


def test_migrated_entries_do_not_block(ledger, tmp_path, monkeypatch):
    _run(ledger, tmp_path)
    index = ledger.versions("train")
    (old,) = index.entries(target=Trainer._target_)
    monkeypatch.setattr(Trainer, "VERSION", 2)
    index.move(old, MIGRATED)
    _run(ledger, tmp_path)


def test_base_class_bump_blocks_subclass(ledger, tmp_path, monkeypatch):
    _run(ledger, tmp_path, trainer_target=MiningGoldTrainer._target_, alpha=0.7)
    monkeypatch.setattr(Trainer, "VERSION", 2)
    with pytest.raises(StaleRunsError, match="MiningGoldTrainer"):
        _run(ledger, tmp_path, trainer_target=MiningGoldTrainer._target_, alpha=0.7)


def test_unrelated_class_is_not_blocked(ledger, tmp_path, monkeypatch):
    _run(ledger, tmp_path, trainer_target=MiningGoldTrainer._target_, alpha=0.7)
    monkeypatch.setattr(MiningGoldTrainer, "VERSION", 3)
    # Plain Trainer runs share Method but not MiningGoldTrainer.
    _run(ledger, tmp_path)


def test_namespaces_do_not_block_each_other(ledger, tmp_path, monkeypatch):
    _run(ledger, tmp_path, namespace="train")
    monkeypatch.setattr(Trainer, "VERSION", 2)
    _run(ledger, tmp_path, namespace="evaluate")


def test_newer_stored_version_is_its_own_error(ledger, tmp_path, monkeypatch):
    monkeypatch.setattr(Trainer, "VERSION", 2)
    _run(ledger, tmp_path)
    monkeypatch.setattr(Trainer, "VERSION", 1)
    with pytest.raises(NewerVersionError, match="old checkout"):
        _run(ledger, tmp_path)


def test_skip_version_check_still_records(ledger, tmp_path, monkeypatch):
    _run(ledger, tmp_path)
    monkeypatch.setattr(Trainer, "VERSION", 2)
    with skip_version_check():
        _run(ledger, tmp_path)
    chains = {v.version_chain for v in ledger.versions("train").versions(Trainer._target_)}
    assert chains == {"Trainer:1", "Trainer:2"}
    with pytest.raises(StaleRunsError):
        _run(ledger, tmp_path)


def test_schema_guard_refuses_field_change_without_bump(ledger, tmp_path, monkeypatch):
    _run(ledger, tmp_path)
    import config_ledger.versions as versions

    real = versions.schema_fields
    monkeypatch.setattr(versions, "schema_fields",
                        lambda cls: (*real(cls), "momentum") if cls is Trainer else real(cls))
    with pytest.raises(SchemaChangedError, match=r"added \['momentum'\].*still 'Trainer:1'"):
        _run(ledger, tmp_path)
    with pytest.raises(SchemaChangedError):
        ledger.check_configs("train", [make_config()])
    with skip_version_check(), pytest.raises(SchemaChangedError):  # sweep tasks are guarded too
        _run(ledger, tmp_path)
    monkeypatch.setattr(Trainer, "VERSION", 2)
    (old,) = ledger.versions("train").entries(target=Trainer._target_)
    ledger.versions("train").move(old, STALE)
    _run(ledger, tmp_path)  # bumped: a new map with the new field set


def test_lookup_is_never_blocked(ledger, tmp_path, monkeypatch):
    _run(ledger, tmp_path)
    monkeypatch.setattr(Trainer, "VERSION", 2)
    assert ledger.lookup_output_folder("train", tmp_path / "out" / "train", make_config()) is None


def test_compare_chains():
    assert compare_chains("A:1|B:2", "A:1|B:2") == "live"
    assert compare_chains("A:1|B:2", "A:2|B:2") == "older"
    assert compare_chains("A:2|B:1", "A:1|B:2") == "newer"
    assert compare_chains("B:1", "A:1|B:1") == "older"  # restructured hierarchy
