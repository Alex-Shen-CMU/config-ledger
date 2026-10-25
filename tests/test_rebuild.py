"""`config-ledger rebuild`: sidecars and version index from run manifests."""

import json
import shutil

import pytest

from config_ledger import maintenance as m
from config_ledger.cli import main
from config_ledger.store import RUN_MANIFEST_FILENAME
from config_ledger.versions import FRESH, MIGRATED, STALE
from tests.conftest import make_config
from tests.test_maintenance import _run, _statuses


def _rebuild(ledger, *roots, apply=False):
    args = ["rebuild", "--ledger", str(ledger.root)]
    for root in roots:
        args += ["--root", str(root)]
    return main([*args, "--apply"] if apply else args)


def _sidecars(ledger):
    """Every sidecar, without its creation time."""
    found = {}
    for path in sorted(ledger.root.glob("*/??/*.json")):
        stored = json.loads(path.read_text())
        stored.pop("created")
        found[str(path.relative_to(ledger.root))] = stored
    return found


def _manifest(folder):
    return json.loads((folder / RUN_MANIFEST_FILENAME).read_text())


def _write_manifest(folder, manifest):
    (folder / RUN_MANIFEST_FILENAME).write_text(json.dumps(manifest))


def test_run_manifest_keeps_every_hash_row_and_tree(ledger, tmp_path, bumped):
    folder, old = _run(ledger, tmp_path, learning_rate=0.05)
    bumped()
    m.upgrade(ledger, apply=True)
    manifest = _manifest(folder)
    first, latest = manifest["config_hashes"]
    assert first == old.hash
    assert manifest["configs"][first]["config"] == old.materialized
    assert manifest["configs"][latest]["tree"] == manifest["tree"]
    assert manifest["configs"][first]["tree"]["method"]["trainer"]["learning_rate"] == 0.05


def test_rebuild_into_an_empty_ledger_round_trips(ledger, tmp_path, bumped, capsys):
    """Several hashes -> one folder after an upgrade, plus a plain run: the rebuilt sidecars and
    index equal the originals, with the upgraded-from version restored as ``migrated``."""
    _, upgraded = _run(ledger, tmp_path, learning_rate=0.05)
    bumped()
    m.upgrade(ledger, apply=True)
    _, plain = _run(ledger, tmp_path, seed=3)
    sidecars = _sidecars(ledger)
    statuses = {f: _statuses(ledger, f) for f in (upgraded.folder_id, plain.folder_id)}
    assert len(sidecars) == 3
    assert statuses[upgraded.folder_id][("Trainer", "Trainer:1")] == MIGRATED

    shutil.rmtree(ledger.root)
    assert _rebuild(ledger, tmp_path / "out") == 0
    out = capsys.readouterr().out
    assert "3 WOULD-CREATE" in out and "Dry run." in out
    assert "Trainer @ Trainer:1 [migrated]" in out
    assert not ledger.root.exists()

    assert _rebuild(ledger, tmp_path / "out", apply=True) == 0
    assert "3 CREATED" in capsys.readouterr().out
    assert _sidecars(ledger) == sidecars
    assert {f: _statuses(ledger, f) for f in statuses} == statuses

    assert _rebuild(ledger, tmp_path / "out") == 0
    out = capsys.readouterr().out
    assert "nothing to do" in out and "5 sidecar(s) / index record(s) already current" in out
    # The rebuilt ledger works: the same configs land in the same folders.
    _, again = _run(ledger, tmp_path, seed=3)
    assert again.folder_id == plain.folder_id


def test_rebuild_fills_in_a_partly_lost_ledger(ledger, tmp_path):
    _, entry = _run(ledger, tmp_path)
    ledger.sidecar_path("train", entry.hash).unlink()
    outcomes = m.rebuild(ledger, [tmp_path / "out"], apply=True)
    assert m.summarize(outcomes) == {m.CREATED: 1, m.CURRENT: 1}
    assert ledger.find("train", ledger.identity(make_config())).folder_id == entry.folder_id


def test_stale_status_comes_back_fresh(ledger, tmp_path, bumped):
    _, entry = _run(ledger, tmp_path)
    bumped()
    m.mark_stale(ledger, apply=True)
    assert _statuses(ledger, entry.folder_id)[("Trainer", "Trainer:1")] == STALE
    shutil.rmtree(ledger.root / "train" / "_versions")
    m.rebuild(ledger, [tmp_path / "out"], apply=True)
    assert _statuses(ledger, entry.folder_id)[("Trainer", "Trainer:1")] == FRESH


def test_existing_sidecar_of_another_folder_is_a_conflict(ledger, tmp_path, capsys):
    folder, entry = _run(ledger, tmp_path)
    shutil.rmtree(ledger.root / "train" / "_versions")
    sidecar = ledger.sidecar_path("train", entry.hash)
    stored = json.loads(sidecar.read_text())
    sidecar.write_text(json.dumps({**stored, "folder_id": "someone-else"}))

    assert _rebuild(ledger, tmp_path / "out", apply=True) == 1
    out = capsys.readouterr().out
    assert f"sidecar {entry.hash} belongs to folder someone-else" in out
    assert "not indexed until the conflict is resolved" in out
    assert json.loads(sidecar.read_text())["folder_id"] == "someone-else"  # not clobbered
    assert _statuses(ledger, entry.folder_id) == {}


def test_two_manifests_claiming_one_hash_write_neither(ledger, tmp_path, capsys):
    folder, entry = _run(ledger, tmp_path)
    copy = tmp_path / "out" / "copy-id"
    shutil.copytree(folder, copy)
    _write_manifest(copy, {**_manifest(copy), "folder_id": "copy-id"})
    shutil.rmtree(ledger.root)

    assert _rebuild(ledger, tmp_path / "out", apply=True) == 1
    out = capsys.readouterr().out
    assert f"also claimed by the run manifest of folder(s) {entry.folder_id}" in out
    assert "also claimed by the run manifest of folder(s) copy-id" in out
    assert not ledger.sidecar_path("train", entry.hash).exists()


def test_same_folder_id_at_two_paths_fails(ledger, tmp_path):
    folder, entry = _run(ledger, tmp_path)
    shutil.copytree(folder, tmp_path / "elsewhere" / entry.folder_id)
    outcomes = m.rebuild(ledger, [tmp_path / "out", tmp_path / "elsewhere"])
    assert [o.status for o in outcomes] == [m.FAILED, m.FAILED]
    assert "the same folder_id was found at" in outcomes[0].detail


def test_renamed_folder_fails_and_is_skipped(ledger, tmp_path, capsys):
    folder, entry = _run(ledger, tmp_path)
    folder.rename(folder.with_name("renamed"))
    shutil.rmtree(ledger.root)
    assert _rebuild(ledger, tmp_path / "out", apply=True) == 1
    assert f"named 'renamed' but its manifest says folder_id '{entry.folder_id}'" in (
        capsys.readouterr().out)
    assert not ledger.root.exists()


@pytest.mark.parametrize("damage", ["tampered", "missing"])
def test_manifest_rows_must_rehash_to_their_hashes(ledger, tmp_path, damage):
    folder, entry = _run(ledger, tmp_path)
    manifest = _manifest(folder)
    if damage == "tampered":
        manifest["configs"][entry.hash]["config"]["Method.name"] = "other"
    else:
        del manifest["configs"]
    _write_manifest(folder, manifest)
    (outcome,) = m.rebuild(ledger, [tmp_path / "out"], apply=True)
    assert outcome.status == m.FAILED
    assert ("hashes to" if damage == "tampered" else "no stored row") in outcome.detail


def test_moved_outputs_relocate_index_entries(ledger, tmp_path, bumped, capsys):
    folder, entry = _run(ledger, tmp_path)
    shutil.move(tmp_path / "out", tmp_path / "moved")
    new_folder = (tmp_path / "moved" / entry.folder_id).resolve()
    assert _rebuild(ledger, tmp_path / "moved") == 0
    assert f"pointed at {folder.resolve()}" in capsys.readouterr().out
    assert _rebuild(ledger, tmp_path / "moved", apply=True) == 0
    assert "1 MOVED" in capsys.readouterr().out
    entries = ledger.versions("train").folder_entries(entry.folder_id)
    assert {e.output_folder for e in entries} == {str(new_folder)}
    # purge-stale now deletes the folder where it actually is.
    bumped()
    m.mark_stale(ledger, apply=True)
    m.purge_stale(ledger, apply=True)
    assert not new_folder.exists()


def test_orphans_are_reported_not_deleted(ledger, tmp_path, capsys):
    folder, entry = _run(ledger, tmp_path)
    _, kept = _run(ledger, tmp_path, seed=1)
    shutil.rmtree(folder)
    assert _rebuild(ledger, tmp_path / "out", apply=True) == 0
    out = capsys.readouterr().out
    assert f"ORPHAN        train/{entry.folder_id}" in out
    assert "1 sidecar(s), 2 index entr(ies)" in out
    assert ledger.sidecar_path("train", entry.hash).exists()
    assert len(ledger.versions("train").folder_entries(entry.folder_id)) == 2


def test_namespace_filter(ledger, tmp_path):
    _run(ledger, tmp_path)
    ledger.resolve_output_folder("evaluate", tmp_path / "out", make_config(seed=5))
    shutil.rmtree(ledger.root)
    outcomes = m.rebuild(ledger, [tmp_path / "out"], namespaces=["evaluate"], apply=True)
    assert {o.namespace for o in outcomes} == {"evaluate"}
    assert not (ledger.root / "train").exists()
