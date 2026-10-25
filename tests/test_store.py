import errno
import json
import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from config_ledger import _files, store
from config_ledger.store import RUN_MANIFEST_FILENAME, claim_file
from tests.conftest import make_config


def test_get_or_create_is_idempotent(ledger):
    identity = ledger.identity(make_config())
    first = ledger.get_or_create("train", identity)
    assert ledger.get_or_create("train", identity).folder_id == first.folder_id
    assert ledger.find("train", identity).folder_id == first.folder_id


def test_find_returns_none_when_unregistered(ledger):
    assert ledger.find("train", ledger.identity(make_config())) is None


def test_namespaces_are_separate(ledger):
    identity = ledger.identity(make_config())
    assert ledger.get_or_create("train", identity).folder_id != ledger.get_or_create("eval", identity).folder_id


def test_sidecar_records_row_and_tree(ledger):
    identity = ledger.identity(make_config(seed=7))
    entry = ledger.get_or_create("train", identity)
    stored = json.loads(ledger.sidecar_path("train", entry.hash).read_text())
    assert stored["config"] == identity.materialized
    assert stored["tree"] == identity.tree
    assert "policy" not in stored and "identity" not in stored


def test_concurrent_claims_resolve_to_one_winner(ledger):
    identity = ledger.identity(make_config())
    with ThreadPoolExecutor(16) as pool:
        ids = {e.folder_id for e in pool.map(lambda _: ledger.get_or_create("train", identity), range(64))}
    assert len(ids) == 1
    leftovers = [f for f in os.listdir(ledger.sidecar_path("train", identity.hash).parent) if ".tmp." in f]
    assert leftovers == []


def test_claim_file_loser_adopts_winner(tmp_path):
    path = tmp_path / "x.json"
    assert claim_file(path, {"who": "a"})["who"] == "a"
    assert claim_file(path, {"who": "b"})["who"] == "a"


def test_register_points_new_hash_at_existing_folder(ledger):
    old = ledger.get_or_create("train", ledger.identity(make_config()))
    new_identity = ledger.identity(make_config(seed=9))
    assert ledger.register("train", new_identity, old.folder_id).folder_id == old.folder_id


def test_run_manifest_accumulates_hashes(ledger, tmp_path):
    folder, entry = ledger.resolve_output_folder("train", tmp_path / "out", make_config())
    other = ledger.identity(make_config(seed=1))
    ledger.write_run_manifest(folder, ledger.register("train", other, entry.folder_id), other)
    manifest = json.loads((folder / RUN_MANIFEST_FILENAME).read_text())
    assert manifest["config_hashes"] == [entry.hash, other.hash]
    assert manifest["folder_id"] == entry.folder_id == folder.name
    assert manifest["tree"] == other.tree


def test_lookup_output_folder_is_read_only(ledger, tmp_path):
    assert ledger.lookup_output_folder("train", tmp_path, make_config()) is None
    folder, _ = ledger.resolve_output_folder("train", tmp_path, make_config())
    assert ledger.lookup_output_folder("train", tmp_path, make_config()) == folder


def test_changed_default_is_a_new_identity(ledger, monkeypatch):
    """Every value is hashed: a config relying on a changed default no longer hits the old run."""
    from tests.conftest import Trainer

    trainer = {"_target_": Trainer._target_}
    implicit = {"m": {"_target_": "tests.conftest.Method", "trainer": trainer}}
    old = ledger.get_or_create("train", ledger.identity(implicit))
    monkeypatch.setattr(Trainer.__dataclass_fields__["learning_rate"], "default", 0.2)
    assert ledger.find("train", ledger.identity(implicit)) is None
    explicit_old = {"m": {**implicit["m"], "trainer": {**trainer, "learning_rate": 0.1}}}
    assert ledger.find("train", ledger.identity(explicit_old)).folder_id == old.folder_id


def _busy_replace(monkeypatch, failures, code=errno.ETXTBSY):
    """``os.replace`` fails ``failures`` times, as on NFSv4 when another client holds the target."""
    real_replace = os.replace
    calls = []

    def replace(src, dst):
        calls.append(dst)
        if len(calls) <= failures:
            raise OSError(code, os.strerror(code), str(src), None, str(dst))
        real_replace(src, dst)

    monkeypatch.setattr(_files.os, "replace", replace)
    monkeypatch.setattr(store.time, "sleep", lambda _: None)
    return calls


def test_run_manifest_retries_busy_replace(ledger, tmp_path, monkeypatch):
    calls = _busy_replace(monkeypatch, failures=3)
    identity = ledger.identity(make_config())
    entry = ledger.get_or_create("train", identity)
    path = ledger.write_run_manifest(tmp_path, entry, identity)
    assert len(calls) == 4
    assert json.loads(path.read_text())["config_hashes"] == [identity.hash]
    assert not list(tmp_path.glob("*.tmp.*")), "failed attempts must not leave temp files"


def test_run_manifest_stops_retrying_once_the_hash_is_there(ledger, tmp_path, monkeypatch):
    identity = ledger.identity(make_config())
    entry = ledger.get_or_create("train", identity)
    ledger.write_run_manifest(tmp_path, entry, identity)  # the "other task" wins first
    calls = _busy_replace(monkeypatch, failures=100)
    ledger.write_run_manifest(tmp_path, entry, identity)
    assert calls == []


def test_run_manifest_gives_up_and_reraises(ledger, tmp_path, monkeypatch):
    calls = _busy_replace(monkeypatch, failures=10_000)
    identity = ledger.identity(make_config())
    entry = ledger.get_or_create("train", identity)
    with pytest.raises(OSError) as raised:
        ledger.write_run_manifest(tmp_path, entry, identity)
    assert raised.value.errno == errno.ETXTBSY and len(calls) == store.RUN_MANIFEST_ATTEMPTS
    assert not list(tmp_path.glob("*.tmp.*"))


def test_run_manifest_does_not_retry_other_errors(ledger, tmp_path, monkeypatch):
    calls = _busy_replace(monkeypatch, failures=1, code=errno.EACCES)
    identity = ledger.identity(make_config())
    entry = ledger.get_or_create("train", identity)
    with pytest.raises(PermissionError):
        ledger.write_run_manifest(tmp_path, entry, identity)
    assert len(calls) == 1
