import omegaconf
import pandas as pd
import pytest

pytest.importorskip("hydra")
from config_ledger.hydra import ledger_from_config, sweeps  # noqa: E402
from tests.conftest import make_config  # noqa: E402

LR = "method.trainer.learning_rate"
SEED = "method.trainer.seed"


def _driver_config(sweep_choice="lr_sweep"):
    cfg = make_config()
    cfg.hydra = {
        "sweeper": {"params": {LR: "0.1, 0.2", SEED: "0", "+method.name": "fixture"}},
        "runtime": {"choices": {"sweep": sweep_choice}},
    }
    return cfg


def _compose(overrides):
    applied = sweeps.overrides_to_dict(overrides)
    return make_config(learning_rate=float(applied.get(LR, 0.1)), seed=int(applied.get(SEED, 0)))


GRID = [[f"{LR}=0.1", f"{SEED}=0"], [f"{LR}=0.2", f"{SEED}=0"]]


def test_rows_cover_every_param_and_resolve_folder_ids(ledger):
    rows = sweeps.build_rows(ledger, "train", _driver_config(), GRID, _compose)
    assert [r[LR] for r in rows] == ["0.1", "0.2"]
    assert all(r[SEED] == "0" and r["method.name"] == "fixture" for r in rows)
    assert rows[0]["folder_id"] != rows[1]["folder_id"]
    # What the task itself would resolve for the same overrides.
    assert ledger.get_or_create("train", ledger.identity(_compose(GRID[1]))).folder_id == rows[1]["folder_id"]


def test_write_lands_under_namespace_and_sweep_name(ledger):
    path = sweeps.write_sweep_manifest(ledger, "train", _driver_config(), GRID, _compose)
    assert path.parent == sweeps.sweep_dir(ledger, "train", "lr_sweep")
    assert len(pd.read_csv(path)) == 2


def test_write_never_raises(ledger):
    assert sweeps.write_sweep_manifest(ledger, "train", _driver_config(), GRID, lambda o: 1 / 0) is None


def test_load_dedups_resubmissions_and_builds_grid(ledger):
    sweeps.write_sweep_manifest(ledger, "train", _driver_config(), GRID, _compose)
    sweeps.write_sweep_manifest(ledger, "train", _driver_config(), GRID, _compose)
    table = sweeps.load_sweep(ledger, "train", "lr_sweep")
    assert len(table) == 2
    assert table.axes() == [LR]
    grid = table.grid([LR])
    assert set(grid) == {("0.1",), ("0.2",)}


def test_grid_refuses_unpinned_varying_columns(ledger):
    grid = [[f"{LR}=0.1", f"{SEED}=0"], [f"{LR}=0.2", f"{SEED}=1"]]
    sweeps.write_sweep_manifest(ledger, "train", _driver_config(), grid, _compose)
    table = sweeps.load_sweep(ledger, "train", "lr_sweep")
    with pytest.raises(ValueError, match="neither axes nor pinned"):
        table.grid([LR])
    assert len(table.pin({SEED: "1"}).grid([LR])) == 1
    with pytest.raises(KeyError):
        table.pin({"nope": "1"})


def test_load_missing_sweep_is_a_clear_error(ledger):
    with pytest.raises(FileNotFoundError, match="config-ledger launchers"):
        sweeps.load_sweep(ledger, "train", "nothing")


def test_ledger_from_config_reads_top_level_block(tmp_path):
    cfg = omegaconf.OmegaConf.create({"ledger": {"root": str(tmp_path), "namespace": "train"}})
    ledger, namespace = ledger_from_config(cfg)
    assert (ledger.root, namespace) == (tmp_path, "train")
    with pytest.raises(KeyError, match="ledger"):
        ledger_from_config(omegaconf.OmegaConf.create({}))


def test_sweep_version_check_runs_once_over_the_grid(ledger, tmp_path, monkeypatch):
    from config_ledger import StaleRunsError
    from tests.conftest import Trainer

    ledger.resolve_output_folder("train", tmp_path / "out", _compose(GRID[0]))
    assert sweeps.check_sweep_versions(ledger, "train", GRID, _compose) is True
    monkeypatch.setattr(Trainer, "VERSION", 2)
    calls = []
    real = ledger.versions
    monkeypatch.setattr(ledger, "versions", lambda ns: calls.append(ns) or real(ns))
    with pytest.raises(StaleRunsError, match="Trainer:1"):
        sweeps.check_sweep_versions(ledger, "train", GRID, _compose)
    assert calls == ["train"]


def test_sweep_version_check_falls_back_when_grid_cannot_be_composed(ledger):
    assert sweeps.check_sweep_versions(ledger, "train", GRID, lambda o: 1 / 0) is False


def test_same_second_manifests_do_not_overwrite_each_other(ledger, monkeypatch):
    from datetime import datetime

    # The second write's first name is taken, so it retries with a fresh timestamp.
    times = iter([datetime(2026, 1, 1), datetime(2026, 1, 1), datetime(2026, 1, 1, 0, 0, 0, 1)])
    clock = type("Clock", (), {"now": staticmethod(lambda: next(times))})
    monkeypatch.setattr(sweeps, "datetime", clock)
    first = sweeps.write_sweep_manifest(ledger, "train", _driver_config(), GRID, _compose)
    second = sweeps.write_sweep_manifest(ledger, "train", _driver_config(), GRID[:1], _compose)
    assert (first.name, second.name) == ("20260101T000000_000000.csv", "20260101T000000_000001.csv")
    assert [len(pd.read_csv(p)) for p in (first, second)] == [2, 1]


def test_parent_sweep_must_have_a_manifest(ledger):
    with pytest.raises(sweeps.ParentSweepManifestMissingError, match="manifest_only"):
        sweeps.check_parent_sweep(ledger, "train", "lr_sweep", GRID, _compose)
    assert not sweeps.sweeps_dir(ledger).exists()
    assert ledger.find("train", ledger.identity(_compose(GRID[0]))) is None, "nothing claimed"


def test_scoped_submission_within_parent_passes_and_writes_nothing(ledger):
    sweeps.write_sweep_manifest(ledger, "train", _driver_config(), GRID, _compose)
    sweeps.check_parent_sweep(ledger, "train", "lr_sweep", GRID[1:], _compose)
    assert len(sweeps.list_manifest_csvs(ledger, "train", "lr_sweep")) == 1


def test_scoped_submission_outside_parent_is_refused(ledger):
    sweeps.write_sweep_manifest(ledger, "train", _driver_config(), GRID, _compose)
    # Claimed elsewhere (another sweep) but not in the parent's grid: still a mismatch.
    ledger.get_or_create("train", ledger.identity(_compose([f"{LR}=0.3", f"{SEED}=0"])))
    scoped = [GRID[0], [f"{LR}=0.3", f"{SEED}=0"], [f"{LR}=0.1", f"{SEED}=7"]]
    with pytest.raises(sweeps.ParentSweepMismatchError, match="2 of 3") as raised:
        sweeps.check_parent_sweep(ledger, "train", "lr_sweep", scoped, _compose)
    assert f"{LR}=0.3" in str(raised.value) and f"{SEED}=7" in str(raised.value)
    assert ledger.find("train", ledger.identity(_compose(scoped[2]))) is None, "nothing claimed"


def test_ledger_settings_reads_parent_sweep(tmp_path):
    from config_ledger.hydra import ledger_settings

    block = {"root": str(tmp_path), "namespace": "train"}
    assert ledger_settings({"ledger": block})["parent_sweep"] is None
    assert ledger_settings({"ledger": {**block, "parent_sweep": "full"}})["parent_sweep"] == "full"
