import pytest

pytest.importorskip("hydra")
from config_ledger import RunDecision  # noqa: E402
from config_ledger.hydra import run_status  # noqa: E402
from config_ledger.hydra.run_status import SweepRunStatus, plan_launches, run_status_for  # noqa: E402
from tests.conftest import make_config  # noqa: E402


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    fresh = {}
    monkeypatch.setattr(run_status, "_REGISTRY", fresh)
    return fresh


def _status(namespace="train"):
    class Status(SweepRunStatus, namespace=namespace):
        calls = []

        @classmethod
        def check_run_status(cls, config, output_folder):
            cls.calls.append(output_folder)
            done = (output_folder / "summary.json").exists()
            return (not done, "complete" if done else "partial")   # a plain tuple is accepted

    return Status


def test_subclass_keyword_registers_by_namespace(registry):
    status = _status()
    assert run_status_for("train") is status and status.namespace == "train"

    class Unregistered(SweepRunStatus):
        @classmethod
        def check_run_status(cls, config, output_folder):
            return RunDecision(True, "")

    assert set(registry) == {"train"} and Unregistered.namespace is None


def test_reregistering_the_same_class_replaces_another_class_is_refused():
    first = _status()
    second = _status()   # same module and qualified name: a re-import
    assert first is not second and run_status_for("train") is second
    with pytest.raises(TypeError, match="already has a run status"):
        class Other(SweepRunStatus, namespace="train"):
            pass


def test_missing_bad_or_abstract_registrations_are_errors():
    with pytest.raises(LookupError, match="no run status is registered for namespace 'train'"):
        run_status_for("train")
    with pytest.raises(TypeError, match="non-empty string"):
        class Empty(SweepRunStatus, namespace=""):
            pass

    class Abstract(SweepRunStatus, namespace="evaluate"):
        pass

    with pytest.raises(TypeError, match="does not override check_run_status"):
        run_status_for("evaluate")


def _job(tmp_path, seed, outputs=True):
    block = {"root": str(tmp_path / "ledger"), "namespace": "train"}
    if outputs:
        block["outputs"] = str(tmp_path / "out")
    return make_config(seed=seed, ledger=block)


def test_plan_launches_checks_only_existing_folders(ledger, tmp_path):
    status = _status()
    grid = [["seed=0"], ["seed=1"], ["seed=2"], ["seed=3"]]
    jobs = {tuple(o): _job(tmp_path, int(o[0][-1])) for o in grid}
    compose = lambda overrides: jobs[tuple(overrides)]  # noqa: E731
    finished, _ = ledger.resolve_output_folder("train", tmp_path / "out", jobs[("seed=0",)])
    (finished / "summary.json").touch()
    partial, _ = ledger.resolve_output_folder("train", tmp_path / "out", jobs[("seed=1",)])
    ledger.get_or_create("train", ledger.identity(jobs[("seed=2",)]))   # claimed by a manifest only

    decisions = plan_launches(ledger, "train", grid, compose, status)

    assert [d.launch for d in decisions] == [False, True, True, True]
    assert all(isinstance(d, RunDecision) for d in decisions)
    assert [d.reason for d in decisions[:2]] == ["complete", "partial"]
    assert decisions[2].reason.startswith("no output folder yet")
    assert decisions[3].reason == "not in the ledger yet"
    assert status.calls == [finished, partial]
    unclaimed = ledger.identity(jobs[("seed=3",)])
    assert ledger.find("train", unclaimed) is None, "planning must not claim"


def test_plan_launches_needs_ledger_outputs(ledger, tmp_path):
    with pytest.raises(KeyError, match="ledger.outputs"):
        plan_launches(ledger, "train", [["seed=0"]],
                      lambda o: _job(tmp_path, 0, outputs=False), _status())
