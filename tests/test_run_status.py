from pathlib import Path

import pytest

from config_ledger import RunDecision, RunStatus


class DoneWhenMarked(RunStatus):
    @classmethod
    def check_run_status(cls, config, output_folder):
        if (Path(output_folder) / "done").exists():
            return RunDecision(False, "marked done")
        return RunDecision(True, "not done")


def test_decision_unpacks_as_bool_and_reason(tmp_path):
    launch, reason = DoneWhenMarked.check_run_status({}, tmp_path)
    assert (launch, reason) == (True, "not done")
    (tmp_path / "done").touch()
    decision = DoneWhenMarked.check_run_status({}, tmp_path)
    assert decision == RunDecision(launch=False, reason="marked done")


def test_base_check_is_abstract_and_raises_when_called(tmp_path):
    class Forgot(RunStatus):
        pass

    with pytest.raises(NotImplementedError, match="Forgot must override check_run_status"):
        Forgot.check_run_status({}, tmp_path)
    with pytest.raises(TypeError):
        Forgot()
