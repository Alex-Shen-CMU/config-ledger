import json
import subprocess
import sys
import textwrap
from pathlib import Path

import omegaconf
import pytest

from config_ledger import compute_identity, materialize
from config_ledger.adapters import MappingAdapter, adapter_for
from config_ledger.adapters.omegaconf import OmegaConfAdapter
from tests.conftest import MiningGoldTrainer, make_config

REPO = Path(__file__).resolve().parent.parent


def _plain(cfg: omegaconf.DictConfig) -> dict:
    return omegaconf.OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


@pytest.mark.parametrize(
    "cfg",
    [
        make_config(),
        make_config(seed=3, layers=[4, 2]),
        make_config(MiningGoldTrainer._target_, alpha=0.9),
        make_config(overwrite=True, output_dir="/x"),
    ],
)
def test_dictconfig_and_plain_dict_give_same_row_and_hash(cfg):
    plain = _plain(cfg)
    assert materialize(plain) == materialize(cfg)
    assert compute_identity(plain).hash == compute_identity(cfg).hash


def test_interpolation_inside_identity_node_is_resolved():
    cfg = omegaconf.OmegaConf.create({"lr": 0.3, **make_config()})
    cfg.method.trainer.learning_rate = "${lr}"
    assert materialize(cfg)["Method.trainer.learning_rate"] == "0.3"


def test_execution_control_blocks_are_never_resolved():
    cfg = make_config()
    cfg.ledger = {"root": "${hydra:runtime.cwd}/ledger"}  # resolver unavailable outside a job
    assert compute_identity(cfg).hash == compute_identity(make_config()).hash


@pytest.mark.xfail(strict=True, reason="pre-existing: top-level `.items()` resolves scalars")
def test_top_level_scalar_interpolation_is_never_resolved():
    cfg = make_config()
    cfg.outputs = "${hydra:runtime.cwd}/out"
    assert compute_identity(cfg).hash == compute_identity(make_config()).hash


def test_dispatch():
    assert isinstance(adapter_for(make_config()), OmegaConfAdapter)
    assert isinstance(adapter_for(omegaconf.OmegaConf.create([1])), OmegaConfAdapter)
    assert isinstance(adapter_for({"a": 1}), MappingAdapter)
    assert adapter_for([1, 2]) is None
    assert adapter_for(3) is None
    assert OmegaConfAdapter().as_mapping(omegaconf.OmegaConf.create([1])) is None


def test_core_works_without_omegaconf(tmp_path):
    """Import the core and claim a folder with every `omegaconf` / `pydantic` import blocked,
    from a plain dict and from a tree of dataclass instances."""
    script = textwrap.dedent(
        f"""
        import json, sys

        class Block:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in ("omegaconf", "pydantic"):
                    raise ImportError(name + " blocked")

        sys.meta_path.insert(0, Block())
        sys.path[:0] = [{str(REPO / "src")!r}, {str(REPO)!r}]
        from dataclasses import dataclass, field
        from typing import ClassVar
        from config_ledger import Ledger, VersionedConfig, compute_identity, execution_field

        @dataclass
        class Trainer(VersionedConfig):
            VERSION: ClassVar[int] = 1
            learning_rate: float = 0.1
            device: str = execution_field(default="cpu")

        @dataclass
        class Config:
            trainer: Trainer = field(default_factory=Trainer)
            output_dir: str = "/out"

        cfg = json.loads(sys.argv[1])
        ledger = Ledger({str(tmp_path / "ledger")!r})
        folder, _ = ledger.resolve_output_folder("ns", {str(tmp_path / "runs")!r}, cfg)
        dc_folder, _ = ledger.resolve_output_folder(
            "dc", {str(tmp_path / "runs")!r}, Config(Trainer(learning_rate=0.3, device="cuda")))
        again, _ = ledger.resolve_output_folder(
            "dc", {str(tmp_path / "runs")!r}, {{"trainer": {{"_target_": Trainer._target_,
                                                          "learning_rate": 0.3}}}})
        print(json.dumps({{"hash": compute_identity(cfg).hash, "same": str(dc_folder) == str(again),
                           "loaded": {{"omegaconf", "pydantic"}} & set(sys.modules) != set()}}))
        """
    )
    # conftest's schema module imports omegaconf, so use a schema-free node: the `_target_`
    # just has to resolve to a class, which `collections.OrderedDict` does.
    cfg = {"node": {"_target_": "collections.OrderedDict", "x": 1}, "overwrite": True}
    out = subprocess.run(
        [sys.executable, "-I", "-c", script, json.dumps(cfg)],
        check=True, capture_output=True, text=True,
    )
    result = json.loads(out.stdout)
    assert result["loaded"] is False
    assert result["same"] is True
    assert result["hash"] == compute_identity(omegaconf.OmegaConf.create(cfg)).hash
