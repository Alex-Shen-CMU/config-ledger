"""OmegaConf adapter (extra ``config-ledger[omegaconf]``): reads Hydra's ``DictConfig`` lazily.

Nodes are kept as ``DictConfig`` rather than converted, so ``.items()`` resolves
interpolations only inside identity nodes. Execution-control blocks that reference
runtime-only resolvers (``outputs: ${hydra:runtime.cwd}/...``) are never touched, which a
blanket ``OmegaConf.to_container(cfg, resolve=True)`` would trip over. Leaves (``ListConfig``
/ ``DictConfig`` values of a field) are converted with interpolations resolved.

Imported by :mod:`config_ledger.adapters` only once ``omegaconf`` is already loaded.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig, OmegaConf


class OmegaConfAdapter:
    def handles(self, value: Any) -> bool:
        return OmegaConf.is_config(value)

    def as_mapping(self, value: Any) -> Mapping[str, Any] | None:
        # A ListConfig is a leaf. DictConfig is a MutableMapping, returned unconverted.
        return value if isinstance(value, DictConfig) else None

    def to_plain(self, value: Any) -> Any:
        return OmegaConf.to_container(value, resolve=True)


ADAPTER = OmegaConfAdapter()
