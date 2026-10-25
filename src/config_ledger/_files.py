"""Small JSON file primitives shared by the store and the version index.

Everything the ledger writes is either insert-only (:func:`claim_file`, atomic ``os.link()``)
or a whole-file replace (:func:`write_json_atomic`, atomic ``os.replace()``); see
:mod:`config_ledger.store` for why no locks or databases are used.
"""

from __future__ import annotations

import contextlib
import errno
import json
import os
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        with open(path) as file:
            return json.load(file)
    except FileNotFoundError:
        return None


# Raised by ``os.replace`` on NFSv4 when another client has the target open or is replacing it
# at the same moment (seen on a real cluster). The replace did not happen; retry it.
BUSY_ERRNOS = frozenset({errno.ETXTBSY, errno.EBUSY})


def is_busy(error: OSError) -> bool:
    return error.errno in BUSY_ERRNOS


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Replace ``path`` atomically (last writer wins). Used for the reverse index only.

    The temp file is removed if the replace fails, e.g. with a busy error (:func:`is_busy`),
    which callers that may race should retry.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with open(temp_path, "w") as file:
            json.dump(payload, file, indent=2, sort_keys=True)
        os.replace(temp_path, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.remove(temp_path)
        raise


def claim_file(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Atomically create ``path`` holding ``payload``, or return whatever is already there.

    ``os.link()`` is atomic and fails with ``FileExistsError`` even on NFS, so of two racing
    writers exactly one wins; the loser discards its own payload and adopts the winner's.
    The temp file is always removed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    with open(temp_path, "w") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
    try:
        os.link(temp_path, path)
    except FileExistsError:
        pass
    finally:
        os.remove(temp_path)
    stored = read_json(path)
    if stored is None:
        raise RuntimeError(f"{path} vanished immediately after being claimed")
    return stored
