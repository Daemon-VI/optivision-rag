"""Canonical serialisation and content digests.

Everything the selection and certificate layers call an *identity* is a SHA-256
digest computed here, so two parties can check that they are talking about the
same candidate list, corpus, query set or fitted state.

``canonical_json`` produces the same bytes as
``json.dumps(obj, sort_keys=True, separators=(",", ":"))`` for JSON-native input.
A digest is an integrity check. It is not a signature: anyone can recompute it.
"""

from __future__ import annotations

import enum
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

_BLOCK_BYTES = 16 * 1024 * 1024


def _plain(obj: Any) -> Any:
    """``obj`` as JSON-native values, refusing anything whose encoding would be ambiguous."""
    if obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, enum.Enum):
        return _plain(obj.value)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        if not math.isfinite(value):
            raise ValueError("canonical JSON cannot hold NaN or infinity; use None")
        return value
    if isinstance(obj, np.ndarray):
        return _plain(obj.tolist())
    if isinstance(obj, Mapping):
        out = {}
        for key, value in obj.items():
            if not isinstance(key, str):
                raise TypeError(f"canonical JSON needs string keys, got {type(key).__name__}")
            out[key] = _plain(value)
        return out
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    raise TypeError(f"canonical JSON cannot hold a {type(obj).__name__}")


def canonical_json(obj: Any) -> bytes:
    """Deterministic JSON: sorted keys, no whitespace, ASCII, finite numbers only."""
    return json.dumps(_plain(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(obj: Any) -> str:
    """SHA-256 of the canonical JSON of ``obj``."""
    return sha256_hex(canonical_json(obj))


def array_digest(a: Any) -> str:
    """SHA-256 of an array's dtype, shape and values (little-endian, C order).

    Arrays that compare equal but differ in dtype or shape have different digests.
    The values are hashed in blocks, so a large memory-mapped array is not copied whole.
    """
    a = np.asarray(a)
    if a.dtype.hasobject or a.dtype.fields is not None:
        # a structured dtype's ``str`` names only its size, so two layouts would share a digest
        raise TypeError("array_digest needs a numeric, boolean or fixed-width string array")
    big_endian = a.dtype.byteorder == ">" or (a.dtype.byteorder == "=" and sys.byteorder == "big")
    dtype = a.dtype.newbyteorder("<") if big_endian else a.dtype
    h = hashlib.sha256()
    h.update(f"{dtype.str}|{','.join(str(n) for n in a.shape)}|".encode("ascii"))
    if a.ndim == 0:
        h.update(np.ascontiguousarray(a, dtype=dtype).tobytes())
        return h.hexdigest()
    row_bytes = max(1, dtype.itemsize * int(np.prod(a.shape[1:], dtype=np.int64)))
    step = max(1, _BLOCK_BYTES // row_bytes)
    for lo in range(0, a.shape[0], step):
        h.update(np.ascontiguousarray(a[lo:lo + step], dtype=dtype).tobytes())
    return h.hexdigest()


def strings_digest(items: Sequence[str]) -> str:
    """SHA-256 of an ordered sequence of strings. A bare string is refused: it would be
    read as its characters."""
    if isinstance(items, (str, bytes)):
        raise TypeError("strings_digest needs a sequence of strings, not one string")
    return digest([str(s) for s in items])
