# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict ``aiperf-canonical-json-v1`` codec used for semantic digests.

The codec accepts only the bounded JSON algebra used by the evaluator
protocol.  It rejects duplicate keys, surrogate code points, non-finite
numbers, booleans disguised as integers, and integers outside the combined
signed/unsigned 64-bit domain.  Object keys are ordered by their UTF-8 bytes;
floats use Python's shortest round-trip IEEE-754 spelling with negative zero
normalized to JSON integer zero.
"""

from __future__ import annotations

import json as _stdlib_json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import orjson

CANONICAL_JSON_CODEC = "aiperf-canonical-json-v1"
# Backwards-compatible spelling for callers that treat it as a version label.
CANONICAL_JSON_VERSION = CANONICAL_JSON_CODEC
MAX_CANONICAL_DEPTH = 64
MAX_CANONICAL_NODES = 65_536
MAX_CANONICAL_COLLECTION_ITEMS = 16_384
MAX_CANONICAL_STRING_BYTES = 1024 * 1024
MIN_CANONICAL_INTEGER = -(2**63)
MAX_CANONICAL_INTEGER = 2**64 - 1


class CanonicalJsonError(ValueError):
    """A value or input document is outside the canonical JSON domain."""


def canonical_dumps(value: Any) -> bytes:
    """Serialize one JSON value into deterministic UTF-8 bytes."""
    remaining = [MAX_CANONICAL_NODES]
    return _encode(value, depth=0, remaining=remaining).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Return the hexadecimal SHA-256 of canonical semantic JSON."""
    import hashlib

    return hashlib.sha256(canonical_dumps(value)).hexdigest()


def canonical_loads(payload: str | bytes, *, max_bytes: int = 1 << 20) -> Any:
    """Parse and validate canonical-domain JSON while rejecting duplicate keys."""
    raw = payload.encode("utf-8") if isinstance(payload, str) else payload
    if len(raw) > max_bytes:
        raise CanonicalJsonError(
            f"canonical JSON exceeds byte limit {max_bytes}: {len(raw)}"
        )
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise CanonicalJsonError("canonical JSON is not valid UTF-8") from error
    try:
        value = _stdlib_json.JSONDecoder(
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        ).decode(text)
    except _stdlib_json.JSONDecodeError as error:
        raise CanonicalJsonError(f"invalid JSON: {error.msg}") from error
    # Encoding performs the complete domain/depth/item validation.  Parsing is
    # intentionally not spelling-canonical: the wire protocol accepts ordinary
    # JSON, while semantic digests always use canonical_dumps().
    canonical_dumps(value)
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalJsonError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise CanonicalJsonError(f"non-finite JSON number {value!r} is forbidden")


def _consume(remaining: list[int]) -> None:
    remaining[0] -= 1
    if remaining[0] < 0:
        raise CanonicalJsonError(
            f"canonical JSON exceeds node limit {MAX_CANONICAL_NODES}"
        )


def _encode(value: Any, *, depth: int, remaining: list[int]) -> str:
    if depth > MAX_CANONICAL_DEPTH:
        raise CanonicalJsonError(
            f"canonical JSON exceeds depth limit {MAX_CANONICAL_DEPTH}"
        )
    _consume(remaining)
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        if not MIN_CANONICAL_INTEGER <= value <= MAX_CANONICAL_INTEGER:
            raise CanonicalJsonError(
                f"integer {value} is outside the canonical 64-bit domain"
            )
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalJsonError("non-finite floats are forbidden")
        if value == 0.0:
            return "0"
        # orjson and serde_json both use the Ryu finite-float formatter.  In
        # particular this avoids CPython repr's zero-padded exponent spelling.
        return orjson.dumps(value).decode("ascii")
    if isinstance(value, str):
        _validate_unicode(value)
        return orjson.dumps(value).decode("utf-8")
    if isinstance(value, Mapping):
        if len(value) > MAX_CANONICAL_COLLECTION_ITEMS:
            raise CanonicalJsonError(
                "canonical JSON object exceeds collection item limit "
                f"{MAX_CANONICAL_COLLECTION_ITEMS}"
            )
        items: list[tuple[str, Any]] = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalJsonError("canonical JSON object keys must be strings")
            _validate_unicode(key)
            items.append((key, item))
        items.sort(key=lambda item: item[0].encode("utf-8"))
        return (
            "{"
            + ",".join(
                _encode(key, depth=depth + 1, remaining=remaining)
                + ":"
                + _encode(item, depth=depth + 1, remaining=remaining)
                for key, item in items
            )
            + "}"
        )
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        if len(value) > MAX_CANONICAL_COLLECTION_ITEMS:
            raise CanonicalJsonError(
                "canonical JSON array exceeds collection item limit "
                f"{MAX_CANONICAL_COLLECTION_ITEMS}"
            )
        return (
            "["
            + ",".join(
                _encode(item, depth=depth + 1, remaining=remaining) for item in value
            )
            + "]"
        )
    raise CanonicalJsonError(
        f"unsupported canonical JSON value type {type(value).__name__}"
    )


def _validate_unicode(value: str) -> None:
    if len(value.encode("utf-8", errors="surrogatepass")) > MAX_CANONICAL_STRING_BYTES:
        raise CanonicalJsonError(
            f"canonical JSON string exceeds byte limit {MAX_CANONICAL_STRING_BYTES}"
        )
    for character in value:
        codepoint = ord(character)
        if 0xD800 <= codepoint <= 0xDFFF:
            raise CanonicalJsonError("Unicode surrogate code points are forbidden")
