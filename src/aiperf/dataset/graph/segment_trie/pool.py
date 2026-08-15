# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Content-addressed segment pool primitives (neutral stdlib-only leaf).

These dataclasses live in a dependency-free leaf module so ``models.py`` can
type ``ParsedGraph.segment_pool`` as ``SegmentPool | None`` without pulling in
``adapters/__init__`` (which eagerly imports the adapter modules -> ``models``
and would form a partial-init import cycle).

Disambiguation -- "pool" names three unrelated things in the graph subsystem.
Here it is the BUILD-plane interned CONTENT store (:class:`Segment`,
:class:`SegmentPool`, plus the dynamo shims
:class:`~aiperf.dataset.graph.adapters.dynamo.store_backed_pool.InterningSegmentPool`
and
:class:`~aiperf.dataset.graph.adapters.dynamo.store_backed_pool.StoreBackedSegmentPool`).
It is NOT ``aiperf.graph.dynamic_pool`` (the worker's RUNTIME cache of captured
replies, keyed by trace; ``GraphPoolMissingError`` refers to that one) and NOT
``adapters/shared/pool.py`` (a multiprocessing WORKER pool for parse dispatch).
"""

from __future__ import annotations

import hashlib
from array import array
from dataclasses import dataclass, field
from typing import Any

import orjson


@dataclass(slots=True, frozen=True)
class Segment:
    id: str
    """Content-addressed, prefix-dependent id (opaque to consumers)."""
    role: str
    """Message role (e.g. ``"user"``); ``message.get("role", "")`` for raw segments."""
    content: str
    """Message content; ``""`` for raw segments (the wire blob is authoritative)."""
    parent_id: str | None
    """Prefix segment id this one extends, or ``None`` at the path root."""
    wire_json: str | None = None
    """Verbatim ``orjson.dumps(message)`` for a raw-authored segment (key order and
    extra keys preserved). ``None`` for a role/content segment, whose blob is derived
    as ``{"role", "content"}`` at persist time (the existing normalized behavior)."""


def segment_id(parent_id: str | None, role: str, tokens: list[int]) -> str:
    """Content-addressed, prefix-dependent id for one role/content segment.

    The id is OPAQUE (consumers treat it as a name, not a value): only its
    determinism, prefix-dependence (``parent_id`` framing), role-framing, and
    dedup-consistency matter. The token list is hashed via a bulk C-level
    ``array("q", tokens).tobytes()`` int image -- a deterministic, injective
    byte encoding of the int sequence -- instead of the per-token
    ``str().encode()`` join that dominated the cold-build profile (~44M
    per-token calls on a corpus-scale trace). The ``\\x00`` delimiters keep the
    ``parent_id`` / ``role`` / tokens fields framed so distinct fields cannot
    alias, and the fixed-width 8-byte little-endian image means two distinct
    token lists cannot collide via concatenation.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update((parent_id or "").encode())
    h.update(b"\x00")
    h.update(role.encode())
    h.update(b"\x00")
    # token ids are the canonical, tokenization-1:1 content key; the bulk
    # fixed-width int image is a deterministic injective byte encoding.
    h.update(array("q", tokens).tobytes())
    return h.hexdigest()


def text_segment_id(parent_id: str | None, role: str, content: str) -> str:
    """Content-addressed id for a text-authored segment (no tokenizer needed).

    Same ``parent_id`` / ``role`` framing as :func:`segment_id`, with a distinct
    domain tag before the payload so a text-derived id can never alias a
    token-derived id for the same prefix and role. Ids are opaque to every
    consumer; only determinism and dedup-consistency matter, so a producer with
    no tokenizer at parse time can key on UTF-8 content bytes instead of token
    ids.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update((parent_id or "").encode())
    h.update(b"\x00")
    h.update(role.encode())
    h.update(b"\x00text\x00")
    h.update(content.encode())
    return h.hexdigest()


def raw_segment_id(parent_id: str | None, wire_json: str) -> str:
    """Content-addressed id for a raw-wire-JSON segment (verbatim authored message).

    Same ``parent_id`` framing as :func:`text_segment_id`, with a distinct ``raw``
    domain tag before the payload so a raw-derived id can never alias a text- or
    token-derived id for the same prefix. The payload is the verbatim
    ``orjson.dumps(message)`` bytes -- so two messages differing only in key order
    or an extra key hash to distinct ids, preserving the byte-verbatim contract.
    Ids are opaque; only determinism, prefix-dependence, and dedup-consistency matter.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update((parent_id or "").encode())
    h.update(b"\x00raw\x00")
    h.update(wire_json.encode())
    return h.hexdigest()


@dataclass(slots=True)
class SegmentPool:
    _by_id: dict[str, Segment] = field(default_factory=dict)

    @property
    def by_id(self) -> dict[str, Segment]:
        """Content-addressed id -> Segment map (read as-is; do not mutate)."""
        return self._by_id

    def add(
        self,
        *,
        role: str,
        content: str,
        tokens: list[int],
        parent_id: str | None,
    ) -> str:
        sid = segment_id(parent_id, role, tokens)
        if sid not in self._by_id:
            self._by_id[sid] = Segment(
                id=sid, role=role, content=content, parent_id=parent_id
            )
        return sid

    def add_text(
        self,
        *,
        role: str,
        content: str,
        parent_id: str | None,
    ) -> str:
        sid = text_segment_id(parent_id, role, content)
        if sid not in self._by_id:
            self._by_id[sid] = Segment(
                id=sid, role=role, content=content, parent_id=parent_id
            )
        return sid

    def add_raw_message(self, *, message: dict[str, Any], parent_id: str | None) -> str:
        """Intern a raw-authored message verbatim (key order and extra keys kept).

        The message is serialized ONCE via ``orjson.dumps`` and that blob is the
        segment's authoritative wire form: :func:`raw_segment_id` keys on it, so
        two messages differing only in key order or an extra key dedup distinctly.
        ``role`` records ``message.get("role", "")`` for envelope framing and
        ``content`` is ``""`` (the wire blob, not the derived dict, is persisted).
        """
        wire_json = orjson.dumps(message).decode()
        sid = raw_segment_id(parent_id, wire_json)
        if sid not in self._by_id:
            self._by_id[sid] = Segment(
                id=sid,
                role=message.get("role", ""),
                content="",
                parent_id=parent_id,
                wire_json=wire_json,
            )
        return sid

    def get(self, sid: str) -> Segment:
        return self._by_id[sid]

    def materialize(self, path_ids: list[str]) -> list[dict]:
        out: list[dict] = []
        for i in path_ids:
            s = self._by_id[i]
            if s.wire_json is not None:
                out.append(orjson.loads(s.wire_json))
            else:
                out.append({"role": s.role, "content": s.content})
        return out
