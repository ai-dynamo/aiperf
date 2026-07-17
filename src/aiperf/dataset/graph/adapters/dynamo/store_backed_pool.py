# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamo-scoped :class:`SegmentPool` shims for the two dynamo build routes.

Both dynamo routes re-list every message chain into each node's
``prompt_segment_ids``, so absent interning the same ``segment_id`` hexdigest is
born as one fresh str per re-listing (``~H/N`` copies per node) even though
:meth:`SegmentPool.add` already stores exactly one canonical
:class:`~aiperf.dataset.graph.segment_ir.pool.Segment` per unique value. These
two shims make ``add`` return that first-born canonical object so every
re-listing shares it -- values are byte-identical (interning changes str
identity only), so the store bytes, sidecar, and envelope are unaffected.

* :class:`InterningSegmentPool` -- the EAGER route
  (``from_dynamo_trace(direct_store=None)``). It subclasses
  :class:`SegmentPool` and returns ``self._by_id[sid].id`` (the first-born
  ``Segment.id``) from every ``add*``. The canonical is the object already
  retained as the ``_by_id`` key/Segment, so the intern table is FREE.

* :class:`StoreBackedSegmentPool` -- the DIRECT write-through route
  (``GraphStoreBuilder._build_graph_store_streaming``), which constructs the
  unified store BEFORE the parse and threads it into
  ``from_dynamo_trace(direct_store=...)``. Every ``pool.add`` interns its segment
  STRAIGHT INTO the store at parse time (no second RAM pool copy) -- the store's
  own ``_ids`` first-occurrence dedup gives the same handle stream the eager
  ``build_unified_trie_store_interned`` drain would assign, so the produced store
  is byte-identical (both routes intern in ``build_trie_ir``'s content-loop
  first-occurrence order, the single ordering authority). A handle-indexed
  ``_sids`` list holds one canonical str per unique value so repeats return the
  first-born object instead of the fresh hexdigest.

The direct shim's ``_by_id`` stays empty (segments live in the store, not the
pool), so the returned ``ParsedGraph.segment_pool`` no-ops the interned drain's
put loop and ``strip_replay_text`` replaces it with a fresh empty ``SegmentPool``
before the content-free sidecar is msgpack-encoded (nothing ever encodes a live
shim).

Both shims live here rather than in ``pool.py`` so ``pool.py`` stays a
dependency-free stdlib leaf: the store type is referenced only under
``TYPE_CHECKING`` here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiperf.dataset.graph.segment_ir.pool import Segment, SegmentPool, segment_id

if TYPE_CHECKING:
    from aiperf.dataset.graph_segment_unified_store import (
        GraphSegmentUnifiedBackingStore,
    )

_DIRECT_ONLY = (
    "StoreBackedSegmentPool is the dynamo DIRECT write-through shim: the dynamo "
    "content path interns segments ONLY via add() (add_message_chain + "
    "trie_content emit/small-prompt), and nothing reads the pool back after "
    "build_trie_ir. {method}() is unsupported -- an adapter that reaches it "
    "(text/raw segments, or a read-back materialize/get) must build through the "
    "eager SegmentPool/InterningSegmentPool route, not the direct store route."
)


class InterningSegmentPool(SegmentPool):
    """Eager-route :class:`SegmentPool` that returns the FIRST-BORN canonical sid.

    Every ``add*`` returns ``self._by_id[sid].id`` -- the ``Segment.id`` string
    stored on the segment's first occurrence -- instead of the freshly computed
    hexdigest :meth:`SegmentPool.add` returns. On a dedup hit the fresh duplicate
    dies immediately (refcount) and the canonical is what flows into every node's
    ``prompt_segment_ids``, so a value re-listed across turns/sessions shares one
    str object. On the first occurrence the returned object IS the fresh sid
    (``_by_id[sid].id is sid``), so there is zero cost beyond one dict hit.

    This depends on ``Segment(id=sid)`` being stored with the FIRST-BORN sid
    (:meth:`SegmentPool.add` inserts only when ``sid not in self._by_id``), which
    is stable while ``pool.py`` is frozen. Values are byte-identical to a plain
    ``SegmentPool`` (identity-only change), so store/sidecar/envelope bytes hold.

    ``SegmentPool`` is ``@dataclass(slots=True)``; ``__slots__ = ()`` keeps this
    subclass slot-only (no per-pool ``__dict__``).
    """

    __slots__ = ()

    def add(
        self, *, role: str, content: str, tokens: list[int], parent_id: str | None
    ) -> str:
        sid = super().add(
            role=role, content=content, tokens=tokens, parent_id=parent_id
        )
        return self._by_id[sid].id

    def add_text(self, *, role: str, content: str, parent_id: str | None) -> str:
        sid = super().add_text(role=role, content=content, parent_id=parent_id)
        return self._by_id[sid].id

    def add_raw_message(self, *, message: dict[str, Any], parent_id: str | None) -> str:
        sid = super().add_raw_message(message=message, parent_id=parent_id)
        return self._by_id[sid].id


class StoreBackedSegmentPool(SegmentPool):
    """A :class:`SegmentPool` whose :meth:`add` writes through to the unified store.

    Only :meth:`add` is a real operation (the dynamo path's sole pool call); every
    other pool method raises :class:`NotImplementedError` naming the dynamo-only
    write-through contract, so any non-dynamo adopter fails loud instead of
    silently interning into an empty ``_by_id`` the store never sees.

    A handle-indexed ``_sids`` list holds one canonical sid str per unique value
    (in first-occurrence order) so a repeated segment returns the first-born
    object, mirroring :class:`InterningSegmentPool` on the direct route.
    """

    __slots__ = ("_store", "_sids")

    def __init__(self, store: GraphSegmentUnifiedBackingStore) -> None:
        super().__init__()
        self._store = store
        # Handle -> canonical sid str. ``put_segment`` returns dense insertion
        # indexes on first occurrence, so index i holds the sid first interned at
        # handle i; a repeat's handle indexes back into this list to return the
        # first-born object (the canonical intern table for the direct route).
        self._sids: list[str] = []

    def add(
        self,
        *,
        role: str,
        content: str,
        tokens: list[int],
        parent_id: str | None,
    ) -> str:
        """Compute the content-addressed id, write it through, return the canonical.

        The sid VALUE is what :meth:`SegmentPool.add` returns. ``store.put_segment``
        dedups first-occurrence on its ``_ids`` map (a repeat ``sid`` is a no-op
        that returns the existing handle) and derives the persisted blob as
        ``orjson.dumps({"role", "content"})`` -- byte-identical to what the eager
        drain writes for the same token/text segment. The returned OBJECT is the
        first-born canonical: on a first occurrence ``handle == len(self._sids)``
        so the fresh sid is appended and returned; on a repeat ``handle`` indexes
        the earlier canonical out of ``_sids``. The defensive third branch (a
        ``handle`` outrunning ``_sids`` because the store was pre-populated before
        this shim existed -- never in production, where the shim is the store's
        sole parse-time writer) degrades to returning the fresh value-correct sid.
        """
        sid = segment_id(parent_id, role, tokens)
        handle = self._store.put_segment(sid, role, content)
        if handle == len(self._sids):
            self._sids.append(sid)
            return sid
        if handle < len(self._sids):
            return self._sids[handle]
        return sid

    def add_text(self, *, role: str, content: str, parent_id: str | None) -> str:
        raise NotImplementedError(_DIRECT_ONLY.format(method="add_text"))

    def add_raw_message(self, *, message: dict[str, Any], parent_id: str | None) -> str:
        raise NotImplementedError(_DIRECT_ONLY.format(method="add_raw_message"))

    def get(self, sid: str) -> Segment:
        raise NotImplementedError(_DIRECT_ONLY.format(method="get"))

    def materialize(self, path_ids: list[str]) -> list[dict]:
        raise NotImplementedError(_DIRECT_ONLY.format(method="materialize"))


__all__ = ["InterningSegmentPool", "StoreBackedSegmentPool"]
