# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The segment trie node envelope contract -- the adapter-facing seam.

An adapter opts a dispatchable ``LlmNode`` into the segment trie by stamping
``metadata["trie"]["prompt_segment_ids"]`` (an ordered path into the run's
``SegmentPool``) via :func:`stamp_prompt_segment_ids`. The build plane reads that
path back with :func:`read_prompt_segment_ids`.

The build-plane envelope (see ``store_builder._trie_envelope``) re-derives the
wire-body overrides / ``stream`` from the node's own fields, and the sidecar
strips ``metadata["trie"]`` contents to ``{}`` (keeping the key as a routing
marker). The trace adapters (dynamo) stamp only the path. The contract also
reserves ``assembly``/``capture`` extras for dynamic-slot nodes, which the eager
interned drain resolves and persists into the manifest (store-addressed
``items`` + ``capture``); no shipped lowering stamps them.
"""

from __future__ import annotations

from typing import Any

import msgspec

from aiperf.dataset.graph.models import LlmNode

# The metadata key under which the segment trie envelope lives. Internal to this
# module -- other modules do NOT import this constant; they key off the literal
# ``"trie"`` string directly (``graph_trace_planner.is_trie_graph`` routes on it, the
# graph_meta sidecar strips-but-keeps it). The string VALUE is the stable
# cross-module contract; the name is not part of the public surface.
TRIE_META_KEY = "trie"


def stamp_prompt_segment_ids(
    node: LlmNode, prompt_segment_ids: list[str], *, extra: dict[str, Any] | None = None
) -> LlmNode:
    """Return ``node`` with ``metadata["trie"]`` carrying ``prompt_segment_ids``.

    ``prompt_segment_ids`` is the ordered ``SegmentPool`` path the worker walks to
    materialize the prompt. ``extra`` supplies companion keys (the reserved
    ``assembly``/``capture`` dynamic-slot keys, which the
    eager interned drain persists), merged after ``prompt_segment_ids`` so
    the ``metadata["trie"]`` dict has a deterministic key order. Any pre-existing
    ``node.metadata`` is preserved.
    ``LlmNode`` is frozen, so this returns a replaced copy.
    """
    trie: dict[str, Any] = {"prompt_segment_ids": prompt_segment_ids}
    if extra:
        trie.update(extra)
    new_meta = {**(node.metadata or {}), TRIE_META_KEY: trie}
    return msgspec.structs.replace(node, metadata=new_meta)


def read_prompt_segment_ids(node: LlmNode) -> list[str] | None:
    """Read a node's ``metadata["trie"]["prompt_segment_ids"]`` path, or ``None``.

    ``None`` means the node is not part of the segment trie (no envelope, or a
    malformed one) -- the build plane skips it (mints no manifest).
    """
    trie_meta = (node.metadata or {}).get(TRIE_META_KEY)
    if not isinstance(trie_meta, dict):
        return None
    path = trie_meta.get("prompt_segment_ids")
    return path if isinstance(path, list) else None


__all__ = ["read_prompt_segment_ids", "stamp_prompt_segment_ids"]
