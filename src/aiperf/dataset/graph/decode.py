# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single home for loose/legacy graph input -> typed IR coercion.

Coerces hand-authored / foreign dicts into the typed IR:
  - LlmNode dispatch + kind->node_type rename + messages->prompt alias (a node
    setting BOTH ``prompt`` and ``messages`` is rejected; so is a prompt-less
    node that only carries ``expected.input_tokens`` — synth-token prompt
    fabrication is not supported on the native lowering path)
  - static-edge coercion; non-finite delay values are rejected on edges and
    nodes (NaN/Inf discipline: they cross the msgpack boundary and would gate
    the successor forever at runtime)
  - vendor-key folding into the ``extra`` catch-all for ProvenanceSpec (that
    model does NOT forbid unknown fields, so msgspec would silently drop the
    vendor keys otherwise)

Trusted writers and the store bypass this with ``codecs.*`` directly.
"""

from __future__ import annotations

import math
from typing import Any

import msgspec

from aiperf.dataset.graph.models import (
    GraphRecord,
    LlmNode,
    NodeKind,
    ProvenanceSpec,
    StaticEdge,
)


class GraphDecodeError(ValueError):
    """Raised when loose graph input cannot be coerced to typed IR."""


_VALID_KINDS = {k.value for k in NodeKind}

# Known (declared) field names of the vendor-key catch-all model. Any
# key NOT in the set (and not "extra") is folded into the model's `extra` dict.
_PROVENANCE_FIELDS = set(ProvenanceSpec.__struct_fields__) - {"extra"}


def _fold_extra(raw: Any, known: set[str], *, loc: str) -> Any:
    """Fold unknown keys of a catch-all model's dict into its ``extra`` field."""
    if not isinstance(raw, dict):
        return raw
    extra_raw = raw.get("extra")
    if extra_raw is not None and not isinstance(extra_raw, dict):
        raise GraphDecodeError(
            f"{loc}.extra: must be a mapping of vendor keys, got "
            f"{type(extra_raw).__name__}"
        )
    out: dict[str, Any] = {}
    extra: dict[str, Any] = dict(extra_raw or {})
    for k, v in raw.items():
        if k == "extra":
            continue
        if k in known:
            out[k] = v
        else:
            extra[k] = v
    if extra:
        out["extra"] = extra
    return out


def _has_expected_input_tokens(raw: dict[str, Any]) -> bool:
    exp = raw.get("expected")
    return isinstance(exp, dict) and isinstance(exp.get("input_tokens"), int)


def _normalize_llm(raw: dict[str, Any], loc: str) -> dict[str, Any]:
    out = dict(raw)
    if "dispatch_overrides" in out:
        raise GraphDecodeError(
            f"{loc}: 'dispatch_overrides' was renamed 'extra_body' "
            f"(Turn-naming standardization); model / streaming / max_tokens / "
            f"raw_tools are first-class node fields"
        )
    if "prompt" in out and "messages" in out:
        raise GraphDecodeError(
            f"{loc}: node sets both 'prompt' and 'messages'; 'messages' is an "
            f"alias for 'prompt' — keep exactly one"
        )
    if "prompt" not in out and "messages" in out:
        out["prompt"] = out.pop("messages")
    if "prompt" not in out and _has_expected_input_tokens(out):
        raise GraphDecodeError(
            f"{loc}: node has no prompt; synth-token fabrication is not "
            f"supported on the native lowering path"
        )
    return out


def _require_finite(loc: str, field: str, value: float | None) -> None:
    if value is not None and not math.isfinite(value):
        raise GraphDecodeError(f"{loc}: {field} must be finite, got {value!r}")


def decode_node(raw: Any, node_id: str | None = None) -> Any:
    """Coerce one loose node spec (dict or already-typed struct) into typed IR.

    ``node_id`` (when the caller knows it, e.g. :func:`decode_graph`) is used
    only for ``graph.nodes.<id>``-style error locations.
    """
    if isinstance(raw, LlmNode):
        return raw
    loc = f"graph.nodes.{node_id}" if node_id is not None else "node"
    if not isinstance(raw, dict):
        raise GraphDecodeError(f"Node spec must be a mapping, got {type(raw).__name__}")
    raw_kind = raw.get("kind")
    explicit = raw.get("node_type") or raw_kind
    body = {k: v for k, v in raw.items() if k != "kind"}

    if explicit is not None and explicit not in _VALID_KINDS:
        field = "kind" if raw_kind is not None else "node_type"
        raise GraphDecodeError(
            f"unknown node {field} {explicit!r}; expected one of {sorted(_VALID_KINDS)}"
        )

    body = _normalize_llm(body, loc)
    body.pop("node_type", None)  # tag is implied by the chosen class
    try:
        node = msgspec.convert(body, type=LlmNode)
    except msgspec.ValidationError as exc:
        raise GraphDecodeError(
            f"{loc}: node decode failed for kind {explicit!r}: {exc}"
        ) from exc
    _require_finite(loc, "min_start_delay_us", node.min_start_delay_us)
    return node


def decode_edge(raw: Any) -> StaticEdge:
    """Coerce one loose edge spec (dict or already-typed struct) into typed IR."""
    if isinstance(raw, StaticEdge):
        return raw
    if not isinstance(raw, dict):
        raise GraphDecodeError(f"Edge spec must be a mapping, got {type(raw).__name__}")
    body = {k: v for k, v in raw.items() if k != "edge_type"}
    try:
        edge = msgspec.convert(body, type=StaticEdge)
    except msgspec.ValidationError as exc:
        raise GraphDecodeError(f"edge decode failed: {exc}") from exc
    loc = f"graph.edges[{edge.source}->{edge.target}]"
    _require_finite(loc, "delay_after_predecessor_us", edge.delay_after_predecessor_us)
    _require_finite(loc, "min_start_delay_us", edge.min_start_delay_us)
    _require_finite(
        loc,
        "delay_after_predecessor_start_us",
        edge.delay_after_predecessor_start_us,
    )
    _require_finite(
        loc,
        "delay_after_predecessor_first_token_us",
        edge.delay_after_predecessor_first_token_us,
    )
    return edge


def decode_graph(raw: Any) -> GraphRecord:
    """Coerce a loose graph dict (or :class:`GraphRecord`) into typed IR."""
    if isinstance(raw, GraphRecord):
        return raw
    if not isinstance(raw, dict):
        raise GraphDecodeError(
            f"Graph spec must be a mapping, got {type(raw).__name__}"
        )
    body = dict(raw)
    if isinstance(body.get("nodes"), dict):
        body["nodes"] = {nid: decode_node(n, nid) for nid, n in body["nodes"].items()}
    if isinstance(body.get("edges"), list):
        body["edges"] = [decode_edge(e) for e in body["edges"]]
    if isinstance(body.get("provenance"), dict):
        body["provenance"] = _fold_extra(
            body["provenance"], _PROVENANCE_FIELDS, loc="graph.provenance"
        )
    try:
        return msgspec.convert(body, type=GraphRecord, from_attributes=False)
    except msgspec.ValidationError as exc:
        raise GraphDecodeError(f"graph decode failed: {exc}") from exc
