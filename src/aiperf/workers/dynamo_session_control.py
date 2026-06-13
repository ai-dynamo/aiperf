# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamo conversation-aware routing via ``nvext.session_control``.

Opt-in helpers that shape the ``nvext.session_control`` block Dynamo frontends
read to pin every turn of a replayed conversation to the same backend worker
(reusing its prefix KV cache). Kept separate from request building so the
worker stays unaware of wire-body internals: the policy (which lifecycle action
to emit) lives in :func:`build_session_control`, and the mechanism (overlaying
it onto the structured request body) in :func:`merge_session_control`. The
single caller is the serialization chokepoint
:meth:`aiperf.workers.inference_client.InferenceClient._send_request_to_transport`,
right after the endpoint formats the request dict.

The verbatim PAYLOAD_BYTES mmap fast path (raw_payload / inputs_json /
mooncake-with-payload datasets) is refused against this feature at dataset load
(:meth:`aiperf.dataset.dataset_manager.DatasetManager._select_mmap_format`), so
this module only ever sees a freshly built request dict -- mirroring how
``--cache-bust`` is handled.
"""

from __future__ import annotations

from typing import Any


def build_session_control(
    *,
    session_id: str,
    is_final_turn: bool,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Build the ``nvext.session_control`` block for a single request.

    Each conversation instance is its own sticky session keyed by its
    X-Correlation-ID (``session_id``). Dynamo's router co-locates turns purely
    by ``session_id``, so the lifecycle is:

    - non-final turn -> ``action: "bind"`` (router-only sticky affinity).
      Re-bind is idempotent on the router and refreshes the inactivity TTL, so
      long inter-turn replay delays cannot silently expire affinity
      mid-conversation. That is why every non-final turn re-binds rather than
      only the first -- it also removes any need for per-worker bind tracking.
    - final turn -> ``action: "close"`` to release the router affinity (and any
      worker-side session) immediately instead of leaking it until the TTL
      reaper fires -- material on high-cardinality runs (millions of sessions).

    Fork/spawn children use their own ``session_id`` rather than a shared
    lineage key: ``close`` is keyed on ``session_id``, so a shared key would let
    a child's ``close`` tear down the parent's still-live affinity. Cross-branch
    KV reuse survives anyway because the child's first (still-unbound) request is
    routed by Dynamo's prefix-overlap router onto the worker already holding the
    shared prefix.
    """
    session_control: dict[str, Any] = {"session_id": session_id}
    if is_final_turn:
        session_control["action"] = "close"
    else:
        session_control["action"] = "bind"
        session_control["timeout"] = timeout_seconds
    return session_control


def merge_session_control(
    payload: dict[str, Any],
    session_control: dict[str, Any],
) -> dict[str, Any]:
    """Return a copy of ``payload`` with ``session_control`` overlaid under ``nvext``.

    Never mutates ``payload`` nor its nested ``nvext`` / ``session_control``
    dicts, so it is safe on a cached ``Turn.raw_payload`` or on an ``extra_body``
    reference shared with the dataset. Existing ``nvext`` keys and any
    pre-existing ``session_control`` fields are preserved; the computed fields
    overlay them.
    """
    merged = dict(payload)
    raw_nvext = merged.get("nvext")
    nvext = dict(raw_nvext) if isinstance(raw_nvext, dict) else {}
    raw_session_control = nvext.get("session_control")
    merged_session_control = (
        dict(raw_session_control) if isinstance(raw_session_control, dict) else {}
    )
    merged_session_control.update(session_control)
    nvext["session_control"] = merged_session_control
    merged["nvext"] = nvext
    return merged
