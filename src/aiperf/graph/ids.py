# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The two graph ID grammars and their only parser.

The agent graph plane carries two distinct, easy-to-confuse identifier shapes:

* **Node id** ``{scope}:{turn}`` -- minted by the recorded adapters and the
  dag_jsonl lowering, one linear chain per recorded session with a 0-based turn
  ordinal. Unshaped, author-chosen ids (``"plan"``, ``"phase:review"``) are
  still accepted by :func:`split_node_id`; no producer in this release emits
  them.
* **Trace instance id** ``{template}::{nonce}`` -- minted per recycle pass so
  two concurrent instances of one template never collide. Build-time stores are
  keyed by the TEMPLATE, so read paths must strip the nonce.

Both were previously re-derived at two call sites each across four modules.
Confusing the template with the instance previously deadlocked spawn dispatch,
so the grammars live here and nowhere else.
"""

from __future__ import annotations

__all__ = ["chain_key", "split_node_id", "template_id"]

_NODE_DELIM = ":"
_INSTANCE_DELIM = "::"


def split_node_id(node_id: str) -> tuple[str, int] | None:
    """``(scope, turn)`` of a ``{scope}:{turn}`` node id; ``None`` otherwise.

    ``None`` means the id is an unshaped, author-chosen one, and callers fall
    back to root-trajectory identity.

    Example:
        >>> split_node_id("sess_A:3")
        ('sess_A', 3)
        >>> split_node_id("phase:review") is None   # unshaped author id
        True
    """
    scope, sep, turn = node_id.rpartition(_NODE_DELIM)
    if sep and scope and turn.isdigit():
        return scope, int(turn)
    return None


def chain_key(node_id: str) -> str:
    """The per-session chain a node id belongs to.

    ``:`` is the only correct delimiter: the lowering's ``_guard_session_scopes``
    forbids ``:`` inside a recorded session id but permits ``_``. Splitting on
    ``_`` both failed to split colon-only ids (every chain a singleton, so a live
    chain looked absent and primed nothing) and merged unrelated sessions sharing
    an underscore prefix (``sess_A:0`` and ``sess_B:1`` both keying ``sess``).

    Chain identity must come from node ids because the sidecar-loaded timing
    plane strips ``metadata["trie"]`` contents, leaving ids + edges as the only
    runtime chain signal. An unshaped id forms a defensive singleton chain.

    Example:
        >>> chain_key("sess_A:3")   # the per-session chain, NOT the trace template
        'sess_A'
        >>> chain_key("sess_B:1")   # underscore-sharing sessions stay distinct
        'sess_B'
        >>> chain_key("plan")       # unshaped id is its own singleton chain
        'plan'
    """
    shaped = split_node_id(node_id)
    return shaped[0] if shaped is not None else node_id


def template_id(trace_id: str) -> str:
    """The nonce-stripped template of a ``{template}::{nonce}`` instance id.

    Build-time manifests are keyed by template, so every recycle instance of one
    template reads the same stores. A trace id with no nonce is already a
    template and returns unchanged.

    Example:
        >>> template_id("t-1::3f2a")   # strips the per-recycle nonce
        't-1'
        >>> template_id("t-1")         # already a template
        't-1'
    """
    return trace_id.split(_INSTANCE_DELIM, 1)[0]
