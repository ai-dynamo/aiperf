# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed graph error codes carried on the credit-return wire.

``CreditReturn.error`` is a plain ``str`` crossing a process boundary, so the
graph failure taxonomy has to survive stringification. Encoding it as a leading
``{code}: {detail}`` token keeps the classification machine-readable without
widening the wire schema, and keeps producer and consumer from drifting apart
the way an ad-hoc prefix constant did.

Note the interaction with ``CreditContext.error``, which is ``str |
ErrorDetails | None``: the worker's return path sends ``str(context.error)``,
and ``ErrorDetails.__str__`` embeds ``code=``/``type=``/``cause=`` noise around
the message. A code must therefore be assigned as a bare ``str`` (what
:func:`format_graph_error` returns) to stay parseable downstream --
:func:`parse_graph_error` deliberately refuses to find a code buried inside an
``ErrorDetails`` repr rather than silently misclassifying it.
"""

from __future__ import annotations

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum

__all__ = ["GraphErrorCode", "format_graph_error", "parse_graph_error"]

_SEPARATOR = ": "


class GraphErrorCode(CaseInsensitiveStrEnum):
    """Machine-readable classification of a pre-dispatch graph failure."""

    #: Also known as a STICKINESS failure: the dispatch adapter converts this
    #: code into ``GraphStickinessError`` (``graph/credit_dispatch_adapter.py``),
    #: which the executor treats as a non-containable trace stop. Emitted when a
    #: dynamic slot's pooled value is absent on the routed worker -- broken
    #: stickiness after a worker re-route, or a dynamic-pool backstop eviction.
    POOL_MISSING = "aiperf.graph.pool_missing"
    #: Emitted when a captured node's reply cannot be reduced to a pool value
    #: (see ``Worker._graph_capture_value``).
    CAPTURE_FAILED = "aiperf.graph.capture_failed"


def format_graph_error(code: GraphErrorCode, detail: str) -> str:
    """Render ``code`` and ``detail`` as the wire error string."""
    return f"{code}{_SEPARATOR}{detail}"


def parse_graph_error(error: str | None) -> GraphErrorCode | None:
    """Recover the :class:`GraphErrorCode` from a wire error string.

    The code must LEAD the string. A stringified ``ErrorDetails`` wraps the
    message in ``message='...'``, so it correctly yields ``None`` rather than
    being misclassified by a naive substring match.

    Example:
        >>> wire = format_graph_error(GraphErrorCode.POOL_MISSING, "trace=t-1")
        >>> wire
        'aiperf.graph.pool_missing: trace=t-1'
        >>> parse_graph_error(wire) is GraphErrorCode.POOL_MISSING
        True
        >>> parse_graph_error("upstream returned 500") is None
        True
        >>> # An ErrorDetails repr buries the code, so it does NOT match. Assign
        >>> # format_graph_error's bare str -- never wrap it in ErrorDetails.
        >>> parse_graph_error("message='aiperf.graph.pool_missing: x' code=500") is None
        True
    """
    if not error:
        return None
    head, sep, _ = error.partition(_SEPARATOR)
    if not sep:
        return None
    try:
        return GraphErrorCode(head)
    except ValueError:
        return None
