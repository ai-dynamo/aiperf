# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared wire output-cap resolution for recorded-trace adapters (weka, dynamo).

Recorded replay always pins each call's generation to the recording via the
native ``LlmNode.max_tokens`` field (``Turn.max_tokens`` naming) -- folded into
the store envelope by ``store_builder._trie_envelope`` and mapped by the worker
to the endpoint's wire token field (``max_completion_tokens``, or
``max_tokens`` under ``--use-legacy-max-tokens``; see
``graph/worker_materialize.py::_apply_dispatch_overrides``). Both adapters
resolve the cap through :func:`wire_output_cap` so zero-output turns are
handled identically.
"""

from __future__ import annotations

from aiperf.common.aiperf_logger import AIPerfLogger

_logger = AIPerfLogger(__name__)


def wire_output_cap(recorded_out: int, *, node_id: str) -> int:
    """Resolve a node's recorded output length to its wire generation cap.

    A recorded length of 0 (a zero-output or aborted turn, or a capture with no
    ``output_tokens``) is not a sendable cap -- ``max_tokens: 0`` is rejected or
    degenerate on OpenAI-compatible servers -- so it upgrades to 1 with a
    warning rather than replaying that turn unbounded or uncapped.
    """
    if recorded_out > 0:
        return recorded_out
    _logger.warning(
        f"node {node_id!r}: recorded output length is 0; upgrading the wire "
        "generation cap to max_output_tokens=1"
    )
    return 1
