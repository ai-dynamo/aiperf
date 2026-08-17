# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Worker call-site coverage for dynamo session-id uniquification.

``Worker._resolve_graph_session_headers`` is the only caller of
:func:`aiperf.graph.worker_materialize.uniquify_dynamo_session_headers`.
Nothing else at unit level reaches it, so a signature change to the callee
(``phase_variant: str`` -> ``phase: CreditPhase``) shipped without the call
site being updated and every graph credit died with a ``TypeError``. These
tests pin the wiring: the call must supply BOTH the trace instance id and the
credit's phase, and the phase must actually reach the emitted suffix.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock

from aiperf.common.enums import CreditPhase
from aiperf.credit.structs import Credit
from aiperf.workers.worker import Worker

TRACE_INSTANCE = "t-1::inst0"
ENVELOPE = {
    "extra_headers": {
        "x-dynamo-session-id": "sess-abc",
        "x-dynamo-parent-session-id": "sess-parent",
        "x-dynamo-session-final": "true",
    }
}


def _worker_with_resolver() -> MagicMock:
    """Mock worker self carrying the REAL header-resolution method."""
    self = MagicMock()
    self._resolve_graph_session_headers = types.MethodType(
        Worker._resolve_graph_session_headers, self
    )
    return self


def _credit(phase: CreditPhase) -> Credit:
    return Credit(
        id=1,
        phase=phase,
        conversation_id="t-1",
        x_correlation_id="t-1::corr0",
        turn_index=0,
        num_turns=1,
        issued_at_ns=0,
        trace_id=TRACE_INSTANCE,
        node_ordinal=0,
    )


def test_resolve_graph_session_headers_suffixes_identity_headers() -> None:
    """Both identity headers get the ``::{phase}-{nonce}`` suffix; final is untouched."""
    worker = _worker_with_resolver()

    headers = worker._resolve_graph_session_headers(
        ENVELOPE, _credit(CreditPhase.PROFILING)
    )

    assert headers is not None
    assert headers["x-dynamo-session-id"] == "sess-abc::profiling-inst0"
    assert headers["x-dynamo-parent-session-id"] == "sess-parent::profiling-inst0"
    # session-final is forwarded verbatim: each instance closes only its own session.
    assert headers["x-dynamo-session-final"] == "true"
    # Input envelope is not mutated.
    assert ENVELOPE["extra_headers"]["x-dynamo-session-id"] == "sess-abc"


def test_resolve_graph_session_headers_warmup_and_profiling_differ() -> None:
    """The invariant the phase argument exists for: same trace instance, distinct
    sessions across phases, so warmup's session-final cannot evict profiling KV."""
    worker = _worker_with_resolver()

    warmup = worker._resolve_graph_session_headers(
        ENVELOPE, _credit(CreditPhase.WARMUP)
    )
    profiling = worker._resolve_graph_session_headers(
        ENVELOPE, _credit(CreditPhase.PROFILING)
    )

    assert warmup is not None and profiling is not None
    assert warmup["x-dynamo-session-id"] == "sess-abc::warmup-inst0"
    assert warmup["x-dynamo-session-id"] != profiling["x-dynamo-session-id"], (
        "warmup and profiling instances of one slot must not share a server session"
    )
