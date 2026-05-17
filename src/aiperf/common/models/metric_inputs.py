# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wire-only record-pipeline input."""

from __future__ import annotations

import msgspec

from aiperf.common.enums import CreditPhase


class MetricInputs(msgspec.Struct, kw_only=True):
    """Per-record routing + payload addressing shipped from worker to record processor.

    Wire-only msgspec.Struct. msgspec-encoded for speed and binary-safety on
    the records-pipeline channel (msgpack post-Phase-3c). The records pipeline
    reads *only* this off the wire; static per-(conversation, turn) inputs
    (``max_tokens``, ``audio_duration_seconds``) come from ``DatasetMetadata``
    looked up by ``conversation_id`` + ``turn_index``.

    Payload addressing is one-of:

    (a) ``payload_bytes`` set to non-null ``bytes`` -> the pre-encoded JSON
        request payload rides the wire as a msgpack ``bin`` span (zero base64
        overhead, length-prefixed, byte-identical round-trip). Used for
        CONVERSATION-format datasets (worker computes via
        ``endpoint.format_payload``), error records, and any on-the-fly
        rewriting path.

    (b) ``payload_bytes`` set to ``None`` (the default) -> records process
        resolves via its own ``MemoryMapDatasetClientStore`` keyed by
        ``(conversation_id, turn_index)``. Used for PAYLOAD_BYTES datasets
        where the bytes are durable on disk and don't need to ride ZMQ.

    The ``payload_bytes_or_none`` / ``has_inline_payload`` accessors stay for
    source compatibility with the prior ``msgspec.Raw`` shape, though the
    underlying field is now plain ``bytes | None``.
    """

    credit_num: int
    credit_phase: CreditPhase
    conversation_id: str
    turn_index: int
    x_request_id: str
    x_correlation_id: str
    credit_issued_ns: int | None = None
    agent_depth: int = 0
    parent_correlation_id: str | None = None
    payload_bytes: bytes | None = None

    @property
    def payload_bytes_or_none(self) -> bytes | None:
        """The inline payload as plain ``bytes``, or ``None`` when absent."""
        return self.payload_bytes

    @property
    def has_inline_payload(self) -> bool:
        """Whether an inline payload is present on this record."""
        return self.payload_bytes is not None
