# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""msgspec-tagged-union round-trip coverage for ``InferenceServerResponseUnion``.

Task 3a.1 migrated ``SSEMessage`` / ``TextResponse`` / ``BinaryResponse`` from
``@dataclass(slots=True)`` (with ``__pydantic_config__``) to
``msgspec.Struct`` discriminated by ``tag="..."``. These tests confirm:

- Each variant encodes and decodes via msgspec.json with the correct tag.
- A ``msgspec.json.Decoder(InferenceServerResponseUnion)`` dispatches to the
  right subclass purely off the tag field.
- Binary content survives the JSON wire trip (msgspec.json base64-encodes
  bytes by default — no custom serializer needed).
- The bridge inside Pydantic ``RequestRecord.responses`` (field_serializer +
  field_validator) round-trips a mixed list with the correct types restored.
"""

from __future__ import annotations

import msgspec
import pytest
from pytest import param

from aiperf.common.models.record_models import (
    BinaryResponse,
    InferenceServerResponseUnion,
    RequestRecord,
    SSEField,
    SSEMessage,
    TextResponse,
)


def _decode_as_union(
    value: InferenceServerResponseUnion,
) -> InferenceServerResponseUnion:
    """Encode via msgspec.json and decode against the discriminated union."""
    encoded = msgspec.json.encode(value)
    return msgspec.json.decode(encoded, type=InferenceServerResponseUnion)


class TestStandaloneRoundTrip:
    """Each response Struct round-trips via msgspec.json with tag dispatch."""

    def test_sse_message_roundtrip(self) -> None:
        msg = SSEMessage(
            perf_ns=100,
            packets=[SSEField(name="data", value="hello")],
        )
        back = _decode_as_union(msg)
        assert isinstance(back, SSEMessage)
        assert back.perf_ns == 100
        assert len(back.packets) == 1
        assert back.packets[0].name == "data"
        assert back.packets[0].value == "hello"

    def test_text_response_roundtrip(self) -> None:
        msg = TextResponse(perf_ns=200, text="payload", content_type="text/plain")
        back = _decode_as_union(msg)
        assert isinstance(back, TextResponse)
        assert back.perf_ns == 200
        assert back.text == "payload"
        assert back.content_type == "text/plain"

    def test_binary_response_roundtrip_preserves_non_utf8_bytes(self) -> None:
        """Binary content survives the JSON wire via msgspec.json's base64 default."""
        msg = BinaryResponse(
            perf_ns=300,
            raw_bytes=b"\x00\x01\xff\xfe",
            content_type="application/octet-stream",
        )
        back = _decode_as_union(msg)
        assert isinstance(back, BinaryResponse)
        assert back.perf_ns == 300
        assert back.raw_bytes == b"\x00\x01\xff\xfe"
        assert back.content_type == "application/octet-stream"

    def test_tag_field_present_on_wire(self) -> None:
        """Every variant emits the discriminating ``type`` tag in its encoded form."""
        cases = (
            (SSEMessage(perf_ns=1), "sse"),
            (TextResponse(perf_ns=2, text="x"), "text"),
            (BinaryResponse(perf_ns=3, raw_bytes=b""), "binary"),
        )
        for msg, expected_tag in cases:
            blob = msgspec.json.encode(msg)
            decoded = msgspec.json.decode(blob, type=dict)
            assert decoded["type"] == expected_tag


class TestUnionDispatch:
    """The tagged union decoder picks the right subtype for every variant."""

    @pytest.mark.parametrize(
        "factory,expected_type",
        [
            param(lambda: SSEMessage(perf_ns=1), SSEMessage, id="sse"),
            param(
                lambda: TextResponse(perf_ns=2, text="x"),
                TextResponse,
                id="text",
            ),
            param(
                lambda: BinaryResponse(perf_ns=3, raw_bytes=b"\x00"),
                BinaryResponse,
                id="binary",
            ),
        ],
    )  # fmt: skip
    def test_union_dispatch(self, factory, expected_type) -> None:
        msg = factory()
        back = _decode_as_union(msg)
        assert isinstance(back, expected_type)


class TestRequestRecordBridge:
    """``RequestRecord`` (Task 3a.4: now msgspec.Struct) ferries the response union natively."""

    def test_mixed_responses_roundtrip(self) -> None:
        responses: list[InferenceServerResponseUnion] = [
            SSEMessage(perf_ns=10, packets=[SSEField(name="data", value="chunk")]),
            TextResponse(perf_ns=20, text="body"),
            BinaryResponse(perf_ns=30, raw_bytes=b"\x10\x20\x30"),
        ]
        rec = RequestRecord(responses=responses, status=200)
        rt = msgspec.json.decode(msgspec.json.encode(rec), type=RequestRecord)
        assert [type(r).__name__ for r in rt.responses] == [
            "SSEMessage",
            "TextResponse",
            "BinaryResponse",
        ]
        sse = rt.responses[0]
        assert isinstance(sse, SSEMessage)
        assert sse.packets[0].value == "chunk"
        binary = rt.responses[2]
        assert isinstance(binary, BinaryResponse)
        assert binary.raw_bytes == b"\x10\x20\x30"

    def test_empty_responses_roundtrip(self) -> None:
        rec = RequestRecord(status=204)
        rt = msgspec.json.decode(msgspec.json.encode(rec), type=RequestRecord)
        assert rt.responses == []
