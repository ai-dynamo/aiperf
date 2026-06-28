# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for binary-response JSON serialization.

BinaryResponse.raw_bytes crosses the JSON-encoded ZMQ bus and raw exports.
Pydantic's default bytes->JSON is utf8, which raises on non-text audio/video
payloads; the Base64Bytes annotation + field_serializer on the responses list
(routing around SerializeAsAny) base64-encode it instead. These tests pin that
behavior so a regression surfaces here rather than only in a live run.
"""

import base64

import orjson

from aiperf.common.models import RequestRecord
from aiperf.common.models.record_models import BinaryResponse, SSEMessage, TextResponse

# Non-utf8 bytes: an mp3 frame header (0xff 0xfb) plus bytes that utf8 rejects.
_AUDIO = bytes([0xFF, 0xFB, 0x00, 0xA4, 0x80, 0xE4])


def _request_record(responses: list) -> RequestRecord:
    return RequestRecord(
        start_perf_ns=1,
        timestamp_ns=1,
        end_perf_ns=10,
        status=200,
        responses=responses,
    )


class TestBinaryResponseSerialization:
    def test_binary_bytes_base64_encoded_and_orjson_safe(self):
        rec = _request_record(
            [BinaryResponse(perf_ns=2, raw_bytes=_AUDIO, content_type="audio/wav")]
        )
        dumped = rec.model_dump(exclude_none=True, mode="json")
        raw = dumped["responses"][0]["raw_bytes"]
        assert isinstance(raw, str)
        assert base64.b64decode(raw) == _AUDIO
        # The whole point: orjson must not raise UnicodeDecodeError on the dump.
        orjson.dumps(dumped)

    def test_binary_json_roundtrip_restores_exact_bytes(self):
        rec = _request_record(
            [BinaryResponse(perf_ns=2, raw_bytes=_AUDIO, content_type="audio/mpeg")]
        )
        blob = orjson.dumps(rec.model_dump(exclude_none=True, mode="json"))
        restored = RequestRecord.model_validate(orjson.loads(blob))
        assert restored.responses[0].raw_bytes == _AUDIO
        assert restored.responses[0].content_type == "audio/mpeg"

    def test_mixed_responses_list_roundtrip_preserves_types(self):
        rec = _request_record(
            [
                SSEMessage.parse(b'data: {"a": 1}', 1),
                TextResponse(perf_ns=2, text="hi", content_type="text/plain"),
                BinaryResponse(perf_ns=3, raw_bytes=_AUDIO, content_type="audio/wav"),
            ]
        )
        blob = orjson.dumps(rec.model_dump(exclude_none=True, mode="json"))
        restored = RequestRecord.model_validate(orjson.loads(blob))

        assert isinstance(restored.responses[0], SSEMessage)
        assert isinstance(restored.responses[1], TextResponse)
        assert isinstance(restored.responses[2], BinaryResponse)
        assert restored.responses[1].get_text() == "hi"
        assert restored.responses[2].raw_bytes == _AUDIO

    def test_in_memory_bytes_not_double_encoded(self):
        rec = _request_record([BinaryResponse(perf_ns=2, raw_bytes=_AUDIO)])
        # python mode keeps the raw bytes (no base64 round-trip in memory).
        assert rec.responses[0].raw_bytes == _AUDIO
        assert rec.model_dump(mode="python")["responses"][0]["raw_bytes"] == _AUDIO
