# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal AWS ``application/vnd.amazon.eventstream`` binary frame encoder for
the mock server's SageMaker ``InvokeEndpointWithResponseStream`` route.

The pinned ``botocore`` ships an eventstream *decoder* (``EventStreamBuffer``)
but no public encoder, so the mock server hand-rolls the wire format here: a
prelude (total length + headers length + prelude CRC32), a headers block, the
payload, and a trailing full-message CRC32. Frames produced here round-trip
through the production :class:`aiperf.transports.sagemaker_eventstream.EventStreamReader`.
"""

from __future__ import annotations

import struct
import zlib

_HEADER_TYPE_STRING = 7


def _encode_header(name: str, value: str) -> bytes:
    """Encode one string-valued eventstream header (name length + name + type
    byte + value length + value)."""
    name_b = name.encode("utf-8")
    value_b = value.encode("utf-8")
    return (
        struct.pack(">B", len(name_b))
        + name_b
        + struct.pack(">B", _HEADER_TYPE_STRING)
        + struct.pack(">H", len(value_b))
        + value_b
    )


def encode_payload_part(payload: bytes) -> bytes:
    """Wrap ``payload`` in one ``PayloadPart`` eventstream frame with valid
    prelude and message CRC32 checksums."""
    headers = (
        _encode_header(":message-type", "event")
        + _encode_header(":event-type", "PayloadPart")
        + _encode_header(":content-type", "application/json")
    )
    headers_len = len(headers)
    total_len = 4 + 4 + 4 + headers_len + len(payload) + 4
    prelude = struct.pack(">II", total_len, headers_len)
    prelude_crc = struct.pack(">I", zlib.crc32(prelude) & 0xFFFFFFFF)
    message_no_crc = prelude + prelude_crc + headers + payload
    message_crc = struct.pack(">I", zlib.crc32(message_no_crc) & 0xFFFFFFFF)
    return message_no_crc + message_crc
