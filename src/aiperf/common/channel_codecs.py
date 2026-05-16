# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Channel-specific transport codec instances."""

from __future__ import annotations

from typing import Any

import msgspec

from aiperf.common.inference_wire import InferenceResultsWireMessage
from aiperf.common.message_codecs import MsgspecStructCodec, get_message_codec
from aiperf.common.metric_records_wire import (
    MetricRecordsBatchWireMessage,
    MetricRecordsWireMessage,
)

RAW_INFERENCE_CODEC = MsgspecStructCodec(
    decode_type=InferenceResultsWireMessage,
    cache_key="raw-inference-msgpack",
)

class RecordsChannelCodec:
    cache_key = "records-msgpack"

    def __init__(self) -> None:
        self._records_codec = MsgspecStructCodec(
            decode_type=(MetricRecordsWireMessage | MetricRecordsBatchWireMessage),
            cache_key=self.cache_key,
        )

    def encode(self, message: Any) -> bytes:
        return self._records_codec.encode(message)

    def decode(self, data: bytes) -> Any:
        try:
            return self._records_codec.decode(data)
        except msgspec.ValidationError:
            return get_message_codec().decode(data)


RECORDS_CODEC = RecordsChannelCodec()
