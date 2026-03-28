# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Channel-specific transport codec instances."""

from __future__ import annotations

from aiperf.common.inference_wire import InferenceResultsWireMessage
from aiperf.common.message_codecs import MsgspecStructCodec
from aiperf.common.metric_records_wire import (
    MetricRecordsBatchWireMessage,
    MetricRecordsWireMessage,
)

RAW_INFERENCE_CODEC = MsgspecStructCodec(
    decode_type=InferenceResultsWireMessage,
    cache_key="raw-inference-msgpack",
)

RECORDS_CODEC = MsgspecStructCodec(
    decode_type=MetricRecordsWireMessage | MetricRecordsBatchWireMessage,
    cache_key="records-msgpack",
)
