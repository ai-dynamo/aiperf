# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import MessageType
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models import RequestRecord
from aiperf.common.models.record_models import MetricResult


class InferenceResultsMessage(
    BaseServiceMessage, kw_only=True, tag=MessageType.INFERENCE_RESULTS.value
):
    """Single inference result record."""

    record: RequestRecord


class RealtimeMetricsMessage(
    BaseServiceMessage, kw_only=True, tag=MessageType.REALTIME_METRICS.value
):
    """Real-time metrics summary."""

    metrics: list[MetricResult]
