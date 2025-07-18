# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class SSEFieldType(CaseInsensitiveStrEnum):
    """Field types in an SSE message."""

    DATA = "data"
    EVENT = "event"
    ID = "id"
    RETRY = "retry"
    COMMENT = "comment"


class SSEEventType(CaseInsensitiveStrEnum):
    """Event types in an SSE message. Many of these are custom and not defined by the SSE spec."""

    ERROR = "error"
    LLM_METRICS = "llm_metrics"
