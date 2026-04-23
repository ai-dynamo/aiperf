# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregator re-exports for record-related models.

The implementations live in sibling modules, grouped by concern, so each
file stays under the 500-line ergonomics ceiling. Existing imports of the
form ``from aiperf.common.models.record_models import X`` keep working.
"""

from aiperf.common.models.dataset_models import Turn
from aiperf.common.models.inference_response_models import (
    BinaryResponse,
    InferenceServerResponse,
    SSEField,
    SSEMessage,
    TextResponse,
)
from aiperf.common.models.metric_result_models import (
    MetricResult,
    MetricValue,
    ProcessRecordsResult,
    ProfileResults,
)
from aiperf.common.models.parsed_response_models import (
    BaseResponseData,
    EmbeddingResponseData,
    ImageDataItem,
    ImageResponseData,
    ImageRetrievalResponseData,
    ParsedResponse,
    ParsedResponseRecord,
    RAGSources,
    RankingsResponseData,
    ReasoningResponseData,
    TextResponseData,
    TokenCounts,
    VideoResponseData,
)
from aiperf.common.models.record_export_models import (
    MetricRecordInfo,
    RawRecordInfo,
    decode_metric_record_info_json,
    decode_raw_record_info_json,
)
from aiperf.common.models.request_record_models import (
    RequestInfo,
    RequestRecord,
)

__all__ = [
    "BaseResponseData",
    "BinaryResponse",
    "EmbeddingResponseData",
    "ImageDataItem",
    "ImageResponseData",
    "ImageRetrievalResponseData",
    "InferenceServerResponse",
    "MetricRecordInfo",
    "MetricResult",
    "MetricValue",
    "ParsedResponse",
    "ParsedResponseRecord",
    "ProcessRecordsResult",
    "ProfileResults",
    "RAGSources",
    "RankingsResponseData",
    "RawRecordInfo",
    "ReasoningResponseData",
    "RequestInfo",
    "RequestRecord",
    "SSEField",
    "SSEMessage",
    "TextResponse",
    "TextResponseData",
    "TokenCounts",
    "Turn",
    "VideoResponseData",
    "decode_metric_record_info_json",
    "decode_raw_record_info_json",
]
