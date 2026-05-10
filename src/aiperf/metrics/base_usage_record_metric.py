# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base class for metrics that read a single field from `record.final_usage`.

The vast majority of `Usage*` metrics share the same shape: extract one
property from the merged streaming usage, raise `NoMetricValue` when absent.
Subclasses provide just two extra class attributes (`usage_field` and
`missing_message`) instead of a duplicated `_parse_record` loop.

The streaming-walk-back logic lives once on `ParsedResponseRecord.final_usage`
(via `Usage.merge_streaming`); subclasses never re-implement it.
"""

from typing import ClassVar, Generic

from aiperf.common.enums import MetricValueTypeVarT
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict


class BaseUsageRecordMetric(
    BaseRecordMetric[MetricValueTypeVarT], Generic[MetricValueTypeVarT]
):
    """Reads `getattr(record.final_usage, usage_field)`.

    Subclass and set `usage_field` (the property name on `Usage`) and
    `missing_message` (the human-readable string raised by `NoMetricValue`
    when the field is absent in the merged usage). All other metric metadata
    — tag, header, unit, flags — is set the same way as on plain
    `BaseRecordMetric` subclasses.

    Example:
        class UsagePromptCacheReadTokensMetric(BaseUsageRecordMetric[int]):
            tag = "usage_prompt_cache_read_tokens"
            header = "Usage Prompt Cache Read Tokens"
            unit = GenericMetricUnit.TOKENS
            flags = MetricFlags.LARGER_IS_BETTER
            console_group = MetricConsoleGroup.USAGE
            required_metrics = None

            usage_field = "prompt_cache_read_tokens"
            missing_message = (
                "Usage prompt cache-read token count not available: ..."
            )
    """

    # The base class itself is not a registerable metric — it has no tag.
    # Subclasses flip this in __init_subclass__ so they DO register normally.
    __is_abstract__: ClassVar[bool] = True

    usage_field: ClassVar[str]
    """Name of the property to read from `record.final_usage`."""

    missing_message: ClassVar[str]
    """Human-readable detail raised inside `NoMetricValue` when the field
    is absent (either because no chunk had any usage, or because every
    chunk left the specific field as None)."""

    def __init_subclass__(cls, **kwargs) -> None:
        cls.__is_abstract__ = False
        return super().__init_subclass__(**kwargs)

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> MetricValueTypeVarT:
        usage = record.final_usage
        if usage is None:
            raise NoMetricValue(self.missing_message)
        value = getattr(usage, self.usage_field)
        if value is None:
            raise NoMetricValue(self.missing_message)
        return value
