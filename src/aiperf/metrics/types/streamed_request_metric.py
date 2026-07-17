# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-record streaming predicate that gates every per-record streaming metric.

This metric's PRESENCE in a record's ``MetricRecordDict`` is the predicate: it
parses to 1 for a request that actually streamed on the wire
(``RequestRecord.streamed``), and raises ``NoMetricValue`` otherwise so it is
absent from that record's dict.

Streaming metrics whose ``_parse_record`` reads response timing directly declare
this tag in ``required_metrics`` (so the topological order computes it first) and
check membership; when the record did not stream the tag is absent and the
streaming metric skips itself instead of, e.g., reporting full request latency as
TTFT for a non-streamed record whose single response timestamp is the completion.

It is a ``BaseRecordMetric`` (not an aggregate counter) so RECORD-typed streaming
metrics may depend on it: the dependency validator only permits a RECORD metric to
depend on other RECORD metrics. It carries ``INTERNAL`` (excluded from the
``summarize()`` JSON/CSV export, where a constant-1 stat row is meaningless),
``NO_INDIVIDUAL_RECORDS`` (excluded from per-record exports), and
``console_group=NONE`` (hidden from the console); INTERNAL metrics stay computable
as dependencies since ``get_filters`` never disallows INTERNAL. The user-facing
streamed-request count is the companion ``StreamedRequestCountMetric`` aggregate.
"""

from aiperf.common.constants import STREAMED_REQUEST_TAG
from aiperf.common.enums import GenericMetricUnit, MetricConsoleGroup, MetricFlags
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict


class StreamedRequestMetric(BaseRecordMetric[int]):
    """Per-record streaming predicate that gates streaming metrics.

    Formula:
        ```
        Streamed Request = 1 if request.streamed else skip
        ```
    """

    tag = STREAMED_REQUEST_TAG
    header = "Streamed Request"
    short_header = "Streamed Request"
    short_header_hide_unit = True
    unit = GenericMetricUnit.REQUESTS
    display_order = 1102
    flags = (
        MetricFlags.STREAMING_TOKENS_ONLY
        | MetricFlags.NO_INDIVIDUAL_RECORDS
        | MetricFlags.INTERNAL
    )
    console_group = MetricConsoleGroup.NONE
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        """Return 1 iff the underlying request streamed; else raise to stay absent.

        Raises:
            NoMetricValue: If the request did not stream on the wire.
        """
        if not record.request.streamed:
            raise NoMetricValue("request did not stream")
        return 1
