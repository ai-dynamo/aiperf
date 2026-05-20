# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Round-trip tests for the cache-bust fields on the records-pipeline models.

Covers Slice 2 of the cache-bust subsystem: ``RecordContext`` and
``MetricRecordInfo`` carry the marker text + target enum so the raw-JSONL
consumer can correlate inserted bytes with the originating session.
"""

import orjson

from aiperf.common.enums import CacheBustTarget, CreditPhase
from aiperf.common.models.error_models import ErrorDetails
from aiperf.common.models.record_models import (
    MetricRecordInfo,
    MetricRecordMetadata,
    RecordContext,
)


def _make_record_context(**overrides) -> RecordContext:
    defaults = dict(
        credit_num=0,
        credit_phase=CreditPhase.PROFILING,
        conversation_id="c",
        turn_index=0,
        x_request_id="r",
        x_correlation_id="x",
    )
    defaults.update(overrides)
    return RecordContext(**defaults)


class TestRecordContextCacheBust:
    def test_defaults(self):
        ctx = _make_record_context()
        assert ctx.cache_bust_marker is None
        assert ctx.cache_bust_target is None

    def test_explicit_fields(self):
        ctx = _make_record_context(
            cache_bust_marker="\n<!-- cb:rc -->\n",
            cache_bust_target=CacheBustTarget.SYSTEM_PREFIX,
        )
        assert ctx.cache_bust_marker == "\n<!-- cb:rc -->\n"
        assert ctx.cache_bust_target is CacheBustTarget.SYSTEM_PREFIX

    def test_json_roundtrip_with_values(self):
        ctx = _make_record_context(
            cache_bust_marker="\n<!-- cb:json -->\n",
            cache_bust_target=CacheBustTarget.FIRST_TURN_SUFFIX,
        )
        dumped = orjson.dumps(ctx.model_dump(mode="json"))
        loaded = orjson.loads(dumped)
        roundtrip = RecordContext.model_validate(loaded)
        assert roundtrip.cache_bust_marker == "\n<!-- cb:json -->\n"
        assert roundtrip.cache_bust_target is CacheBustTarget.FIRST_TURN_SUFFIX

    def test_json_roundtrip_defaults_none(self):
        ctx = _make_record_context()
        roundtrip = RecordContext.model_validate(
            orjson.loads(orjson.dumps(ctx.model_dump(mode="json")))
        )
        assert roundtrip.cache_bust_marker is None
        assert roundtrip.cache_bust_target is None


class TestMetricRecordInfoCacheBust:
    def _make(self, **overrides) -> MetricRecordInfo:
        metadata = MetricRecordMetadata(
            session_num=0,
            request_start_ns=1,
            request_end_ns=2,
            worker_id="w",
            record_processor_id="rp",
            benchmark_phase=CreditPhase.PROFILING,
        )
        defaults = dict(metadata=metadata, metrics={})
        defaults.update(overrides)
        return MetricRecordInfo(**defaults)

    def test_defaults(self):
        info = self._make()
        assert info.cache_bust_marker is None
        assert info.cache_bust_target is None

    def test_explicit_fields(self):
        info = self._make(
            cache_bust_marker="\n<!-- cb:mri -->\n",
            cache_bust_target=CacheBustTarget.FIRST_TURN_PREFIX,
        )
        assert info.cache_bust_marker == "\n<!-- cb:mri -->\n"
        assert info.cache_bust_target is CacheBustTarget.FIRST_TURN_PREFIX

    def test_json_roundtrip(self):
        info = self._make(
            cache_bust_marker="\n<!-- cb:rt -->\n",
            cache_bust_target=CacheBustTarget.SYSTEM_SUFFIX,
            error=ErrorDetails(type="x", code=500, message="boom"),
        )
        dumped = orjson.dumps(info.model_dump(mode="json"))
        roundtrip = MetricRecordInfo.model_validate(orjson.loads(dumped))
        assert roundtrip.cache_bust_marker == "\n<!-- cb:rt -->\n"
        assert roundtrip.cache_bust_target is CacheBustTarget.SYSTEM_SUFFIX
