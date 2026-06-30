# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial coverage for ``RawRecordWriterProcessor`` Fragment splicing.

These tests pin the *current* behaviour of the ``payload_bytes`` fast path
in ``buffered_write`` — including the known Wave-2 bug where a broad
``except Exception`` silently drops records that explode during
serialisation. One ``xfail(strict=True)`` case documents the desired
post-fix behaviour (propagate or increment a counter).
"""

from typing import Any

import orjson
import pytest

from aiperf.common.config import UserConfig
from aiperf.common.enums import CreditPhase, ModelSelectionStrategy
from aiperf.common.models import (
    ParsedResponse,
    ParsedResponseRecord,
    RequestInfo,
    RequestRecord,
    TextResponse,
)
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.models.record_models import (
    ErrorDetails,
    RawRecordInfo,
    TokenCounts,
)
from aiperf.plugin.enums import EndpointType
from aiperf.post_processors.raw_record_writer_processor import (
    RawRecordAggregator,
    RawRecordWriterProcessor,
)
from tests.unit.post_processors.conftest import (
    create_exporter_config,
    create_metric_metadata,
    raw_record_processor,
)


def _make_request_info(
    *,
    payload_bytes: bytes | None,
    conversation_id: str = "conv-adv",
) -> RequestInfo:
    return RequestInfo(
        model_endpoint=ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="test-model")],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.RAW,
                base_url="http://localhost:8000",
            ),
        ),
        turns=[],
        payload_bytes=payload_bytes,
        turn_index=0,
        credit_num=0,
        credit_phase=CreditPhase.PROFILING,
        x_request_id="req-adv",
        x_correlation_id="corr-adv",
        conversation_id=conversation_id,
    )


def _make_parsed_record(
    *,
    payload_bytes: bytes | None,
    conversation_id: str = "conv-adv",
    status: int = 200,
    error: ErrorDetails | None = None,
) -> ParsedResponseRecord:
    from aiperf.common.models import TextResponseData

    request = RequestRecord(
        request_info=_make_request_info(
            payload_bytes=payload_bytes,
            conversation_id=conversation_id,
        ),
        model_name="test-model",
        start_perf_ns=1_000_000_000,
        timestamp_ns=1_000_000_000,
        end_perf_ns=2_000_000_000,
        status=status,
        request_headers={"Content-Type": "application/json"},
        responses=[TextResponse(text="ok", perf_ns=2_000_000_000)],
        error=error,
    )
    return ParsedResponseRecord(
        request=request,
        responses=[
            ParsedResponse(perf_ns=2_000_000_000, data=TextResponseData(text="ok"))
        ],
        token_counts=TokenCounts(input=1, output=1, reasoning=None),
    )


def _make_raw_record(
    *,
    payload_bytes: Any,
    payload: dict[str, Any] | None = None,
) -> RawRecordInfo:
    """Build a ``RawRecordInfo`` directly, bypassing ``_build_export_record``."""
    return RawRecordInfo(
        metadata=create_metric_metadata(),
        start_perf_ns=1_000_000_000,
        payload=payload,
        payload_bytes=payload_bytes,
        request_headers={},
        response_headers=None,
        status=200,
        responses=[TextResponse(text="ok", perf_ns=2_000_000_000)],
        error=None,
    )


class TestBufferedWritePayloadBytesFastPath:
    """Pin current behaviour of ``buffered_write``'s ``payload_bytes`` fast path."""

    @pytest.mark.asyncio
    async def test_buffered_write_none_payload_bytes_falls_through_to_generic_mixin_path(
        self,
        user_config_raw: UserConfig,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """When ``payload_bytes is None``, the override must delegate to the
        mixin's generic ``buffered_write`` (``model_dump`` serialisation).
        """
        from aiperf.common.mixins.buffered_jsonl_writer_mixin import (
            BufferedJSONLWriterMixin,
        )

        called: dict[str, int] = {"count": 0}
        original = BufferedJSONLWriterMixin.buffered_write

        async def spy(self, record):
            called["count"] += 1
            return await original(self, record)

        monkeypatch.setattr(BufferedJSONLWriterMixin, "buffered_write", spy)

        record = _make_raw_record(payload_bytes=None, payload={"k": "v"})

        async with raw_record_processor("processor-none", user_config_raw) as processor:
            await processor.buffered_write(record)

        assert called["count"] == 1, (
            "payload_bytes=None must delegate to the generic mixin path via "
            "super().buffered_write()"
        )
        lines = processor.output_file.read_text().splitlines()
        assert len(lines) == 1
        parsed = orjson.loads(lines[0])
        assert parsed["payload"] == {"k": "v"}

    @pytest.mark.asyncio
    async def test_buffered_write_empty_bytes_payload_bytes_spliced_verbatim(
        self,
        user_config_raw: UserConfig,
    ):
        """``payload_bytes=b""`` — the per-record JSON re-validation was removed,
        so empty bytes splice verbatim (no drop, no counter). orjson emits
        ``"payload":`` with no value, i.e. a deliberately corrupt line: the
        accepted tradeoff of trusting dataset-load-time validation instead of
        re-parsing every record on the export hot path.
        """
        record = _make_raw_record(payload_bytes=b"")

        async with raw_record_processor(
            "processor-empty", user_config_raw
        ) as processor:
            await processor.buffered_write(record)
            assert processor.dropped_record_count == 0
            assert processor.lines_written == 1

        line = processor.output_file.read_bytes().splitlines()[0]
        # Spliced verbatim -> the line is no longer valid JSON.
        with pytest.raises(orjson.JSONDecodeError):
            orjson.loads(line)

    @pytest.mark.asyncio
    async def test_buffered_write_invalid_json_payload_bytes_spliced_verbatim(
        self,
        user_config_raw: UserConfig,
    ):
        """``payload_bytes=b"}"`` — with the per-record ``orjson.loads`` ingest
        check removed, invalid bytes splice verbatim via ``orjson.Fragment``
        (no drop, no counter). The emitted line is corrupt; this documents the
        accepted tradeoff of trusting upstream/dataset-load validation.
        """
        record = _make_raw_record(payload_bytes=b"}")

        async with raw_record_processor(
            "processor-bad-json", user_config_raw
        ) as processor:
            await processor.buffered_write(record)
            assert processor.dropped_record_count == 0
            assert processor.lines_written == 1

        line = processor.output_file.read_bytes().splitlines()[0]
        with pytest.raises(orjson.JSONDecodeError):
            orjson.loads(line)

    @pytest.mark.asyncio
    async def test_buffered_write_truncated_json_payload_bytes_spliced_verbatim(
        self,
        user_config_raw: UserConfig,
    ):
        """Truncated JSON ``b'{"a":1'`` — with the ingest ``orjson.loads`` check
        removed, the partial bytes splice verbatim (no drop, no counter) and the
        emitted line is corrupt: the accepted tradeoff of skipping a per-record
        re-parse on the export hot path.
        """
        record = _make_raw_record(payload_bytes=b'{"a":1')

        async with raw_record_processor(
            "processor-trunc", user_config_raw
        ) as processor:
            await processor.buffered_write(record)
            assert processor.dropped_record_count == 0
            assert processor.lines_written == 1

        line = processor.output_file.read_bytes().splitlines()[0]
        with pytest.raises(orjson.JSONDecodeError):
            orjson.loads(line)

    @pytest.mark.asyncio
    async def test_buffered_write_payload_bytes_with_trailing_whitespace_still_valid_fragment(
        self,
        user_config_raw: UserConfig,
    ):
        """Trailing whitespace inside the payload bytes is also spliced
        verbatim. With current behaviour this embeds whitespace between
        the payload value and the subsequent comma — the JSONL line is
        still parseable by orjson (whitespace is tolerated inside JSON
        objects).
        """
        record = _make_raw_record(payload_bytes=b'{"a":1}  \n')

        async with raw_record_processor("processor-ws", user_config_raw) as processor:
            await processor.buffered_write(record)

        raw = processor.output_file.read_bytes().rstrip(b"\n")
        # The trailing whitespace from payload_bytes lives inside the line
        assert b'{"a":1}  \n' in raw
        # And the line still parses cleanly
        parsed = orjson.loads(raw)
        assert parsed["payload"] == {"a": 1}

    @pytest.mark.asyncio
    async def test_buffered_write_payload_bytes_containing_nul_byte_behavior(
        self,
        user_config_raw: UserConfig,
    ):
        """A NUL *escape* (``\\u0000``) inside a JSON string is valid JSON
        and must round-trip through the Fragment splice path untouched.
        """
        payload_bytes = b'{"a":"\\u0000"}'
        record = _make_raw_record(payload_bytes=payload_bytes)

        async with raw_record_processor("processor-nul", user_config_raw) as processor:
            await processor.buffered_write(record)

        raw = processor.output_file.read_bytes().rstrip(b"\n")
        assert payload_bytes in raw
        parsed = orjson.loads(raw)
        assert parsed["payload"] == {"a": "\x00"}

    @pytest.mark.asyncio
    async def test_buffered_write_extremely_large_payload_bytes_1mb_splices_clean(
        self,
        user_config_raw: UserConfig,
    ):
        """1 MB of valid JSON must splice cleanly without re-encoding."""
        large_string = "a" * (1024 * 1024)
        payload_dict = {"model": "m", "prompt": large_string}
        payload_bytes = orjson.dumps(payload_dict)
        assert len(payload_bytes) >= 1024 * 1024

        record = _make_raw_record(payload_bytes=payload_bytes)

        async with raw_record_processor(
            "processor-large", user_config_raw
        ) as processor:
            await processor.buffered_write(record)

        raw = processor.output_file.read_bytes().rstrip(b"\n")
        # The verbatim bytes appear as a substring
        assert payload_bytes in raw
        parsed = orjson.loads(raw)
        assert parsed["payload"] == payload_dict

    @pytest.mark.asyncio
    async def test_buffered_write_non_json_non_bytes_payload_bytes_dropped_with_counter(
        self,
        user_config_raw: UserConfig,
    ):
        """``payload_bytes=123`` (int) — even without the per-record JSON
        re-validation, ``orjson.Fragment(123)`` raises ``TypeError`` at
        construction, which the serialisation ``except`` catches so the record
        is dropped with the counter bumped (genuine serialisation failures are
        still surfaced).

        We construct the ``RawRecordInfo`` via ``model_construct`` because
        pydantic validation would reject ``payload_bytes=123``.
        """
        record = RawRecordInfo.model_construct(
            metadata=create_metric_metadata(),
            start_perf_ns=1_000_000_000,
            payload=None,
            payload_bytes=123,  # type: ignore[arg-type]
            request_headers={},
            response_headers=None,
            status=200,
            responses=[TextResponse(text="ok", perf_ns=2_000_000_000)],
            error=None,
        )

        async with raw_record_processor("processor-int", user_config_raw) as processor:
            await processor.buffered_write(record)
            assert processor.lines_written == 0
            assert processor.dropped_record_count == 1

        # No record made it to disk (file may be deleted when no lines written)
        assert (
            not processor.output_file.exists()
            or not processor.output_file.read_text().strip()
        )

    @pytest.mark.asyncio
    async def test_buffered_write_flush_triggered_when_buffer_reaches_batch_size(
        self,
        user_config_raw: UserConfig,
    ):
        """After ``batch_size`` writes the buffer is drained and scheduled
        for async flush; ``lines_written`` increments per-record and the
        in-memory ``_buffer`` is emptied.
        """
        async with raw_record_processor(
            "processor-flush", user_config_raw
        ) as processor:
            batch_size = processor._batch_size
            assert batch_size >= 1

            payload_bytes = b'{"k":"v"}'
            for _ in range(batch_size):
                await processor.buffered_write(
                    _make_raw_record(payload_bytes=payload_bytes)
                )

            assert processor.lines_written == batch_size
            # Buffer should have been handed off to a flush task
            assert processor._buffer == []

        # After stop(), the flush has completed and file has N lines
        lines = processor.output_file.read_text().splitlines()
        assert len(lines) == batch_size
        for line in lines:
            assert orjson.loads(line)["payload"] == {"k": "v"}

    @pytest.mark.asyncio
    async def test_buffered_write_model_dump_raising_exotic_field_drops_with_counter(
        self,
        user_config_raw: UserConfig,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """If ``model_dump`` itself explodes after ingest validation passes,
        the narrow fallback catch surfaces the failure via a visible
        ``dropped_record_count`` bump rather than silently swallowing.
        """

        def boom(self, **kwargs):
            raise RuntimeError("model_dump exploded")

        monkeypatch.setattr(RawRecordInfo, "model_dump", boom)

        record = _make_raw_record(payload_bytes=b'{"a":1}')

        async with raw_record_processor("processor-boom", user_config_raw) as processor:
            # Must not raise — fallback catch surfaces via counter
            await processor.buffered_write(record)
            assert processor.lines_written == 0
            assert processor.dropped_record_count == 1

        # Nothing written
        assert (
            not processor.output_file.exists()
            or not processor.output_file.read_text().strip()
        )


class TestBuildExportRecord:
    """Pin ``_build_export_record`` behaviour for edge shapes."""

    def test_build_export_record_error_record_produces_null_payload_and_bytes(
        self,
        user_config_raw: UserConfig,
    ):
        """An error record that never reached transport carries no
        ``payload_bytes`` on its ``RequestInfo`` — ``_build_export_record``
        must emit ``payload=None, payload_bytes=None`` so the writer falls
        through to the generic mixin path and serialises ``error`` instead.
        """
        processor = RawRecordWriterProcessor(
            service_id="processor-err",
            user_config=user_config_raw,
        )

        error = ErrorDetails(code=500, message="boom")
        record = _make_parsed_record(
            payload_bytes=None,
            conversation_id="conv-err",
            status=500,
            error=error,
        )
        metadata = create_metric_metadata(conversation_id="conv-err")

        export = processor._build_export_record(record, metadata)
        assert export.payload is None
        assert export.payload_bytes is None
        assert export.error is not None
        assert export.error.code == 500
        assert export.status == 500


class TestAggregatorUnlinkSemantics:
    """Pin ``RawRecordAggregator.export`` input-file lifecycle."""

    @pytest.mark.asyncio
    async def test_aggregator_unlinks_inputs_after_concat_always(
        self,
        user_config_raw: UserConfig,
        sample_parsed_record_with_raw_responses: ParsedResponseRecord,
    ):
        """After a successful aggregation, every ``raw_records_*.jsonl``
        input file is unlinked from the staging directory and the
        staging directory itself is removed.
        """
        raw_dir = user_config_raw.output.artifact_directory / "raw_records"

        # Build three processor files with one record each
        async with raw_record_processor("processor-A", user_config_raw) as proc_a:
            await proc_a.process_record(
                sample_parsed_record_with_raw_responses,
                create_metric_metadata(conversation_id="c-a"),
            )
        async with raw_record_processor("processor-B", user_config_raw) as proc_b:
            await proc_b.process_record(
                sample_parsed_record_with_raw_responses,
                create_metric_metadata(conversation_id="c-b"),
            )
        async with raw_record_processor("processor-C", user_config_raw) as proc_c:
            await proc_c.process_record(
                sample_parsed_record_with_raw_responses,
                create_metric_metadata(conversation_id="c-c"),
            )

        inputs_before = sorted(raw_dir.glob("raw_records_*.jsonl"))
        assert len(inputs_before) == 3

        exporter_config = create_exporter_config(user_config_raw)
        aggregator = RawRecordAggregator(exporter_config=exporter_config)
        await aggregator.export()

        # All input files removed, staging dir removed
        for f in inputs_before:
            assert not f.exists(), f"aggregator must unlink {f}"
        assert not raw_dir.exists()

        # Output file has all three records concatenated
        assert aggregator.output_file.exists()
        lines = aggregator.output_file.read_text().splitlines()
        assert len(lines) == 3


class TestWave2FixCounter:
    """Wave-2 visibility fix for silent drops."""

    @pytest.mark.asyncio
    async def test_buffered_write_invalid_json_payload_bytes_raises_or_increments_counter_post_fix(
        self,
        user_config_raw: UserConfig,
    ):
        """Post-Wave-2: invalid/unserialisable ``payload_bytes`` must either
        propagate OR increment a dedicated ``dropped_record_count``-style
        attribute so operators can see drops.
        """
        # Use the same shape as test_non_json_non_bytes which hits the
        # TypeError path (orjson.Fragment rejects a non-bytes/str int).
        record = RawRecordInfo.model_construct(
            metadata=create_metric_metadata(),
            start_perf_ns=1_000_000_000,
            payload=None,
            payload_bytes=123,  # type: ignore[arg-type]
            request_headers={},
            response_headers=None,
            status=200,
            responses=[TextResponse(text="ok", perf_ns=2_000_000_000)],
            error=None,
        )

        async with raw_record_processor(
            "processor-wave2", user_config_raw
        ) as processor:
            raised = False
            try:
                await processor.buffered_write(record)
            except Exception:
                raised = True

            counter = getattr(processor, "dropped_record_count", None)
            if counter is None:
                counter = getattr(processor, "drop_count", None)
            if counter is None:
                counter = getattr(processor, "failed_write_count", None)
            # Post-fix: EITHER the exception propagates OR a counter was bumped.
            assert raised or (counter is not None and counter >= 1), (
                "post-Wave-2 fix must surface serialisation failures via "
                "exception or a visible counter"
            )
