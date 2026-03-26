#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local microbenchmarks for the record-processing drain path.

Usage:
    uv run python dev/benchmarks/record_processing_benchmark.py
    uv run python dev/benchmarks/record_processing_benchmark.py --scenario parser
    uv run python dev/benchmarks/record_processing_benchmark.py --scenario rp
    uv run python dev/benchmarks/record_processing_benchmark.py --scenario export
    uv run python dev/benchmarks/record_processing_benchmark.py --json
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import gc
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import orjson

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aiperf.common.enums import ExportLevel
from aiperf.common.inference_wire import (
    build_inference_results_wire_message,
    decode_inference_results_wire_message,
    encode_inference_results_wire_message,
    wire_message_to_request_record,
)
from aiperf.common.messages.inference_messages import MetricRecordsMessage
from aiperf.common.models import (
    MetricRecordInfo,
    ParsedResponse,
    ParsedResponseRecord,
    ReasoningResponseData,
    RequestInfo,
    RequestRecord,
    Text,
    TextResponse,
    TextResponseData,
    TokenCounts,
    Turn,
    Usage,
)
from aiperf.common.models.record_models import (
    MetricRecordMetadata,
    MetricValue,
    RawRecordInfo,
)
from aiperf.config import BenchmarkConfig
from aiperf.endpoints.openai_chat import ChatEndpoint
from aiperf.post_processors.metric_results_processor import MetricResultsProcessor
from aiperf.records.inference_result_parser import InferenceResultParser
from aiperf.records.record_processor_service import RecordProcessor

_MINIMAL_CONFIG_KWARGS: dict[str, Any] = {
    "models": ["test-model"],
    "endpoint": {
        "type": "chat",
        "urls": ["http://localhost:8000/v1/test"],
    },
    "datasets": {
        "default": {
            "type": "synthetic",
            "entries": 1,
            "prompts": {"isl": 128, "osl": 64},
        }
    },
    "phases": {"default": {"type": "concurrency", "requests": 10, "concurrency": 1}},
}


@dataclass(slots=True)
class BenchmarkSample:
    name: str
    repeats: int
    warmup_runs: int
    items: int
    mean_seconds: float
    best_seconds: float
    stdev_seconds: float
    items_per_second: float
    microseconds_per_item: float
    details: dict[str, Any]


class WordCountTokenizer:
    """Tokenizer stub that makes tokenization cost visible without external downloads."""

    def __init__(self) -> None:
        self.encode_calls = 0

    def encode(self, text: str) -> list[int]:
        self.encode_calls += 1
        return list(range(len(text.split())))


class FakeEndpoint:
    """Endpoint stub that returns prebuilt parsed responses."""

    def __init__(self, run: Any) -> None:
        self.run = run
        self.responses: list[ParsedResponse] = []

    def extract_response_data(
        self, request_record: RequestRecord
    ) -> list[ParsedResponse]:
        return self.responses


class SyntheticMetricProcessor:
    """Small async processor used to measure RP fanout/aggregation overhead."""

    def __init__(self, processor_index: int, metrics_per_processor: int) -> None:
        self._result = {
            f"bench.metric_{processor_index}_{metric_index}": processor_index
            + metric_index / 10
            for metric_index in range(metrics_per_processor)
        }

    async def process_record(
        self, record: ParsedResponseRecord, metadata: MetricRecordMetadata
    ) -> dict[str, float]:
        return self._result


class FakeEndpointMetadata:
    produces_tokens = True
    tokenizes_input = True
    supports_audio = False
    supports_images = False
    supports_videos = False
    produces_videos = False


def _noop(*args: Any, **kwargs: Any) -> None:
    return None


def _make_config(*, use_server_token_count: bool) -> BenchmarkConfig:
    endpoint = dict(_MINIMAL_CONFIG_KWARGS["endpoint"])
    endpoint["use_server_token_count"] = use_server_token_count
    return BenchmarkConfig(**{**_MINIMAL_CONFIG_KWARGS, "endpoint": endpoint})


def _make_run(config: BenchmarkConfig) -> SimpleNamespace:
    return SimpleNamespace(
        cfg=config,
        resolved=SimpleNamespace(tokenizer_names={}),
    )


def _make_turn(prompt_words: int, request_index: int) -> Turn:
    prompt = " ".join(f"prompt_{request_index}_{idx}" for idx in range(prompt_words))
    return Turn(role="user", texts=[Text(contents=[prompt])])


def _make_request_record(
    config: BenchmarkConfig,
    request_index: int,
    *,
    prompt_words: int,
    raw_response_count: int,
) -> RequestRecord:
    turn = _make_turn(prompt_words, request_index)
    request_info = RequestInfo(
        config=config,
        turns=[turn],
        turn_index=0,
        credit_num=request_index,
        session_num=request_index,
        credit_phase="profiling",
        x_request_id=f"request-{request_index}",
        x_correlation_id=f"correlation-{request_index}",
        conversation_id=f"conversation-{request_index}",
        system_message="system prompt for parser benchmark",
        user_context_message="user context for parser benchmark",
    )
    start_perf_ns = 1_000_000_000 + request_index * 1_000_000
    responses = [
        TextResponse(
            perf_ns=start_perf_ns + (response_index + 1) * 1_000,
            text=f"raw_chunk_{request_index}_{response_index}",
            content_type="text/plain",
        )
        for response_index in range(raw_response_count)
    ]
    return RequestRecord(
        request_info=request_info,
        model_name="test-model",
        timestamp_ns=1_700_000_000_000_000_000 + request_index,
        start_perf_ns=start_perf_ns,
        recv_start_perf_ns=start_perf_ns + 500,
        end_perf_ns=start_perf_ns + (raw_response_count + 2) * 1_000,
        status=200,
        responses=responses,
        turns=[turn],
    )


def _make_parsed_responses(
    *,
    parsed_response_count: int,
    output_words_per_response: int,
    include_usage: bool,
) -> list[ParsedResponse]:
    responses: list[ParsedResponse] = []
    for response_index in range(parsed_response_count):
        text = " ".join(
            f"output_{response_index}_{idx}" for idx in range(output_words_per_response)
        )
        usage: Usage | None = None
        if include_usage and response_index == parsed_response_count - 1:
            usage = Usage(
                {
                    "prompt_tokens": 512,
                    "completion_tokens": parsed_response_count
                    * output_words_per_response,
                    "completion_tokens_details": {"reasoning_tokens": 0},
                }
            )
        responses.append(
            ParsedResponse(
                perf_ns=2_000_000_000 + response_index * 1_000,
                data=TextResponseData(text=text),
                usage=usage,
            )
        )
    return responses


def _make_chunk_responses(
    *, parsed_response_count: int, output_words_per_response: int, include_usage: bool
) -> list[TextResponse]:
    responses: list[TextResponse] = []
    for response_index in range(parsed_response_count):
        content = " ".join(
            f"chunk_{response_index}_{idx}" for idx in range(output_words_per_response)
        )
        payload: dict[str, Any] = {
            "object": "chat.completion.chunk",
            "choices": [{"delta": {"content": content}}],
        }
        if include_usage and response_index == parsed_response_count - 1:
            payload["usage"] = {
                "prompt_tokens": 512,
                "completion_tokens": parsed_response_count * output_words_per_response,
                "completion_tokens_details": {"reasoning_tokens": 0},
            }
        responses.append(
            TextResponse(
                perf_ns=2_500_000_000 + response_index * 1_000,
                text=orjson.dumps(payload).decode(),
                content_type="application/json",
            )
        )
    return responses


def _make_chunk_payload_texts(
    *, parsed_response_count: int, output_words_per_response: int, include_usage: bool
) -> list[str]:
    return [
        response.text
        for response in _make_chunk_responses(
            parsed_response_count=parsed_response_count,
            output_words_per_response=output_words_per_response,
            include_usage=include_usage,
        )
    ]


def _normalize_parsed_responses(
    responses: list[ParsedResponse],
) -> list[tuple[str | None, str | None, int | None, int | None, int | None]]:
    normalized: list[
        tuple[str | None, str | None, int | None, int | None, int | None]
    ] = []
    for response in responses:
        data = response.data
        content = None
        reasoning = None
        if isinstance(data, ReasoningResponseData):
            content = data.content
            reasoning = data.reasoning
        elif data is not None:
            content = data.get_text()
        usage = response.usage
        normalized.append(
            (
                content,
                reasoning,
                usage.prompt_tokens if usage else None,
                usage.completion_tokens if usage else None,
                usage.reasoning_tokens if usage else None,
            )
        )
    return normalized


def _normalize_json_obj(
    json_obj: dict[str, Any],
) -> tuple[str | None, str | None, int | None, int | None, int | None]:
    choices = json_obj.get("choices") or []
    first_choice = choices[0] if choices else {}
    delta = first_choice.get("delta") or first_choice.get("message") or {}
    usage = json_obj.get("usage") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    return (
        delta.get("content"),
        delta.get("reasoning_content") or delta.get("reasoning"),
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
        completion_details.get("reasoning_tokens"),
    )


def _normalize_record_via_current_endpoint(
    endpoint: ChatEndpoint, payload_texts: list[str]
) -> list[tuple[str | None, str | None, int | None, int | None, int | None]]:
    record = _make_request_record(
        _make_config(use_server_token_count=True),
        0,
        prompt_words=1,
        raw_response_count=len(payload_texts),
    )
    record.responses = [
        TextResponse(
            perf_ns=2_500_000_000 + idx * 1_000,
            text=payload,
            content_type="application/json",
        )
        for idx, payload in enumerate(payload_texts)
    ]
    return _normalize_parsed_responses(endpoint.extract_response_data(record))


def _make_metric_metadata(request_index: int) -> MetricRecordMetadata:
    base_time = 1_700_000_000_000_000_000 + request_index * 1_000
    return MetricRecordMetadata(
        request_num=request_index,
        session_num=request_index,
        x_request_id=f"request-{request_index}",
        x_correlation_id=f"correlation-{request_index}",
        conversation_id=f"conversation-{request_index}",
        turn_index=0,
        request_start_ns=base_time,
        request_ack_ns=base_time + 100,
        request_end_ns=base_time + 900,
        worker_id="worker-0",
        record_processor_id="record-processor-0",
        benchmark_phase="profiling",
    )


def _make_metric_record_info(metric_count: int) -> MetricRecordInfo:
    metrics = {
        f"bench.metric_{metric_index}": MetricValue(
            value=metric_index + 0.5,
            unit="ms",
        )
        for metric_index in range(metric_count)
    }
    return MetricRecordInfo(
        metadata=_make_metric_metadata(0),
        metrics=metrics,
        error=None,
    )


def _make_raw_record_info(raw_response_count: int) -> RawRecordInfo:
    responses = [
        TextResponse(
            perf_ns=3_000_000_000 + response_index * 1_000,
            text=f"raw payload {response_index} " * 24,
            content_type="text/plain",
        )
        for response_index in range(raw_response_count)
    ]
    return RawRecordInfo(
        metadata=_make_metric_metadata(0),
        start_perf_ns=3_000_000_000,
        payload={
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "user prompt " * 64},
            ],
        },
        request_headers={"x-request-id": "request-0"},
        response_headers=None,
        status=200,
        responses=responses,
        error=None,
    )


def _build_chat_endpoint(*, use_server_token_count: bool) -> ChatEndpoint:
    run = _make_run(_make_config(use_server_token_count=use_server_token_count))
    endpoint = ChatEndpoint(run=run)
    for method in ("trace", "trace_or_debug", "debug", "info", "warning", "error"):
        setattr(endpoint, method, _noop)
    return endpoint


def _build_metric_results_processor(
    *, use_server_token_count: bool
) -> MetricResultsProcessor:
    with patch(
        "aiperf.plugin.plugins.get_endpoint_metadata",
        return_value=FakeEndpointMetadata(),
    ):
        processor = MetricResultsProcessor(
            run=_make_run(_make_config(use_server_token_count=use_server_token_count))
        )
    for method in ("trace", "trace_or_debug", "debug", "info", "warning", "error"):
        setattr(processor, method, _noop)
    return processor


def _build_parser(
    *, use_server_token_count: bool, parsed_responses: list[ParsedResponse]
) -> tuple[InferenceResultParser, WordCountTokenizer]:
    def communication_init(self: Any, run: Any, **kwargs: Any) -> None:
        self.run = run
        self.comms = MagicMock()
        for method in (
            "trace_or_debug",
            "debug",
            "info",
            "warning",
            "error",
            "exception",
        ):
            setattr(self, method, _noop)

    tokenizer = WordCountTokenizer()
    config = _make_config(use_server_token_count=use_server_token_count)
    run = _make_run(config)
    with (
        patch("aiperf.common.mixins.CommunicationMixin.__init__", communication_init),
        patch("aiperf.plugin.plugins.get_class", return_value=FakeEndpoint),
        patch(
            "aiperf.plugin.plugins.get_endpoint_metadata",
            return_value=FakeEndpointMetadata(),
        ),
    ):
        parser = InferenceResultParser(run=run)
    parser.endpoint.responses = parsed_responses

    async def get_tokenizer(model: str) -> WordCountTokenizer:
        return tokenizer

    parser.get_tokenizer = get_tokenizer  # type: ignore[method-assign]
    return parser, tokenizer


async def _time_async_operation(
    name: str,
    items: int,
    repeats: int,
    warmup_runs: int,
    details: dict[str, Any],
    operation: Any,
) -> BenchmarkSample:
    for warmup_index in range(warmup_runs):
        await operation(-(warmup_index + 1))
    samples: list[float] = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for repeat_index in range(repeats):
            started = time.perf_counter()
            await operation(repeat_index)
            samples.append(time.perf_counter() - started)
    finally:
        if gc_was_enabled:
            gc.enable()
    mean_seconds = statistics.mean(samples)
    best_seconds = min(samples)
    stdev_seconds = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return BenchmarkSample(
        name=name,
        repeats=repeats,
        warmup_runs=warmup_runs,
        items=items,
        mean_seconds=mean_seconds,
        best_seconds=best_seconds,
        stdev_seconds=stdev_seconds,
        items_per_second=items / mean_seconds if mean_seconds else 0.0,
        microseconds_per_item=(mean_seconds / items) * 1_000_000 if items else 0.0,
        details=details,
    )


def _time_sync_operation(
    name: str,
    items: int,
    repeats: int,
    warmup_runs: int,
    details: dict[str, Any],
    operation: Any,
) -> BenchmarkSample:
    for warmup_index in range(warmup_runs):
        operation(-(warmup_index + 1))
    samples: list[float] = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for repeat_index in range(repeats):
            started = time.perf_counter()
            operation(repeat_index)
            samples.append(time.perf_counter() - started)
    finally:
        if gc_was_enabled:
            gc.enable()
    mean_seconds = statistics.mean(samples)
    best_seconds = min(samples)
    stdev_seconds = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return BenchmarkSample(
        name=name,
        repeats=repeats,
        warmup_runs=warmup_runs,
        items=items,
        mean_seconds=mean_seconds,
        best_seconds=best_seconds,
        stdev_seconds=stdev_seconds,
        items_per_second=items / mean_seconds if mean_seconds else 0.0,
        microseconds_per_item=(mean_seconds / items) * 1_000_000 if items else 0.0,
        details=details,
    )


async def benchmark_core_stages(args: argparse.Namespace) -> list[BenchmarkSample]:
    config = _make_config(use_server_token_count=False)
    server_config = _make_config(use_server_token_count=True)
    base_records = [
        _make_request_record(
            config,
            request_index,
            prompt_words=args.prompt_words,
            raw_response_count=args.responses,
        )
        for request_index in range(args.records)
    ]
    parsed_responses = _make_parsed_responses(
        parsed_response_count=args.responses,
        output_words_per_response=args.output_words,
        include_usage=True,
    )
    chunk_responses = _make_chunk_responses(
        parsed_response_count=args.responses,
        output_words_per_response=args.output_words,
        include_usage=True,
    )

    results: list[BenchmarkSample] = []

    def worker_wire_encode(_: int) -> None:
        for record in base_records:
            message = build_inference_results_wire_message(
                service_id="worker-bench",
                record=record,
                raw_payload={"model": "test-model"},
                include_status=True,
            )
            encode_inference_results_wire_message(message)

    encoded_messages = [
        encode_inference_results_wire_message(
            build_inference_results_wire_message(
                service_id="worker-bench",
                record=record,
                raw_payload={"model": "test-model"},
                include_status=True,
            )
        )
        for record in base_records
    ]

    def wire_decode_rehydrate(_: int) -> None:
        for data in encoded_messages:
            message = decode_inference_results_wire_message(data)
            wire_message_to_request_record(config=config, message=message)

    endpoint = _build_chat_endpoint(use_server_token_count=True)
    chunk_records = [
        _make_request_record(
            server_config,
            request_index,
            prompt_words=args.prompt_words,
            raw_response_count=args.responses,
        )
        for request_index in range(args.records)
    ]

    def endpoint_chunk_parse(_: int) -> None:
        for record in chunk_records:
            clone = record.model_copy(deep=True)
            clone.responses = copy.deepcopy(chunk_responses)
            endpoint.extract_response_data(clone)

    parser_client, tokenizer = _build_parser(
        use_server_token_count=False,
        parsed_responses=parsed_responses,
    )
    parser_server, _ = _build_parser(
        use_server_token_count=True,
        parsed_responses=parsed_responses,
    )

    async def token_count_client(_: int) -> None:
        for record in [record.model_copy(deep=True) for record in base_records]:
            await parser_client._compute_client_side_token_counts(
                record, parsed_responses
            )

    async def token_count_server(_: int) -> None:
        for _record in base_records:
            await parser_server._compute_server_token_counts(parsed_responses)

    base_parsed_records = [
        ParsedResponseRecord(
            request=record.model_copy(deep=True),
            responses=list(parsed_responses),
            token_counts=TokenCounts(
                input=512, output=args.responses * args.output_words
            ),
        )
        for record in base_records
    ]
    processor = SimpleNamespace(
        service_id="record-processor-bench",
        records_processors=[
            SyntheticMetricProcessor(processor_index, args.metrics_per_processor)
            for processor_index in range(args.processors)
        ],
        run=SimpleNamespace(
            cfg=SimpleNamespace(
                output=SimpleNamespace(export_level=ExportLevel.SUMMARY)
            )
        ),
    )

    async def rp_metric_processing(_: int) -> None:
        for parsed_record in base_parsed_records:
            metadata = RecordProcessor._create_metric_record_metadata(
                processor,
                parsed_record.request.model_copy(deep=True),
                "worker-0",
                last_response_perf_ns=parsed_record.responses[-1].perf_ns,
            )
            await RecordProcessor._process_record(
                processor,
                parsed_record,
                metadata,
            )

    metric_messages = []
    for parsed_record in base_parsed_records[: min(100, len(base_parsed_records))]:
        metadata = RecordProcessor._create_metric_record_metadata(
            processor,
            parsed_record.request.model_copy(deep=True),
            "worker-0",
            last_response_perf_ns=parsed_record.responses[-1].perf_ns,
        )
        raw_results = await RecordProcessor._process_record(
            processor, parsed_record, metadata
        )
        metric_messages.append(
            MetricRecordsMessage(
                service_id="record-processor-bench",
                metadata=metadata,
                results=raw_results,
                trace_data=None,
                error=None,
            )
        )

    def rp_rm_message_encode(_: int) -> None:
        for message in metric_messages:
            message.to_data()

    metric_record_batch = [message.to_data() for message in metric_messages]

    async def rm_ingest_aggregation(_: int) -> None:
        local_processor = _build_metric_results_processor(use_server_token_count=True)
        for record_data in metric_record_batch:
            await local_processor.process_result(record_data)

    async def export_finalize(_: int) -> None:
        local_processor = _build_metric_results_processor(use_server_token_count=True)
        for record_data in metric_record_batch:
            await local_processor.process_result(record_data)
        await local_processor.summarize()

    worker_wire_encode(0)
    wire_decode_rehydrate(0)
    endpoint_chunk_parse(0)
    await token_count_client(0)
    tokenizer.encode_calls = 0
    await token_count_server(0)
    await rp_metric_processing(0)
    rp_rm_message_encode(0)
    await rm_ingest_aggregation(0)
    await export_finalize(0)

    results.append(
        _time_sync_operation(
            name="worker_wire_encode",
            items=args.records,
            repeats=args.repeats,
            warmup_runs=args.warmup_runs,
            details={"responses_per_record": args.responses},
            operation=worker_wire_encode,
        )
    )
    results.append(
        _time_sync_operation(
            name="wire_decode_rehydrate",
            items=args.records,
            repeats=args.repeats,
            warmup_runs=args.warmup_runs,
            details={"responses_per_record": args.responses},
            operation=wire_decode_rehydrate,
        )
    )
    results.append(
        _time_sync_operation(
            name="endpoint_chunk_parse",
            items=args.records,
            repeats=args.repeats,
            warmup_runs=args.warmup_runs,
            details={"responses_per_record": args.responses},
            operation=endpoint_chunk_parse,
        )
    )
    results.append(
        await _time_async_operation(
            name="token_count_client",
            items=args.records,
            repeats=args.repeats,
            warmup_runs=args.warmup_runs,
            details={"prompt_words": args.prompt_words, "tokenizer_encode_calls": None},
            operation=token_count_client,
        )
    )
    results[-1].details["tokenizer_encode_calls"] = tokenizer.encode_calls
    results.append(
        await _time_async_operation(
            name="token_count_server",
            items=args.records,
            repeats=args.repeats,
            warmup_runs=args.warmup_runs,
            details={"responses_per_record": args.responses},
            operation=token_count_server,
        )
    )
    results.append(
        await _time_async_operation(
            name="rp_metric_processing",
            items=args.records,
            repeats=args.repeats,
            warmup_runs=args.warmup_runs,
            details={"metric_processors": args.processors},
            operation=rp_metric_processing,
        )
    )
    results.append(
        _time_sync_operation(
            name="rp_rm_message_encode",
            items=len(metric_messages),
            repeats=args.repeats,
            warmup_runs=args.warmup_runs,
            details={"metric_messages": len(metric_messages)},
            operation=rp_rm_message_encode,
        )
    )
    results.append(
        await _time_async_operation(
            name="rm_ingest_aggregation",
            items=len(metric_record_batch),
            repeats=args.repeats,
            warmup_runs=args.warmup_runs,
            details={"metric_messages": len(metric_record_batch)},
            operation=rm_ingest_aggregation,
        )
    )
    results.append(
        await _time_async_operation(
            name="export_finalize",
            items=len(metric_record_batch),
            repeats=args.repeats,
            warmup_runs=args.warmup_runs,
            details={"metric_messages": len(metric_record_batch)},
            operation=export_finalize,
        )
    )

    return results


async def benchmark_parse_variants(args: argparse.Namespace) -> list[BenchmarkSample]:
    chunk_payload_texts = _make_chunk_payload_texts(
        parsed_response_count=args.responses,
        output_words_per_response=args.output_words,
        include_usage=True,
    )
    endpoint = _build_chat_endpoint(use_server_token_count=True)
    chunk_records = [
        _make_request_record(
            _make_config(use_server_token_count=True),
            request_index,
            prompt_words=args.prompt_words,
            raw_response_count=args.responses,
        )
        for request_index in range(args.records)
    ]

    reference_normalized = _normalize_record_via_current_endpoint(
        endpoint, chunk_payload_texts
    )

    def current_endpoint_parse(_: int) -> None:
        for _record in chunk_records:
            normalized = _normalize_record_via_current_endpoint(
                endpoint, chunk_payload_texts
            )
            if normalized != reference_normalized:
                raise AssertionError("current endpoint parse diverged from reference")

    def raw_json_decode_only(_: int) -> None:
        for _record in chunk_records:
            normalized = [
                _normalize_json_obj(orjson.loads(payload))
                for payload in chunk_payload_texts
            ]
            if normalized != reference_normalized:
                raise AssertionError("raw json decode variant diverged from reference")

    def dict_walk_extract_only(_: int) -> None:
        for _record in chunk_records:
            normalized = []
            for payload in chunk_payload_texts:
                json_obj = orjson.loads(payload)
                choices = json_obj.get("choices") or []
                first_choice = choices[0] if choices else {}
                delta = first_choice.get("delta") or {}
                usage = json_obj.get("usage") or {}
                completion_details = usage.get("completion_tokens_details") or {}
                normalized.append(
                    (
                        delta.get("content"),
                        delta.get("reasoning_content") or delta.get("reasoning"),
                        usage.get("prompt_tokens"),
                        usage.get("completion_tokens"),
                        completion_details.get("reasoning_tokens"),
                    )
                )
            if normalized != reference_normalized:
                raise AssertionError("dict walk variant diverged from reference")

    def tuple_fastpath_extract(_: int) -> None:
        for _record in chunk_records:
            normalized = []
            for payload in chunk_payload_texts:
                json_obj = orjson.loads(payload)
                first_choice = json_obj["choices"][0]
                delta = first_choice["delta"]
                usage = json_obj.get("usage") or {}
                completion_details = usage.get("completion_tokens_details") or {}
                normalized.append(
                    (
                        delta.get("content"),
                        delta.get("reasoning_content") or delta.get("reasoning"),
                        usage.get("prompt_tokens"),
                        usage.get("completion_tokens"),
                        completion_details.get("reasoning_tokens"),
                    )
                )
            if normalized != reference_normalized:
                raise AssertionError("tuple fastpath variant diverged from reference")

    def guarded_fastpath_with_fallback(_: int) -> None:
        for _record in chunk_records:
            normalized = []
            for payload in chunk_payload_texts:
                json_obj = orjson.loads(payload)
                try:
                    object_type = json_obj.get("object")
                    if object_type == "chat.completion.chunk":
                        first_choice = json_obj["choices"][0]
                        delta = first_choice["delta"]
                    elif object_type == "chat.completion":
                        first_choice = json_obj["choices"][0]
                        delta = first_choice["message"]
                    else:
                        raise KeyError("unsupported object type")

                    usage = json_obj.get("usage") or {}
                    completion_details = usage.get("completion_tokens_details") or {}
                    normalized.append(
                        (
                            delta.get("content"),
                            delta.get("reasoning_content") or delta.get("reasoning"),
                            usage.get("prompt_tokens"),
                            usage.get("completion_tokens"),
                            completion_details.get("reasoning_tokens"),
                        )
                    )
                except (IndexError, KeyError, TypeError):
                    normalized.append(_normalize_json_obj(json_obj))
            if normalized != reference_normalized:
                raise AssertionError("guarded fastpath variant diverged from reference")

    current_endpoint_parse(0)
    raw_json_decode_only(0)
    dict_walk_extract_only(0)
    tuple_fastpath_extract(0)
    guarded_fastpath_with_fallback(0)

    return [
        _time_sync_operation(
            name="parse_variant_current_endpoint",
            items=args.records,
            repeats=args.repeats,
            warmup_runs=args.warmup_runs,
            details={"responses_per_record": args.responses},
            operation=current_endpoint_parse,
        ),
        _time_sync_operation(
            name="parse_variant_raw_json_decode_only",
            items=args.records,
            repeats=args.repeats,
            warmup_runs=args.warmup_runs,
            details={"responses_per_record": args.responses},
            operation=raw_json_decode_only,
        ),
        _time_sync_operation(
            name="parse_variant_dict_walk_extract_only",
            items=args.records,
            repeats=args.repeats,
            warmup_runs=args.warmup_runs,
            details={"responses_per_record": args.responses},
            operation=dict_walk_extract_only,
        ),
        _time_sync_operation(
            name="parse_variant_tuple_fastpath_extract",
            items=args.records,
            repeats=args.repeats,
            warmup_runs=args.warmup_runs,
            details={"responses_per_record": args.responses},
            operation=tuple_fastpath_extract,
        ),
        _time_sync_operation(
            name="parse_variant_guarded_fastpath_fallback",
            items=args.records,
            repeats=args.repeats,
            warmup_runs=args.warmup_runs,
            details={"responses_per_record": args.responses},
            operation=guarded_fastpath_with_fallback,
        ),
    ]


async def benchmark_parser_path(args: argparse.Namespace) -> list[BenchmarkSample]:
    base_records = [
        _make_request_record(
            _make_config(use_server_token_count=False),
            request_index,
            prompt_words=args.prompt_words,
            raw_response_count=args.responses,
        )
        for request_index in range(args.records)
    ]

    results: list[BenchmarkSample] = []
    for use_server_token_count in (False, True):
        parsed_responses = _make_parsed_responses(
            parsed_response_count=args.responses,
            output_words_per_response=args.output_words,
            include_usage=use_server_token_count,
        )
        parser, tokenizer = _build_parser(
            use_server_token_count=use_server_token_count,
            parsed_responses=parsed_responses,
        )

        async def operation(
            _: int, parser_instance: InferenceResultParser = parser
        ) -> None:
            for record in [record.model_copy(deep=True) for record in base_records]:
                await parser_instance.parse_request_record(record)

        warmup_records = [
            record.model_copy(deep=True)
            for record in base_records[: min(10, len(base_records))]
        ]
        for record in warmup_records:
            await parser.parse_request_record(record)
        tokenizer.encode_calls = 0

        sample = await _time_async_operation(
            name=(
                "parser_with_server_token_count"
                if use_server_token_count
                else "parser_with_client_tokenization"
            ),
            items=args.records,
            repeats=args.repeats,
            warmup_runs=args.warmup_runs,
            details={
                "prompt_words": args.prompt_words,
                "responses_per_record": args.responses,
                "output_words_per_response": args.output_words,
                "tokenizer_encode_calls": None,
            },
            operation=operation,
        )
        sample.details["tokenizer_encode_calls"] = tokenizer.encode_calls
        results.append(sample)
    return results


async def benchmark_record_processor_path(
    args: argparse.Namespace,
) -> list[BenchmarkSample]:
    config = _make_config(use_server_token_count=True)
    base_records = [
        _make_request_record(
            config,
            request_index,
            prompt_words=args.prompt_words,
            raw_response_count=args.responses,
        )
        for request_index in range(args.records)
    ]
    parsed_responses = _make_parsed_responses(
        parsed_response_count=args.responses,
        output_words_per_response=args.output_words,
        include_usage=True,
    )
    base_parsed_records = [
        ParsedResponseRecord(
            request=record.model_copy(deep=True),
            responses=list(parsed_responses),
            token_counts=TokenCounts(
                input=512, output=args.responses * args.output_words
            ),
        )
        for record in base_records
    ]
    processor = SimpleNamespace(
        service_id="record-processor-bench",
        records_processors=[
            SyntheticMetricProcessor(processor_index, args.metrics_per_processor)
            for processor_index in range(args.processors)
        ],
        run=SimpleNamespace(
            cfg=SimpleNamespace(
                output=SimpleNamespace(export_level=ExportLevel.SUMMARY)
            )
        ),
    )

    async def operation(_: int) -> None:
        parsed_records = [
            ParsedResponseRecord(
                request=base_record.request.model_copy(deep=True),
                responses=list(base_record.responses),
                token_counts=base_record.token_counts,
            )
            for base_record in base_parsed_records
        ]
        for parsed_record in parsed_records:
            last_response_perf_ns = parsed_record.responses[-1].perf_ns
            metadata = RecordProcessor._create_metric_record_metadata(
                processor,
                parsed_record.request,
                "worker-0",
                last_response_perf_ns=last_response_perf_ns,
            )
            raw_results = await RecordProcessor._process_record(
                processor,
                parsed_record,
                metadata,
            )
            message = MetricRecordsMessage(
                service_id="record-processor-bench",
                metadata=metadata,
                results=raw_results,
                trace_data=None,
                error=None,
            )
            _ = message.to_data()
            RecordProcessor._free_record_data(
                processor, parsed_record.request, parsed_record
            )

    await operation(0)

    return [
        await _time_async_operation(
            name="record_processor_batch_throughput",
            items=args.records,
            repeats=args.repeats,
            warmup_runs=args.warmup_runs,
            details={
                "responses_per_record": args.responses,
                "metric_processors": args.processors,
                "metrics_per_processor": args.metrics_per_processor,
                "includes_rm_merge": True,
            },
            operation=operation,
        )
    ]


async def benchmark_full_path(args: argparse.Namespace) -> list[BenchmarkSample]:
    config = _make_config(use_server_token_count=True)
    base_records = [
        _make_request_record(
            config,
            request_index,
            prompt_words=args.prompt_words,
            raw_response_count=args.responses,
        )
        for request_index in range(args.records)
    ]
    parsed_responses = _make_parsed_responses(
        parsed_response_count=args.responses,
        output_words_per_response=args.output_words,
        include_usage=True,
    )
    parser, _ = _build_parser(
        use_server_token_count=True,
        parsed_responses=parsed_responses,
    )
    processor = SimpleNamespace(
        service_id="record-processor-bench",
        records_processors=[
            SyntheticMetricProcessor(processor_index, args.metrics_per_processor)
            for processor_index in range(args.processors)
        ],
        run=SimpleNamespace(
            cfg=SimpleNamespace(
                output=SimpleNamespace(export_level=ExportLevel.SUMMARY)
            )
        ),
    )

    stage_names = [
        "worker_build_wire_message",
        "worker_encode_wire_message",
        "rp_decode_wire_message",
        "rp_rehydrate_request_record",
        "parser_extract_response_data",
        "parser_compute_server_token_counts",
        "parser_build_parsed_record",
        "rp_create_metric_metadata",
        "rp_process_record",
        "rp_rm_message_encode",
        "rm_ingest_aggregation",
        "export_finalize",
    ]

    async def run_once() -> dict[str, float]:
        timings = {name: 0.0 for name in stage_names}
        metric_messages: list[MetricRecordsMessage] = []
        record_data_batch = []

        for record in base_records:
            worker_record = record.model_copy(deep=True)

            started = time.perf_counter()
            wire_message = build_inference_results_wire_message(
                service_id="worker-bench",
                record=worker_record,
                raw_payload={"model": "test-model"},
                include_status=True,
            )
            timings["worker_build_wire_message"] += time.perf_counter() - started

            started = time.perf_counter()
            encoded = encode_inference_results_wire_message(wire_message)
            timings["worker_encode_wire_message"] += time.perf_counter() - started

            started = time.perf_counter()
            decoded = decode_inference_results_wire_message(encoded)
            timings["rp_decode_wire_message"] += time.perf_counter() - started

            started = time.perf_counter()
            _, rehydrated = wire_message_to_request_record(
                config=config, message=decoded
            )
            timings["rp_rehydrate_request_record"] += time.perf_counter() - started

            started = time.perf_counter()
            extracted_responses = parser.endpoint.extract_response_data(rehydrated)
            if parser.run.cfg.output.export_level != ExportLevel.RAW:
                rehydrated.responses = None
            timings["parser_extract_response_data"] += time.perf_counter() - started

            started = time.perf_counter()
            token_counts = await parser._compute_server_token_counts(
                extracted_responses
            )
            timings["parser_compute_server_token_counts"] += (
                time.perf_counter() - started
            )

            started = time.perf_counter()
            parsed_record = ParsedResponseRecord(
                request=rehydrated,
                responses=extracted_responses,
                token_counts=token_counts,
            )
            timings["parser_build_parsed_record"] += time.perf_counter() - started

            started = time.perf_counter()
            metadata = RecordProcessor._create_metric_record_metadata(
                processor,
                parsed_record.request,
                "worker-0",
                last_response_perf_ns=parsed_record.responses[-1].perf_ns
                if parsed_record.responses
                else parsed_record.request.end_perf_ns,
            )
            timings["rp_create_metric_metadata"] += time.perf_counter() - started

            started = time.perf_counter()
            raw_results = await RecordProcessor._process_record(
                processor,
                parsed_record,
                metadata,
            )
            timings["rp_process_record"] += time.perf_counter() - started

            message = MetricRecordsMessage(
                service_id="record-processor-bench",
                metadata=metadata,
                results=raw_results,
                trace_data=None,
                error=None,
            )
            metric_messages.append(message)

            started = time.perf_counter()
            record_data = message.to_data()
            timings["rp_rm_message_encode"] += time.perf_counter() - started
            record_data_batch.append(record_data)

            RecordProcessor._free_record_data(
                processor, parsed_record.request, parsed_record
            )

        started = time.perf_counter()
        local_processor = _build_metric_results_processor(use_server_token_count=True)
        for record_data in record_data_batch:
            await local_processor.process_result(record_data)
        timings["rm_ingest_aggregation"] += time.perf_counter() - started

        started = time.perf_counter()
        await local_processor.summarize()
        timings["export_finalize"] += time.perf_counter() - started

        return timings

    for _warmup in range(args.warmup_runs):
        await run_once()

    per_stage_samples = {name: [] for name in stage_names}
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _repeat in range(args.repeats):
            timings = await run_once()
            for name, value in timings.items():
                per_stage_samples[name].append(value)
    finally:
        if gc_was_enabled:
            gc.enable()

    return [
        BenchmarkSample(
            name=f"full_path::{name}",
            repeats=args.repeats,
            warmup_runs=args.warmup_runs,
            items=args.records,
            mean_seconds=statistics.mean(samples),
            best_seconds=min(samples),
            stdev_seconds=statistics.stdev(samples) if len(samples) > 1 else 0.0,
            items_per_second=args.records / statistics.mean(samples)
            if statistics.mean(samples)
            else 0.0,
            microseconds_per_item=(statistics.mean(samples) / args.records) * 1_000_000
            if args.records
            else 0.0,
            details={"responses_per_record": args.responses},
        )
        for name, samples in per_stage_samples.items()
    ]


def benchmark_export_path(args: argparse.Namespace) -> list[BenchmarkSample]:
    metric_record = _make_metric_record_info(args.export_metrics)
    raw_record = _make_raw_record_info(args.responses)

    def serialize_metric_record(_: int) -> None:
        for _record_index in range(args.records):
            orjson.dumps(metric_record.model_dump(exclude_none=True, mode="json"))

    def serialize_raw_record(_: int) -> None:
        for _record_index in range(args.records):
            orjson.dumps(raw_record.model_dump(exclude_none=True, mode="json"))

    serialize_metric_record(0)
    serialize_raw_record(0)

    return [
        _time_sync_operation(
            name="record_metrics_export_serialization",
            items=args.records,
            repeats=args.repeats,
            warmup_runs=args.warmup_runs,
            details={
                "metrics_per_record": args.export_metrics,
                "serialization": "orjson.dumps(model_dump(...))",
            },
            operation=serialize_metric_record,
        ),
        _time_sync_operation(
            name="raw_record_export_serialization",
            items=args.records,
            repeats=args.repeats,
            warmup_runs=args.warmup_runs,
            details={
                "raw_responses_per_record": args.responses,
                "serialization": "orjson.dumps(model_dump(...))",
            },
            operation=serialize_raw_record,
        ),
    ]


def _print_results(results: list[BenchmarkSample]) -> None:
    print(
        f"{'benchmark':<36} {'items/s':>12} {'us/item':>12} {'mean ms':>12} {'best ms':>12}"
    )
    print("-" * 92)
    for result in results:
        print(
            f"{result.name:<36}"
            f" {result.items_per_second:>12.0f}"
            f" {result.microseconds_per_item:>12.1f}"
            f" {result.mean_seconds * 1000:>12.2f}"
            f" {result.best_seconds * 1000:>12.2f}"
        )
        print(f"  details: {result.details}")


async def _run_async_scenarios(args: argparse.Namespace) -> list[BenchmarkSample]:
    results: list[BenchmarkSample] = []
    if args.scenario in {"all", "core"}:
        results.extend(await benchmark_core_stages(args))
    if args.scenario in {"all", "parse-variants"}:
        results.extend(await benchmark_parse_variants(args))
    if args.scenario in {"all", "full-path"}:
        results.extend(await benchmark_full_path(args))
    if args.scenario in {"all", "parser"}:
        results.extend(await benchmark_parser_path(args))
    if args.scenario in {"all", "rp"}:
        results.extend(await benchmark_record_processor_path(args))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=[
            "all",
            "core",
            "parse-variants",
            "full-path",
            "parser",
            "rp",
            "export",
        ],
        default="all",
        help="Benchmark scenario to run.",
    )
    parser.add_argument(
        "--records", type=int, default=1000, help="Synthetic records per run."
    )
    parser.add_argument(
        "--responses", type=int, default=8, help="Responses/chunks per record."
    )
    parser.add_argument(
        "--prompt-words", type=int, default=512, help="Prompt words per record."
    )
    parser.add_argument(
        "--output-words",
        type=int,
        default=32,
        help="Output words per parsed response chunk.",
    )
    parser.add_argument(
        "--processors", type=int, default=3, help="Synthetic RP processors."
    )
    parser.add_argument(
        "--metrics-per-processor",
        type=int,
        default=6,
        help="Synthetic metrics emitted by each RP processor.",
    )
    parser.add_argument(
        "--export-metrics",
        type=int,
        default=18,
        help="Metric fields per serialized MetricRecordInfo.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Measured timing repeats per scenario.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Warmup runs per scenario before timing.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit JSON instead of a table."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = asyncio.run(_run_async_scenarios(args))
    if args.scenario in {"all", "export"}:
        results.extend(benchmark_export_path(args))

    if args.json:
        print(
            orjson.dumps(
                [asdict(result) for result in results],
                option=orjson.OPT_INDENT_2,
            ).decode()
        )
        return

    _print_results(results)


if __name__ == "__main__":
    main()
