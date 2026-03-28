#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local microbenchmarks for the record-processing drain path.

Usage:
    uv run python dev/benchmarks/record_processing_benchmark.py
    uv run python dev/benchmarks/record_processing_benchmark.py --scenario parser
    uv run python dev/benchmarks/record_processing_benchmark.py --scenario rp
    uv run python dev/benchmarks/record_processing_benchmark.py --scenario rm-ingest
    uv run python dev/benchmarks/record_processing_benchmark.py --scenario export
    uv run python dev/benchmarks/record_processing_benchmark.py --json
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import gc
import math
import multiprocessing
import socket
import statistics
import sys
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import msgspec
import orjson
import zmq
import zmq.asyncio

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aiperf.common.enums import ExportLevel
from aiperf.common.inference_wire import (
    build_inference_results_wire_message,
    decode_inference_results_wire_message,
    encode_inference_results_wire_message,
    wire_message_to_request_record,
)
from aiperf.common.metric_records_wire import (
    MetricRecordMetadata,
    MetricRecordsData,
    MetricRecordsWireMessage,
    build_metric_records_data,
    build_metric_records_wire_message,
)
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
from aiperf.common.models.dataset_models import Conversation
from aiperf.common.models.record_models import MetricValue, RawRecordInfo
from aiperf.config import BenchmarkConfig
from aiperf.credit.sticky_router import StickyCreditRouter
from aiperf.credit.structs import Credit, CreditContext
from aiperf.dataset.memory_map_utils import (
    MemoryMapDatasetBackingStore,
    MemoryMapDatasetClientStore,
)
from aiperf.endpoints.openai_chat import ChatEndpoint
from aiperf.post_processors.metric_results_processor import MetricResultsProcessor
from aiperf.records.inference_result_parser import InferenceResultParser
from aiperf.records.record_processor_service import RecordProcessor
from aiperf.records.records_manager import RecordsManager
from aiperf.records.records_tracker import RecordsTracker
from aiperf.workers.session_manager import UserSessionManager

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


class BenchmarkMetricProcessor:
    """Lightweight RM downstream processor for synthetic ingestion benchmarks."""

    def __init__(self) -> None:
        self._totals: dict[str, float] = {}

    async def process_result(self, record_data: MetricRecordsData) -> None:
        for tag, value in record_data.metrics.items():
            if isinstance(value, list):
                numeric_value = float(sum(value))
            else:
                numeric_value = float(value)
            self._totals[tag] = self._totals.get(tag, 0.0) + numeric_value


class BenchmarkExportProcessor:
    """Lightweight export-like processor that measures serialization cost only."""

    async def process_result(self, record_data: MetricRecordsData) -> None:
        orjson.dumps(record_data.model_dump(mode="json", exclude_none=True))


class BenchmarkRecordsManager:
    """Minimal object that can run the real RecordsManager hot-path methods."""

    def __init__(self, metric_processors: list[Any]) -> None:
        self._records_tracker = RecordsTracker()
        self._metric_results_processors = metric_processors
        self._error_tracker = SimpleNamespace(
            increment_error_count_for_phase=_noop,
        )
        self._handle_all_records_received = _noop_async
        self.is_trace_enabled = False
        self.trace = _noop

    async def _send_results_to_results_processors(
        self,
        record_data: MetricRecordsData,
    ) -> None:
        await RecordsManager._send_results_to_results_processors(self, record_data)

    async def _on_metric_records(self, message: MetricRecordsWireMessage) -> None:
        await RecordsManager._on_metric_records(self, message)


class FakeEndpointMetadata:
    produces_tokens = True
    tokenizes_input = True
    supports_audio = False
    supports_images = False
    supports_videos = False
    produces_videos = False


class FakeStreamingRouterClient:
    def __init__(self) -> None:
        self.sent: list[tuple[str, object]] = []
        self.receiver = None

    async def send_to(self, worker_id: str, message: object) -> None:
        self.sent.append((worker_id, message))

    def register_receiver(self, receiver: Any) -> None:
        self.receiver = receiver


def _noop(*args: Any, **kwargs: Any) -> None:
    return None


async def _noop_async(*args: Any, **kwargs: Any) -> None:
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


def _make_rm_metric_results(
    *,
    processor_count: int,
    metrics_per_processor: int,
    request_index: int,
) -> list[dict[str, int | float]]:
    results: list[dict[str, int | float]] = []
    for processor_index in range(processor_count):
        result = {
            f"bench.metric_{processor_index}_{metric_index}": float(
                request_index + processor_index + metric_index / 10
            )
            for metric_index in range(metrics_per_processor)
        }
        if processor_index == 0:
            result["request_latency"] = 10.0 + request_index / 1000
            result["output_tokens"] = 800
        results.append(result)
    return results


def _make_rm_metric_message(
    *,
    request_index: int,
    processor_count: int,
    metrics_per_processor: int,
) -> MetricRecordsWireMessage:
    results = _make_rm_metric_results(
        processor_count=processor_count,
        metrics_per_processor=metrics_per_processor,
        request_index=request_index,
    )
    metrics = {}
    for result in results:
        metrics.update(result)
    return build_metric_records_wire_message(
        service_id="record-processor-bench",
        metadata=_make_metric_metadata(request_index),
        metrics=metrics,
        trace_data=None,
        error=None,
    )


def _make_rm_metric_messages(
    args: argparse.Namespace,
) -> list[MetricRecordsWireMessage]:
    return [
        _make_rm_metric_message(
            request_index=request_index,
            processor_count=args.processors,
            metrics_per_processor=args.metrics_per_processor,
        )
        for request_index in range(args.records)
    ]


def _make_rm_metric_data_batch(
    messages: list[MetricRecordsWireMessage],
) -> list[MetricRecordsData]:
    return [
        build_metric_records_data(
            metadata=message.metadata,
            metrics=message.metrics,
            trace_data=None,
            error=None,
        )
        for message in messages
    ]


def _chunked(items: list[Any], parts: int) -> list[list[Any]]:
    if parts <= 1:
        return [items]
    chunk_size = max(1, math.ceil(len(items) / parts))
    return [
        items[index : index + chunk_size] for index in range(0, len(items), chunk_size)
    ]


async def _run_concurrent_batches(
    batches: list[list[Any]],
    worker: Callable[[list[Any]], Awaitable[None]],
) -> None:
    await asyncio.gather(*(worker(batch) for batch in batches if batch))


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


def _build_sticky_credit_router() -> StickyCreditRouter:
    fake_credit_client = FakeStreamingRouterClient()
    fake_return_client = FakeStreamingRouterClient()

    class FakeComms:
        def __init__(self) -> None:
            self._clients = [fake_credit_client, fake_return_client]

        def create_streaming_router_client(
            self, **kwargs: Any
        ) -> FakeStreamingRouterClient:
            return self._clients.pop(0)

    def communication_init(self: Any, run: Any, **kwargs: Any) -> None:
        self.run = run
        self.comms = FakeComms()
        self.service_id = kwargs.get("service_id", "sticky-router-bench")
        self.is_enabled_for = lambda level: False
        for method in (
            "trace_or_debug",
            "trace",
            "debug",
            "info",
            "warning",
            "error",
            "exception",
        ):
            setattr(self, method, _noop)

    config = _make_config(use_server_token_count=True)
    config.endpoint.streaming = True
    run = _make_run(config)
    run.resolved.comm_config = SimpleNamespace(controller_host=None)

    with patch(
        "aiperf.common.mixins.CommunicationMixin.__init__",
        communication_init,
    ):
        router = StickyCreditRouter(run=run, service_id="sticky-router-bench")
    return router


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
            build_metric_records_wire_message(
                service_id="record-processor-bench",
                metadata=metadata,
                metrics={
                    tag: value
                    for result in raw_results
                    for tag, value in result.items()
                },
                trace_data=None,
                error=None,
            )
        )

    def rp_rm_message_encode(_: int) -> None:
        for message in metric_messages:
            build_metric_records_data(
                metadata=message.metadata,
                metrics=message.metrics,
                trace_data=None,
                error=None,
            )

    metric_record_batch = [
        build_metric_records_data(
            metadata=message.metadata,
            metrics=message.metrics,
            trace_data=None,
            error=None,
        )
        for message in metric_messages
    ]

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
            message = build_metric_records_wire_message(
                service_id="record-processor-bench",
                metadata=metadata,
                metrics={
                    tag: value
                    for result in raw_results
                    for tag, value in result.items()
                },
                trace_data=None,
                error=None,
            )
            _ = build_metric_records_data(
                metadata=message.metadata,
                metrics=message.metrics,
                trace_data=None,
                error=None,
            )
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


async def _benchmark_rm_to_data_merge(
    args: argparse.Namespace,
    messages: list[MetricRecordsWireMessage],
) -> BenchmarkSample:
    batches = _chunked(messages, args.producer_tasks)

    async def worker(batch: list[Any]) -> None:
        for message in batch:
            _ = build_metric_records_data(
                metadata=message.metadata,
                metrics=message.metrics,
                trace_data=None,
                error=None,
            )

    async def operation(_: int) -> None:
        await _run_concurrent_batches(batches, worker)

    return await _time_async_operation(
        name="rm_ingest::to_data_merge",
        items=len(messages),
        repeats=args.repeats,
        warmup_runs=args.warmup_runs,
        details={
            "processors": args.processors,
            "metrics_per_processor": args.metrics_per_processor,
            "producer_tasks": args.producer_tasks,
        },
        operation=operation,
    )


async def _benchmark_rm_tracker_only(
    args: argparse.Namespace,
    record_data_batch: list[MetricRecordsData],
) -> BenchmarkSample:
    batches = _chunked(record_data_batch, args.producer_tasks)

    async def worker(batch: list[Any]) -> None:
        tracker = RecordsTracker()
        for record_data in batch:
            tracker.update_from_record_data(record_data)
            tracker.check_and_set_all_records_received_for_phase(
                record_data.metadata.benchmark_phase
            )

    async def operation(_: int) -> None:
        await _run_concurrent_batches(batches, worker)

    return await _time_async_operation(
        name="rm_ingest::tracker_only",
        items=len(record_data_batch),
        repeats=args.repeats,
        warmup_runs=args.warmup_runs,
        details={"phase": "profiling", "producer_tasks": args.producer_tasks},
        operation=operation,
    )


async def _benchmark_rm_metric_processor_only(
    args: argparse.Namespace,
    record_data_batch: list[MetricRecordsData],
) -> BenchmarkSample:
    batches = _chunked(record_data_batch, args.producer_tasks)

    async def worker(batch: list[Any]) -> None:
        processor = BenchmarkMetricProcessor()
        for record_data in batch:
            await processor.process_result(record_data)

    async def operation(_: int) -> None:
        await _run_concurrent_batches(batches, worker)

    return await _time_async_operation(
        name="rm_ingest::metric_processor_only",
        items=len(record_data_batch),
        repeats=args.repeats,
        warmup_runs=args.warmup_runs,
        details={
            "downstream": "BenchmarkMetricProcessor",
            "producer_tasks": args.producer_tasks,
        },
        operation=operation,
    )


async def _benchmark_rm_on_metric_records_total(
    args: argparse.Namespace,
    messages: list[MetricRecordsWireMessage],
) -> BenchmarkSample:
    batches = _chunked(messages, args.producer_tasks)

    async def worker(batch: list[Any]) -> None:
        benchmark_rm = BenchmarkRecordsManager([BenchmarkMetricProcessor()])
        for message in batch:
            await benchmark_rm._on_metric_records(message)

    async def operation(_: int) -> None:
        await _run_concurrent_batches(batches, worker)

    return await _time_async_operation(
        name="rm_ingest::on_metric_records_total",
        items=len(messages),
        repeats=args.repeats,
        warmup_runs=args.warmup_runs,
        details={
            "processors": args.processors,
            "metrics_per_processor": args.metrics_per_processor,
            "producer_tasks": args.producer_tasks,
            "streaming_shape": True,
            "target_output_tokens": 800,
        },
        operation=operation,
    )


async def _benchmark_rm_full_with_exports(
    args: argparse.Namespace,
    messages: list[MetricRecordsWireMessage],
) -> BenchmarkSample:
    batches = _chunked(messages, args.producer_tasks)

    async def worker(batch: list[Any]) -> None:
        benchmark_rm = BenchmarkRecordsManager(
            [BenchmarkMetricProcessor(), BenchmarkExportProcessor()]
        )
        for message in batch:
            await benchmark_rm._on_metric_records(message)

    async def operation(_: int) -> None:
        await _run_concurrent_batches(batches, worker)

    return await _time_async_operation(
        name="rm_ingest::full_with_exports",
        items=len(messages),
        repeats=args.repeats,
        warmup_runs=args.warmup_runs,
        details={
            "includes_export_serialization": True,
            "producer_tasks": args.producer_tasks,
        },
        operation=operation,
    )


async def benchmark_records_manager_ingestion(
    args: argparse.Namespace,
) -> list[BenchmarkSample]:
    messages = _make_rm_metric_messages(args)
    record_data_batch = _make_rm_metric_data_batch(messages)
    results = [
        await _benchmark_rm_to_data_merge(args, messages),
        await _benchmark_rm_tracker_only(args, record_data_batch),
        await _benchmark_rm_metric_processor_only(args, record_data_batch),
        await _benchmark_rm_on_metric_records_total(args, messages),
    ]
    if args.rm_include_exports:
        results.append(await _benchmark_rm_full_with_exports(args, messages))
    return results


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
        metric_messages: list[MetricRecordsWireMessage] = []
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

            message = build_metric_records_wire_message(
                service_id="record-processor-bench",
                metadata=metadata,
                metrics={
                    tag: value
                    for result in raw_results
                    for tag, value in result.items()
                },
                trace_data=None,
                error=None,
            )
            metric_messages.append(message)

            started = time.perf_counter()
            record_data = build_metric_records_data(
                metadata=message.metadata,
                metrics=message.metrics,
                trace_data=None,
                error=None,
            )
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


async def benchmark_zmq_dispatch(args: argparse.Namespace) -> list[BenchmarkSample]:
    config = _make_config(use_server_token_count=True)
    payloads = [
        encode_inference_results_wire_message(
            build_inference_results_wire_message(
                service_id="worker-bench",
                record=_make_request_record(
                    config,
                    request_index,
                    prompt_words=args.prompt_words,
                    raw_response_count=args.responses,
                ),
                raw_payload={"model": "test-model"},
                include_status=True,
            )
        )
        for request_index in range(args.records)
    ]

    context = zmq.asyncio.Context.instance()
    endpoint = f"inproc://record-processing-zmq-{uuid.uuid4().hex}"
    sender = context.socket(zmq.PUSH)
    receiver = context.socket(zmq.PULL)
    receiver.bind(endpoint)
    sender.connect(endpoint)

    async def operation(_: int) -> None:
        for payload in payloads:
            await sender.send(payload)
        for _ in payloads:
            await receiver.recv()

    try:
        await operation(0)
        return [
            await _time_async_operation(
                name="raw_zmq_dispatch",
                items=args.records,
                repeats=args.repeats,
                warmup_runs=args.warmup_runs,
                details={
                    "responses_per_record": args.responses,
                    "payload_bytes": len(payloads[0]) if payloads else 0,
                    "transport": "inproc push/pull",
                },
                operation=operation,
            )
        ]
    finally:
        sender.close(linger=0)
        receiver.close(linger=0)


def _pick_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


async def _tcp_echo_handler(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    try:
        data = await reader.readexactly(4)
        writer.write(data)
        await writer.drain()
    finally:
        writer.close()
        await writer.wait_closed()


async def benchmark_tcp_connect(args: argparse.Namespace) -> list[BenchmarkSample]:
    port = _pick_free_tcp_port()
    server = await asyncio.start_server(_tcp_echo_handler, "127.0.0.1", port)

    async def operation(_: int) -> None:
        for _ in range(args.records):
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(b"ping")
            await writer.drain()
            await reader.readexactly(4)
            writer.close()
            await writer.wait_closed()

    try:
        await operation(0)
        return [
            await _time_async_operation(
                name="tcp_connect_roundtrip",
                items=args.records,
                repeats=args.repeats,
                warmup_runs=args.warmup_runs,
                details={"transport": "plain tcp connect/send/recv/close"},
                operation=operation,
            )
        ]
    finally:
        server.close()
        await server.wait_closed()


def _router_server_process(
    endpoint: str,
    items_per_run: int,
    total_runs: int,
    ready_queue: multiprocessing.Queue,
    done_queue: multiprocessing.Queue,
) -> None:
    context = zmq.Context()
    socket_obj = context.socket(zmq.ROUTER)
    socket_obj.bind(endpoint)
    ready_queue.put("ready")
    try:
        for _ in range(total_runs):
            for _ in range(items_per_run):
                socket_obj.recv_multipart()
            done_queue.put("done")
    finally:
        socket_obj.close(linger=0)
        context.term()


async def benchmark_tcp_dealer_router(
    args: argparse.Namespace,
) -> list[BenchmarkSample]:
    encoder = msgspec.msgpack.Encoder()
    payloads = [
        encoder.encode(
            Credit(
                id=request_index,
                phase="profiling",
                conversation_id=f"conversation-{request_index}",
                x_correlation_id=f"corr-{request_index}",
                turn_index=0,
                num_turns=1,
                issued_at_ns=1_700_000_000_000_000_000 + request_index,
            )
        )
        for request_index in range(args.records)
    ]

    port = _pick_free_tcp_port()
    endpoint = f"tcp://127.0.0.1:{port}"
    total_runs = args.warmup_runs + args.repeats + 1
    ready_queue: multiprocessing.Queue = multiprocessing.Queue()
    done_queue: multiprocessing.Queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_router_server_process,
        args=(endpoint, args.records, total_runs, ready_queue, done_queue),
        daemon=True,
    )
    process.start()
    ready_queue.get(timeout=10)

    context = zmq.asyncio.Context.instance()
    dealer = context.socket(zmq.DEALER)
    dealer.setsockopt(zmq.IDENTITY, b"bench-dealer")
    dealer.connect(endpoint)

    async def operation(_: int) -> None:
        for payload in payloads:
            await dealer.send(payload)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, done_queue.get, True, 30)

    try:
        await operation(0)
        return [
            await _time_async_operation(
                name="tcp_dealer_router_dispatch",
                items=args.records,
                repeats=args.repeats,
                warmup_runs=args.warmup_runs,
                details={
                    "payload_bytes": len(payloads[0]) if payloads else 0,
                    "transport": "tcp dealer/router cross-process",
                },
                operation=operation,
            )
        ]
    finally:
        dealer.close(linger=0)
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)


async def benchmark_sticky_credit_router(
    args: argparse.Namespace,
) -> list[BenchmarkSample]:
    router = _build_sticky_credit_router()
    worker_count = max(1, args.processors)
    for worker_index in range(worker_count):
        router._register_worker(f"worker-{worker_index}")

    credits: list[Credit] = []
    session_count = max(1, args.records // 4)
    credit_id = 0
    for session_index in range(session_count):
        x_correlation_id = f"session-{session_index}"
        for turn_index in range(4):
            credits.append(
                Credit(
                    id=credit_id,
                    phase="profiling",
                    conversation_id=f"conversation-{session_index}",
                    x_correlation_id=x_correlation_id,
                    turn_index=turn_index,
                    num_turns=4,
                    issued_at_ns=1_700_000_000_000_000_000 + credit_id,
                    session_num=session_index,
                )
            )
            credit_id += 1
            if len(credits) >= args.records:
                break
        if len(credits) >= args.records:
            break

    async def operation(_: int) -> None:
        router._sticky_sessions.clear()
        router._unavailable_sessions.clear()
        router._workers_by_load.clear()
        router._min_load = 0
        router._workers_cache = list(router._workers.values())
        now_ns = time.perf_counter_ns()
        for worker_load in router._workers.values():
            worker_load.in_flight_credits = 0
            worker_load.active_sessions = 0
            worker_load.active_session_ids.clear()
            worker_load.active_credit_ids.clear()
            worker_load.active_credits.clear()
            worker_load.last_sent_at_ns = now_ns
            router._workers_by_load[0].add(worker_load.worker_id)
        router._credit_router_client.sent.clear()
        for credit in credits:
            await router.send_credit(credit)

    await operation(0)
    return [
        await _time_async_operation(
            name="sticky_credit_router_dispatch",
            items=len(credits),
            repeats=args.repeats,
            warmup_runs=args.warmup_runs,
            details={
                "worker_count": worker_count,
                "session_count": session_count,
                "turns_per_session": 4,
                "payload_type": "real Credit struct",
            },
            operation=operation,
        )
    ]


async def benchmark_worker_start_path(
    args: argparse.Namespace,
) -> list[BenchmarkSample]:
    config = _make_config(use_server_token_count=True)
    endpoint = _build_chat_endpoint(use_server_token_count=True)
    session_manager = UserSessionManager()
    session_manager.set_default_context_mode(None)

    conversation = Conversation(
        session_id="bench-session",
        turns=[_make_turn(args.prompt_words, 0) for _ in range(4)],
        system_message="system prompt for startup benchmark",
        user_context_message="user context for startup benchmark",
    )

    credits = [
        Credit(
            id=request_index,
            phase="profiling",
            conversation_id=conversation.session_id,
            x_correlation_id=f"session-{request_index // 4}",
            turn_index=request_index % 4,
            num_turns=4,
            issued_at_ns=1_700_000_000_000_000_000 + request_index,
            session_num=request_index // 4,
        )
        for request_index in range(args.records)
    ]

    class Clock:
        @staticmethod
        def now_ns() -> int:
            return 1_700_000_000_000_000_000

    worker_like = SimpleNamespace(
        run=SimpleNamespace(cfg=config),
        _create_request_info=None,
    )

    def create_request_info(
        *,
        x_request_id: str,
        session: Any,
        credit_context: CreditContext,
        system_message: str | None = None,
        user_context_message: str | None = None,
    ):
        from aiperf.common.models import RequestInfo

        credit = credit_context.credit
        return RequestInfo(
            config=config,
            credit_num=credit.id,
            session_num=credit.session_num,
            credit_phase=credit.phase,
            cancel_after_ns=credit.cancel_after_ns,
            x_request_id=x_request_id,
            x_correlation_id=session.x_correlation_id,
            conversation_id=session.conversation.session_id,
            turn_index=session.turn_index,
            turns=session.turn_list,
            drop_perf_ns=credit_context.drop_perf_ns,
            credit_issued_ns=credit.issued_at_ns,
            credit_received_ns=credit_context.credit_received_ns,
            system_message=system_message,
            user_context_message=user_context_message,
            is_final_turn=credit.is_final_turn,
            url_index=session.url_index,
        )

    worker_like._create_request_info = create_request_info

    def operation(_: int) -> None:
        session_manager._cache.clear()
        for credit in credits:
            x_request_id = f"request-{credit.id}"
            session = session_manager.get(credit.x_correlation_id)
            if session is None:
                session = session_manager.create_and_store(
                    credit.x_correlation_id,
                    conversation,
                    credit.num_turns,
                    url_index=credit.url_index,
                )
            session.advance_turn(credit.turn_index)
            credit_context = CreditContext(
                credit=credit,
                drop_perf_ns=1_000_000_000 + credit.id,
                credit_received_ns=Clock.now_ns(),
            )
            request_info = worker_like._create_request_info(
                x_request_id=x_request_id,
                session=session,
                credit_context=credit_context,
                system_message=conversation.system_message,
                user_context_message=conversation.user_context_message,
            )
            payload = endpoint.format_payload(request_info)
            orjson.dumps(payload)
            if credit.is_final_turn:
                session_manager.evict(credit.x_correlation_id)

    operation(0)
    return [
        _time_sync_operation(
            name="worker_credit_to_payload_ready",
            items=args.records,
            repeats=args.repeats,
            warmup_runs=args.warmup_runs,
            details={
                "turns_per_session": 4,
                "worker_stage": "session lookup/create + request info + payload format + json serialize",
            },
            operation=operation,
        )
    ]


async def benchmark_mmap_dataset(args: argparse.Namespace) -> list[BenchmarkSample]:
    backing_store = MemoryMapDatasetBackingStore(
        benchmark_id=f"bench-{uuid.uuid4().hex}"
    )
    await backing_store.initialize()
    conversations = {}
    for request_index in range(args.records):
        conversation = Conversation(
            session_id=f"conversation-{request_index}",
            turns=[_make_turn(args.prompt_words, request_index) for _ in range(4)],
            system_message="system prompt for mmap benchmark",
            user_context_message="user context for mmap benchmark",
        )
        conversations[conversation.session_id] = conversation
    await backing_store.add_conversations(conversations)
    await backing_store.finalize()

    client_store = MemoryMapDatasetClientStore(backing_store.get_client_metadata())
    await client_store.initialize()
    conversation_ids = list(conversations.keys())

    async def operation(_: int) -> None:
        for conversation_id in conversation_ids:
            await client_store.get_conversation(conversation_id)

    try:
        await operation(0)
        return [
            await _time_async_operation(
                name="mmap_get_conversation",
                items=len(conversation_ids),
                repeats=args.repeats,
                warmup_runs=args.warmup_runs,
                details={
                    "turns_per_conversation": 4,
                    "prompt_words": args.prompt_words,
                },
                operation=operation,
            )
        ]
    finally:
        await client_store.stop()
        await backing_store.stop()


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
    if args.scenario in {"all", "rm-ingest"}:
        results.extend(await benchmark_records_manager_ingestion(args))
    if args.scenario in {"all", "zmq"}:
        results.extend(await benchmark_zmq_dispatch(args))
    if args.scenario in {"all", "tcp-connect"}:
        results.extend(await benchmark_tcp_connect(args))
    if args.scenario in {"all", "tcp-zmq"}:
        results.extend(await benchmark_tcp_dealer_router(args))
    if args.scenario in {"all", "sticky-credit"}:
        results.extend(await benchmark_sticky_credit_router(args))
    if args.scenario in {"all", "worker-start"}:
        results.extend(await benchmark_worker_start_path(args))
    if args.scenario in {"all", "mmap"}:
        results.extend(await benchmark_mmap_dataset(args))
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
            "rm-ingest",
            "zmq",
            "tcp-connect",
            "tcp-zmq",
            "sticky-credit",
            "worker-start",
            "mmap",
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
        "--producer-tasks",
        type=int,
        default=4,
        help="Concurrent producer tasks for rm-ingest scenarios.",
    )
    parser.add_argument(
        "--metrics-per-processor",
        type=int,
        default=6,
        help="Synthetic metrics emitted by each RP processor.",
    )
    parser.add_argument(
        "--rm-include-exports",
        action="store_true",
        help="Include export-style downstream processors in rm-ingest benchmark.",
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
