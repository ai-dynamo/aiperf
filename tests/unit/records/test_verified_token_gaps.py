# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

import orjson
import pytest
from pytest import param

from aiperf.common.accumulator_protocols import ExportContext
from aiperf.common.constants import NANOS_PER_MILLIS
from aiperf.common.enums import CreditPhase
from aiperf.common.environment import Environment
from aiperf.common.models import (
    ErrorDetails,
    MetricRecordMetadata,
    ProfileResults,
    RequestRecord,
)
from aiperf.common.types import JsonObject
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.exporters.metrics_json_exporter import MetricsJsonExporter
from aiperf.metrics.accumulator import MetricsAccumulator
from aiperf.post_processors.metric_record_processor import MetricRecordProcessor
from aiperf.records.inference_result_parser import InferenceResultParser
from aiperf.records.verified_token_gaps import observe_chat_token_gaps
from aiperf.transports import sse_utils
from aiperf.transports.sse_utils import AsyncSSEStreamReader
from tests.unit.conftest import make_benchmark_run
from tests.unit.records.conftest import create_test_request_info

GAP = "verified_token_gap_latency"
COVERAGE = "verified_token_gap_request_coverage"
T0 = 1_780_000_000_000_000_000


def _event(choice: JsonObject | None, *, tokens: int = 0) -> bytes:
    chunk: JsonObject = {
        "object": "chat.completion.chunk",
        "choices": [] if choice is None else [choice],
    }
    if choice is None:
        chunk["usage"] = {"completion_tokens": tokens, "prompt_tokens": 1}
    return b"data: " + orjson.dumps(chunk) + b"\n\n"


def _text(text: str, logprob_tokens: list[str] | None, **channels: str) -> bytes:
    return _event(
        {
            "index": 0,
            "delta": {"content": text, **channels},
            "logprobs": (
                None
                if logprob_tokens is None
                else {
                    "content": [
                        {"token": token, "bytes": list(token.encode()), "logprob": -0.1}
                        for token in logprob_tokens
                    ]
                }
            ),
        }
    )


def _single(text: str) -> bytes:
    return _text(text, [text])


async def _stream_record(
    monkeypatch: pytest.MonkeyPatch,
    reads: list[tuple[int, bytes]],
    *,
    tokens: int = 4,
    phase: CreditPhase = CreditPhase.PROFILING,
    terminated: bool = True,
) -> RequestRecord:
    frames = [(1, _event({"index": 0, "delta": {"role": "assistant"}})), *reads]
    if terminated:
        frames.extend(
            [
                (
                    150,
                    _event(
                        {"index": 0, "delta": {"content": ""}, "finish_reason": "stop"}
                    ),
                ),
                (160, _event(None, tokens=tokens)),
                (170, b"data: [DONE]\n\n"),
            ]
        )
    clock = iter(ms * NANOS_PER_MILLIS for ms, _ in frames)

    async def chunks() -> AsyncIterator[bytes]:
        for _, data in frames:
            yield data

    monkeypatch.setattr(
        sse_utils, "time", SimpleNamespace(perf_counter_ns=clock.__next__)
    )
    info = create_test_request_info()
    info.credit_phase = phase
    record = RequestRecord(
        request_info=info,
        model_name="test-model",
        start_perf_ns=1,
        end_perf_ns=180 * NANOS_PER_MILLIS,
        timestamp_ns=T0,
        responses=await AsyncSSEStreamReader(chunks()).read_complete_stream(),
    )
    return RequestRecord.model_validate_json(record.model_dump_json())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "blocked_read",
    [
        param(_text("bc", ["b", "c"]), id="bundled"),
        param(_text("bc", None), id="missing-logprobs"),
        param(_text("bc", ["b"]), id="mismatched-token-bytes"),
        param(_single("b") + _single("c"), id="coalesced-read"),
        param(_text("", ["b"]), id="undelivered-token-bytes"),
        param(b"data:\n\n", id="empty-data"),
        param(_event({"index": 0, "delta": {"reasoning_content": "thinking"}}), id="reasoning-only"),
        param(_text("b", ["b"], reasoning="thinking"), id="mixed-reasoning-and-text"),
    ],
)  # fmt: skip
async def test_unverified_read_breaks_adjacency(
    monkeypatch: pytest.MonkeyPatch, blocked_read: bytes
) -> None:
    record = await _stream_record(
        monkeypatch,
        [
            (4, _single("a")),
            (8, _single("x")),
            (20, blocked_read),
            (35, _single("d")),
            (36, _event(None, tokens=5)),
            (37, b": keepalive\n\n"),
            (41, _single("é")),
        ],
    )
    assert observe_chat_token_gaps(record) == [
        4 * NANOS_PER_MILLIS,
        6 * NANOS_PER_MILLIS,
    ]


def test_goodput_rejects_verified_gap_slo_for_streaming_completions() -> None:
    run = make_benchmark_run(
        endpoint_type="completions",
        streaming=True,
        extra={"artifacts": {"verified_token_gaps": True}, "slos": {COVERAGE: 1.0}},
    )

    with pytest.raises(RuntimeError, match=f"Invalid --goodput: metric.*{COVERAGE}"):
        MetricRecordProcessor(run)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "enabled, good_phase",
    [
        param(True, CreditPhase.PROFILING, id="measured"),
        param(True, CreditPhase.WARMUP, id="no-measurement-samples"),
        param(False, CreditPhase.PROFILING, id="disabled"),
    ],
)  # fmt: skip
async def test_verified_token_gaps_export(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    enabled: bool,
    good_phase: CreditPhase,
) -> None:
    run = make_benchmark_run(
        extra={
            "endpoint": {
                "type": "chat",
                "urls": ["http://localhost:8000/v1"],
                "streaming": True,
                "use_server_token_count": True,
            },
            "artifacts": {"verified_token_gaps": enabled, "dir": tmp_path},
        }
    )
    parser = InferenceResultParser(run)
    processor = MetricRecordProcessor(run)
    if enabled and good_phase == CreditPhase.PROFILING:
        pytest.importorskip("crick")
        monkeypatch.setattr(Environment.METRICS, "LIST_BACKEND", "tdigest")
    accumulator = MetricsAccumulator(run)
    irregular = [
        (10, _single("a")),
        (13, _single("b")),
        (53, _single("c")),
        (61, _single("d")),
        (131, _single("e")),
    ]
    good = await _stream_record(monkeypatch, irregular, tokens=5, phase=good_phase)
    long_gap = await _stream_record(
        monkeypatch,
        [(10, _single("a")), (110, _single("b"))],
        tokens=2,
        phase=good_phase,
    )
    single = await _stream_record(monkeypatch, [(10, _single("a"))], tokens=1)
    unverified = await _stream_record(
        monkeypatch, [(10, _text("a", None)), (13, _text("bc", ["b", "c"]))], tokens=3
    )
    incomplete = await _stream_record(monkeypatch, irregular, terminated=False)
    warmup = await _stream_record(monkeypatch, irregular, phase=CreditPhase.WARMUP)
    failed = await _stream_record(monkeypatch, irregular)
    failed.error = ErrorDetails(code=502, message="stream failed")
    cancelled = await _stream_record(monkeypatch, irregular)
    cancelled.cancellation_perf_ns = cancelled.end_perf_ns

    for index, record in enumerate(
        [good, long_gap, single, unverified, incomplete, warmup, failed, cancelled]
    ):
        assert record.request_info is not None
        phase = record.request_info.credit_phase
        parsed = await parser.parse_request_record(record)
        metrics = await processor.process_record(
            parsed,
            MetricRecordMetadata(
                session_num=index,
                request_start_ns=T0 + index * NANOS_PER_MILLIS,
                request_end_ns=T0 + (180 + index) * NANOS_PER_MILLIS,
                worker_id="worker",
                record_processor_id="processor",
                benchmark_phase=phase,
                phase_index=int(phase == CreditPhase.PROFILING),
            ),
        )
        if record is good:
            assert metrics.metrics["inter_token_latency"] == 30.25 * NANOS_PER_MILLIS
            assert metrics.metrics["inter_chunk_latency"] == [
                ms * NANOS_PER_MILLIS for ms in (3, 40, 8, 70)
            ]
        if not enabled or phase == CreditPhase.WARMUP or record in (failed, cancelled):
            assert GAP not in metrics.metrics
            assert COVERAGE not in metrics.metrics
        if enabled and record is incomplete:
            assert GAP not in metrics.metrics
            assert metrics.metrics[COVERAGE] == 0
        await accumulator.process_record(metrics)

    summary = await accumulator.export_results(
        ExportContext(phase=CreditPhase.PROFILING, phase_index=1)
    )
    results = ProfileResults(
        records=list(summary.results.values()), completed=6, start_ns=T0, end_ns=T0
    )
    exporter = MetricsJsonExporter(
        ExporterConfig(cfg=run.cfg, results=results, telemetry_results=None)
    )
    await exporter.export()
    raw = await asyncio.to_thread(run.cfg.artifacts.profile_export_json_file.read_bytes)
    report = orjson.loads(raw)
    if not enabled:
        assert GAP not in report
        assert COVERAGE not in report
        return
    assert report["metadata"]["verified_token_gaps"]["clock"] == "time.perf_counter_ns"
    if good_phase == CreditPhase.WARMUP:
        assert report[GAP] == {"unit": "ms", "count": 0}
        assert report[COVERAGE]["count"] == 3
        assert report[COVERAGE]["sum"] == 0
        return
    assert report[GAP]["unit"] == "ms"
    assert report[GAP]["count"] == 5
    assert report[GAP]["p50"] == 40
    assert report[GAP]["p99"] == pytest.approx(98.8)
    assert report[GAP]["max"] == 100
    assert report[COVERAGE]["count"] == 5
    assert report[COVERAGE]["sum"] == 2
    assert report[COVERAGE]["avg"] == 0.4
