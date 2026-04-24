#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Calibrate memory estimator against actual Python object sizes.

Simulates realistic benchmark workloads by instantiating the real AIPerf
objects that exist during a run — RequestRecords with SSE/text responses
sized by ISL/OSL, MetricArrays filled to request count, GPU telemetry
columns scaled by duration — then measures deep memory with pympler and
compares against the estimator's predictions.

Usage:
    uv run python scripts/calibrate_memory_estimates.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from pympler import asizeof

from aiperf.common.enums import SSEFieldType
from aiperf.common.models.dataset_models import Text, Turn
from aiperf.common.models.record_models import (
    MetricResult,
    RequestRecord,
    SSEField,
    SSEMessage,
    TextResponse,
)
from aiperf.kubernetes.memory_estimator import (
    MemoryEstimationParams,
    MemoryEstimator,
    _estimate_records_manager,
    _estimate_worker,
)
from aiperf.metrics.metric_dicts import MetricArray

# =============================================================================
# Helpers
# =============================================================================


def _mib(b: int | float) -> float:
    return b / (1024 * 1024)


def _fmt(b: int | float) -> str:
    mib = _mib(b)
    if mib >= 1.0:
        return f"{mib:,.1f} MiB"
    kib = b / 1024
    if kib >= 1.0:
        return f"{kib:,.1f} KiB"
    return f"{b:,} B"


def _next_pow2(n: int) -> int:
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


# Average chars per token for English text (OpenAI tokenizer ~4 chars/token)
_CHARS_PER_TOKEN = 4


# =============================================================================
# Object factories — build realistic objects sized by ISL/OSL
# =============================================================================


def _make_prompt_text(isl: int) -> str:
    """Create a synthetic prompt string with realistic token count."""
    return "x" * (isl * _CHARS_PER_TOKEN)


def _make_turn(isl: int) -> Turn:
    """Create a Turn object with a prompt sized to ISL tokens."""
    return Turn(
        role="user",
        texts=[Text(content=_make_prompt_text(isl))],
    )


def _make_streaming_response(osl: int) -> SSEMessage:
    """Create an SSEMessage with one SSEField per output token (realistic SSE stream).

    Each SSE chunk is a JSON-formatted OpenAI delta like:
      data: {"choices":[{"delta":{"content":"tok"}}]}
    ~70 bytes per chunk.
    """
    now = time.perf_counter_ns()
    return SSEMessage(
        perf_ns=now,
        packets=[
            SSEField(
                name=SSEFieldType.DATA,
                value=f'{{"choices":[{{"delta":{{"content":"{"x" * _CHARS_PER_TOKEN}"}}}}]}}',
            )
            for _ in range(osl)
        ],
    )


def _make_text_response(osl: int) -> TextResponse:
    """Create a non-streaming TextResponse sized to OSL tokens."""
    # Full OpenAI-style JSON body
    content = "y" * (osl * _CHARS_PER_TOKEN)
    body = f'{{"choices":[{{"message":{{"content":"{content}"}}}}],"usage":{{"prompt_tokens":512,"completion_tokens":{osl}}}}}'
    return TextResponse(perf_ns=time.perf_counter_ns(), text=body)


def _make_request_record(
    isl: int, osl: int, streaming: bool, turns: int = 1
) -> RequestRecord:
    """Create a RequestRecord that mirrors what a worker holds in-flight."""
    now_ns = time.perf_counter_ns()
    turn_list = [_make_turn(isl) for _ in range(turns)]

    if streaming:
        responses = [_make_streaming_response(osl)]
    else:
        responses = [_make_text_response(osl)]

    return RequestRecord(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        timestamp_ns=time.time_ns(),
        start_perf_ns=now_ns,
        end_perf_ns=now_ns + 2_000_000_000,
        recv_start_perf_ns=now_ns + 100_000_000 if streaming else None,
        status=200,
        responses=responses,
        turns=turn_list,
    )


# =============================================================================
# Scenario definition
# =============================================================================


@dataclass
class Scenario:
    """Benchmark workload parameters for calibration."""

    name: str
    isl: int
    osl: int
    streaming: bool
    concurrency: int
    total_requests: int
    turns: int = 1
    total_workers: int = 10
    workers_per_pod: int = 10
    num_models: int = 1
    duration_s: float = 300.0
    num_gpus: int = 0


SCENARIOS = [
    Scenario(
        name="Small / streaming / short",
        isl=128,
        osl=64,
        streaming=True,
        concurrency=32,
        total_requests=5_000,
    ),
    Scenario(
        name="Medium / streaming / typical",
        isl=512,
        osl=128,
        streaming=True,
        concurrency=100,
        total_requests=50_000,
    ),
    Scenario(
        name="Large / streaming / long-output",
        isl=2048,
        osl=512,
        streaming=True,
        concurrency=500,
        total_requests=500_000,
        total_workers=50,
        workers_per_pod=10,
    ),
    Scenario(
        name="Medium / non-streaming / typical",
        isl=512,
        osl=128,
        streaming=False,
        concurrency=100,
        total_requests=50_000,
    ),
    Scenario(
        name="Large / non-streaming / long-output",
        isl=2048,
        osl=512,
        streaming=False,
        concurrency=500,
        total_requests=500_000,
        total_workers=50,
        workers_per_pod=10,
    ),
    Scenario(
        name="Multi-turn / streaming",
        isl=256,
        osl=128,
        streaming=True,
        turns=5,
        concurrency=100,
        total_requests=30_000,
    ),
    Scenario(
        name="Extreme / streaming / very long output",
        isl=4096,
        osl=2048,
        streaming=True,
        concurrency=200,
        total_requests=100_000,
        total_workers=20,
        workers_per_pod=10,
        duration_s=1800,
    ),
]


# =============================================================================
# Measurements
# =============================================================================


def measure_single_request(s: Scenario) -> dict[str, float]:
    """Measure the deep size of one in-flight RequestRecord."""
    rec = _make_request_record(s.isl, s.osl, s.streaming, s.turns)
    return {
        "request_record_bytes": asizeof.asizeof(rec),
    }


def measure_inflight_set(s: Scenario) -> dict[str, float]:
    """Measure memory of all in-flight requests at peak concurrency for one worker."""
    conc_per_worker = max(1, s.concurrency // max(s.total_workers, 1))
    records = [
        _make_request_record(s.isl, s.osl, s.streaming, s.turns)
        for _ in range(conc_per_worker)
    ]
    return {
        "inflight_per_worker_bytes": asizeof.asizeof(records),
        "concurrency_per_worker": conc_per_worker,
        "num_records": len(records),
    }


def measure_records_manager(s: Scenario) -> dict[str, float]:
    """Measure metric array accumulation at total_requests scale."""
    num_metrics = 25
    # Cap at 200K for speed — extrapolate for larger
    n = min(s.total_requests, 200_000)

    arrays: dict[str, MetricArray] = {}
    for m in range(num_metrics):
        arrays[f"metric_{m}"] = MetricArray(initial_capacity=256)

    for i in range(n):
        val = float(i) * 0.001
        for ma in arrays.values():
            ma.append(val)

    measured = asizeof.asizeof(arrays)
    # Extrapolate linearly if capped
    if n < s.total_requests:
        scale = s.total_requests / n
        measured = int(measured * scale)

    return {
        "records_manager_bytes": measured,
        "actual_filled": n,
        "extrapolated_to": s.total_requests,
    }


def measure_gpu_telemetry(s: Scenario) -> dict[str, float]:
    """Measure GPU telemetry columnar storage."""
    if s.num_gpus == 0:
        return {"gpu_telemetry_bytes": 0}

    from aiperf.common.models.telemetry_models import GpuMetricTimeSeries

    n_metrics = 12
    sample_metrics = {f"metric_{i}": float(i) * 10.0 for i in range(n_metrics)}
    n_samples = int(s.duration_s)

    ts = GpuMetricTimeSeries()
    for t in range(n_samples):
        ts.append_snapshot(sample_metrics, timestamp_ns=t * 1_000_000_000)

    per_gpu = asizeof.asizeof(ts)
    return {
        "gpu_telemetry_bytes": per_gpu * s.num_gpus,
        "per_gpu_bytes": per_gpu,
    }


# =============================================================================
# Run one scenario
# =============================================================================


def run_scenario(s: Scenario) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {s.name}")
    print(
        f"  ISL={s.isl}  OSL={s.osl}  streaming={s.streaming}  "
        f"concurrency={s.concurrency}  requests={s.total_requests:,}  turns={s.turns}"
    )
    print(f"{'=' * 72}")

    # --- Single request ---
    req_info = measure_single_request(s)
    print(
        f"\n  Single RequestRecord:            {_fmt(req_info['request_record_bytes'])}"
    )

    # --- In-flight set per worker ---
    inflight = measure_inflight_set(s)
    conc_pw = inflight["concurrency_per_worker"]
    print(
        f"  In-flight per worker ({conc_pw} req):  "
        f"{_fmt(inflight['inflight_per_worker_bytes'])}"
    )

    # --- Records manager ---
    rm = measure_records_manager(s)
    rm_mib = _mib(rm["records_manager_bytes"])
    extra = " (extrapolated)" if rm["actual_filled"] < rm["extrapolated_to"] else ""
    print(
        f"  RecordsManager (25 metrics):     {_fmt(rm['records_manager_bytes'])}{extra}"
    )

    # --- Compare with estimator ---
    print()
    print("  Estimator comparison:")

    # Worker estimate
    est_worker = _estimate_worker(
        concurrency_per_worker=conc_pw,
        avg_osl=s.osl,
        streaming=s.streaming,
        max_turns=s.turns,
        avg_isl=s.isl,
        connections_per_worker=500,
    )
    measured_worker_variable = _mib(inflight["inflight_per_worker_bytes"])
    print(
        f"    Worker variable:  measured={measured_worker_variable:>8.2f} MiB  "
        f"estimated={est_worker.variable_mib:>8.2f} MiB  "
        f"ratio={measured_worker_variable / max(est_worker.variable_mib, 0.001):.2f}x"
    )

    # Records manager estimate
    est_rm = _estimate_records_manager(s.total_requests, 25)
    print(
        f"    RecordsManager:   measured={rm_mib:>8.1f} MiB  "
        f"estimated={est_rm.variable_mib:>8.1f} MiB  "
        f"ratio={rm_mib / max(est_rm.variable_mib, 0.001):.2f}x"
    )

    # Full cluster estimate
    from aiperf.common.environment import Environment

    rp_per_pod = max(1, s.workers_per_pod // Environment.RECORD.PROCESSOR_SCALE_FACTOR)
    params = MemoryEstimationParams(
        total_workers=s.total_workers,
        workers_per_pod=s.workers_per_pod,
        num_worker_pods=max(1, s.total_workers // s.workers_per_pod),
        record_processors_per_pod=rp_per_pod,
        max_concurrency=s.concurrency,
        total_requests=s.total_requests,
        total_benchmark_duration_s=s.duration_s,
        dataset_count=1000,
        avg_isl_tokens=s.isl,
        avg_osl_tokens=s.osl,
        max_turns=s.turns,
        streaming=s.streaming,
        num_endpoints=1,
        connections_per_worker=500,
        num_gpus=s.num_gpus,
        gpu_sample_interval_s=1.0,
        num_gpu_metrics=12,
        num_server_metrics_endpoints=0,
        server_metrics_scrape_interval_s=5.0,
        est_unique_metric_series=200,
        est_histogram_metrics=20,
        est_histogram_buckets=10,
        num_models=s.num_models,
        num_standard_metrics=25,
        export_http_trace=False,
    )
    est = MemoryEstimator(params).estimate()

    print()
    print(f"    Cluster estimate:  {est.total_cluster_mib:,.0f} MiB total")
    print(
        f"    Controller:  steady={est.controller.total_steady_state_mib:,.0f} MiB  "
        f"peak={est.controller.total_peak_mib:,.0f} MiB  "
        f"headroom={est.controller.headroom_pct:.0f}%"
    )
    print(
        f"    Worker pod:  steady={est.worker_pod.total_steady_state_mib:,.0f} MiB  "
        f"peak={est.worker_pod.total_peak_mib:,.0f} MiB  "
        f"headroom={est.worker_pod.headroom_pct:.0f}%"
    )

    if est.warnings:
        print()
        for w in est.warnings:
            print(f"    [!] {w}")


# =============================================================================
# Object size reference table
# =============================================================================


def print_object_reference() -> None:
    """Print a reference table of individual object sizes at various ISL/OSL."""
    print()
    print("=" * 72)
    print("  Object Size Reference (deep size via pympler)")
    print("=" * 72)
    print()
    print(f"  {'Object':<45} {'Size':>10}")
    print("  " + "-" * 55)

    # Base objects
    field = SSEField(name=SSEFieldType.DATA, value="x" * 20)
    print(f"  {'SSEField (20-char token)':<45} {_fmt(asizeof.asizeof(field)):>10}")

    # Vary ISL/OSL for streaming
    for isl, osl in [(128, 64), (512, 128), (2048, 512), (4096, 2048)]:
        rec_s = _make_request_record(isl, osl, streaming=True)
        rec_ns = _make_request_record(isl, osl, streaming=False)
        print(
            f"  {'RequestRecord SSE  ISL=' + str(isl) + ' OSL=' + str(osl):<45} "
            f"{_fmt(asizeof.asizeof(rec_s)):>10}"
        )
        print(
            f"  {'RequestRecord text ISL=' + str(isl) + ' OSL=' + str(osl):<45} "
            f"{_fmt(asizeof.asizeof(rec_ns)):>10}"
        )

    # Multi-turn
    rec_mt = _make_request_record(256, 128, streaming=True, turns=5)
    print(
        f"  {'RequestRecord SSE  5-turn ISL=256 OSL=128':<45} {_fmt(asizeof.asizeof(rec_mt)):>10}"
    )

    # MetricResult
    mr = MetricResult(
        tag="ttft",
        header="TTFT",
        unit="ns",
        min=1e5,
        max=5e5,
        avg=2.5e5,
        p50=2.4e5,
        p90=4e5,
        p95=4.5e5,
        p99=4.9e5,
        p1=1.1e5,
        p5=1.2e5,
        p10=1.3e5,
        p25=2e5,
        p75=3.5e5,
        std=5e4,
        count=10000,
        sum=2.5e9,
    )
    print(f"  {'MetricResult (full stats)':<45} {_fmt(asizeof.asizeof(mr)):>10}")
    print()


# =============================================================================
# Record Processor deep-dive
# =============================================================================


def measure_record_processor() -> None:
    """Measure actual RecordProcessor memory components."""
    import gc
    import os

    import orjson

    print()
    print("=" * 72)
    print("  RecordProcessor Memory Deep-Dive")
    print("=" * 72)
    print()

    # --- 1. Tokenizer sizes ---
    print("  Tokenizer loading (HuggingFace AutoTokenizer):")
    print("  " + "-" * 55)

    # Measure RSS delta for tokenizer loading (pympler can't deep-size C extensions)
    def _get_rss_mib() -> float:
        try:
            with open(f"/proc/{os.getpid()}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024
        except FileNotFoundError:
            return 0.0
        return 0.0

    models_to_test = [
        ("gpt2", "GPT-2 (small BPE)"),
        ("meta-llama/Llama-3.2-1B", "Llama-3.2-1B (tiktoken)"),
    ]

    for model_id, label in models_to_test:
        gc.collect()
        rss_before = _get_rss_mib()
        try:
            from aiperf.common.tokenizer import Tokenizer

            tok = Tokenizer.from_pretrained(model_id)
            gc.collect()
            rss_after = _get_rss_mib()
            delta = rss_after - rss_before
            # Also try pympler (may undercount C objects)
            pympler_size = asizeof.asizeof(tok)
            print(
                f"  {label:<40} RSS delta={delta:>6.1f} MiB  "
                f"pympler={_fmt(pympler_size):>10s}"
            )
            del tok
        except Exception as e:
            print(f"  {label:<40} SKIP ({e.__class__.__name__})")

    print()

    # --- 2. Raw record write buffer (ISL/OSL dependent) ---
    print("  Raw record JSONL write buffer (batch_size=10):")
    print("  " + "-" * 55)

    for isl, osl, streaming in [
        (128, 64, True),
        (512, 128, True),
        (2048, 512, True),
        (4096, 2048, True),
        (512, 128, False),
    ]:
        batch: list[bytes] = []
        for _ in range(10):
            rec = _make_request_record(isl, osl, streaming)
            json_bytes = orjson.dumps(rec.model_dump(exclude_none=True, mode="json"))
            batch.append(json_bytes)
        buffer_size = sum(len(b) for b in batch)
        mode = "SSE" if streaming else "text"
        print(
            f"  {mode:>4} ISL={isl:<5} OSL={osl:<5} "
            f"10 records = {_fmt(buffer_size):>10s}  "
            f"per_record = {_fmt(buffer_size // 10):>10s}"
        )

    print()

    # --- 3. Record export buffer (metrics only, batch_size=100) ---
    print("  Record export JSONL buffer (batch_size=100, metrics only):")
    print("  " + "-" * 55)

    # Simulate a metrics export record (~25 metric values)
    sample_export = {
        "metadata": {
            "session_num": 1,
            "timestamp_ns": 1000000000,
            "end_timestamp_ns": 2000000000,
        },
        "metrics": {f"metric_{i}": {"value": 0.12345, "unit": "ns"} for i in range(25)},
    }
    export_bytes = orjson.dumps(sample_export)
    print(f"  Single export record:  {_fmt(len(export_bytes))}")
    print(f"  100 export records:    {_fmt(len(export_bytes) * 100)}")

    print()

    # --- 4. Estimator comparison ---
    from aiperf.kubernetes.memory_estimator import _estimate_record_processor

    print("  Estimator vs reality:")
    print("  " + "-" * 55)
    for n_models in [1, 2, 3]:
        est = _estimate_record_processor(n_models)
        print(
            f"  {n_models} model(s):  estimated={est.steady_state_mib:>6.0f} MiB  "
            f"(base={est.base_mib:.0f} + tokenizer={n_models * 150} + buffers=2)"
        )
    print(
        "  Note: tokenizer 150 MiB is a planning constant. Actual size depends on model."
    )
    print("        GPT-2 ~1 MiB, Llama-3 ~50 MiB, large SentencePiece ~100-200 MiB.")
    print()


# =============================================================================
# Python baseline
# =============================================================================


def print_process_baseline() -> None:
    import os
    import resource

    print("=" * 72)
    print("  Python Process Baseline")
    print("=" * 72)
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_kb = int(line.split()[1])
                    print(f"  Current RSS:    {rss_kb / 1024:.1f} MiB")
                    break
    except FileNotFoundError:
        pass
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    print(f"  Peak RSS:       {rusage.ru_maxrss / 1024:.1f} MiB")
    print()


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    print()
    print("AIPerf Memory Estimator Calibration")
    print("Measures actual object sizes parameterized by ISL, OSL, streaming mode")
    print()

    print_object_reference()

    for scenario in SCENARIOS:
        run_scenario(scenario)

    measure_record_processor()
    print_process_baseline()


if __name__ == "__main__":
    main()
