# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``_render_realtime_block``.

Latency MetricResult percentile inputs are passed in milliseconds because the
upstream accumulator runs ``to_display_unit`` before metrics reach the
renderer (TTFT, ITL, request_latency: unit=ns, display_unit=ms). Numeric
values in these tests reflect what the live pipeline hands the renderer.
"""

import time

from aiperf.common.enums import CreditPhase
from aiperf.common.models.credit_models import PhaseRecordsStats
from aiperf.common.models.record_models import MetricResult


def _mr(
    tag: str,
    *,
    avg: float | None = None,
    p50: float | None = None,
    p75: float | None = None,
    p90: float | None = None,
    p95: float | None = None,
    p99: float | None = None,
    unit: str = "ms",
) -> MetricResult:
    return MetricResult(
        tag=tag,
        header=tag.replace("_", " ").title(),
        unit=unit,
        avg=avg,
        p50=p50,
        p75=p75,
        p90=p90,
        p95=p95,
        p99=p99,
    )


def _phase_stats(
    *,
    completed: int = 1903,
    sent: int = 2031,  # noqa: ARG001 — kept for back-compat call shape
    errors: int = 0,
    elapsed_s: float = 45.2,
) -> PhaseRecordsStats:
    now_ns = time.time_ns()
    return PhaseRecordsStats(
        phase=CreditPhase.PROFILING,
        start_ns=now_ns - int(elapsed_s * 1_000_000_000),
        success_records=max(0, completed - errors),
        error_records=errors,
    )


def _baseline_metrics() -> list[MetricResult]:
    return [
        _mr("request_throughput", avg=39.8, unit="requests/sec"),
        _mr("output_token_throughput", avg=1820, unit="tokens/sec"),
        _mr("time_to_first_token", p50=80, p95=180, p99=240),
        _mr("inter_token_latency", p50=12, p95=22, p99=35),
        _mr("request_latency", p50=320, p95=680, p99=910),
    ]


def test_render_full_block_first_tick() -> None:
    from aiperf.records.records_manager import _render_realtime_block

    block = _render_realtime_block(
        _baseline_metrics(), _phase_stats(), prev_snapshot=None
    )
    lines = block.splitlines()
    assert lines[0] == "[realtime 00:45 profiling]"
    assert lines[1] == (
        "  rps=39.8 (avg 39.8)  tput_in=-/s  tput_out=1,820/s  "
        "done=1,903 ok=1,903 err=0"
    )
    # One labeled percentile row per metric. Values are right-aligned (so the
    # spacing between ``pNN=`` and the number varies); strip spaces to assert
    # on the label/value pairs without depending on column padding.
    packed = block.replace(" ", "")
    assert "ttftp50=80ms" in packed
    assert "p95=180ms" in packed
    assert "p99=240ms" in packed
    assert "itlp50=12ms" in packed
    assert "e2ep50=320ms" in packed


def test_render_uses_prev_snapshot_for_delta_rps() -> None:
    from aiperf.records.records_manager import _render_realtime_block

    block = _render_realtime_block(
        _baseline_metrics(),
        _phase_stats(completed=1080, sent=1208, elapsed_s=35.0),
        prev_snapshot=(900, 30.0),
    )
    lines = block.splitlines()
    assert lines[0] == "[realtime 00:35 profiling]"
    assert lines[1].startswith("  rps=36.0 (avg 39.8)  tput_in=-/s  tput_out=1,820/s")


def test_render_missing_itl_renders_dashes() -> None:
    from aiperf.records.records_manager import _render_realtime_block

    metrics = [
        _mr("request_throughput", avg=39.8, unit="requests/sec"),
        _mr("output_token_throughput", avg=1820, unit="tokens/sec"),
        _mr("time_to_first_token", p50=80, p95=180, p99=240),
        _mr("request_latency", p50=320, p95=680, p99=910),
    ]
    block = _render_realtime_block(metrics, _phase_stats(), prev_snapshot=None)
    # Every percentile column on the itl row is a dash when the metric is absent.
    assert "itlp50=-p75=-p95=-p99=-" in block.replace(" ", "")


def test_render_sub_millisecond_value_renders_lt1ms() -> None:
    from aiperf.records.records_manager import _render_realtime_block

    metrics = [
        _mr("request_throughput", avg=39.8, unit="requests/sec"),
        _mr("output_token_throughput", avg=1820, unit="tokens/sec"),
        _mr("time_to_first_token", p50=0.5, p95=180, p99=240),
        _mr("inter_token_latency", p50=12, p95=22, p99=35),
        _mr("request_latency", p50=320, p95=680, p99=910),
    ]
    block = _render_realtime_block(metrics, _phase_stats(), prev_snapshot=None)
    assert "ttftp50=<1ms" in block.replace(" ", "")


def test_render_elapsed_under_one_hour_uses_mmss() -> None:
    from aiperf.records.records_manager import _render_realtime_block

    block = _render_realtime_block(
        _baseline_metrics(), _phase_stats(elapsed_s=125.0), prev_snapshot=None
    )
    assert block.startswith("[realtime 02:05 profiling]")


def test_render_elapsed_over_one_hour_uses_hmmss() -> None:
    from aiperf.records.records_manager import _render_realtime_block

    block = _render_realtime_block(
        _baseline_metrics(), _phase_stats(elapsed_s=3725.0), prev_snapshot=None
    )
    assert block.startswith("[realtime 1:02:05 profiling]")


def test_render_zero_completed_returns_empty_string() -> None:
    from aiperf.records.records_manager import _render_realtime_block

    metrics = [
        _mr("request_throughput", avg=0.0, unit="requests/sec"),
        _mr("output_token_throughput", avg=0, unit="tokens/sec"),
    ]
    block = _render_realtime_block(
        metrics,
        _phase_stats(completed=0, sent=0, elapsed_s=2.0),
        prev_snapshot=None,
    )
    assert block == ""


def test_render_seq_rows_show_isl_osl_percentiles() -> None:
    from aiperf.records.records_manager import _render_realtime_block

    metrics = _baseline_metrics() + [
        _mr(
            "input_sequence_length",
            avg=178018,
            p50=123952,
            p75=245124,
            p90=391085,
            p99=720485,
            unit="tokens",
        ),
        _mr(
            "output_sequence_length",
            avg=711,
            p50=261,
            p75=664,
            p90=1614,
            p99=7013,
            unit="tokens",
        ),
    ]
    block = _render_realtime_block(metrics, _phase_stats(), prev_snapshot=None)
    packed = block.replace(" ", "")
    # Comma-separated, four percentiles each, on their own labeled rows. ISL/OSL
    # report p90 (not p95) in the third column; the (tokens) suffix marks units.
    assert "islp50=123,952" in packed
    assert "p75=245,124" in packed
    assert "p90=391,085" in packed
    assert "p99=720,485" in packed
    assert "oslp50=261" in packed
    assert "p90=1,614" in packed
    assert "p99=7,013" in packed
    assert "(tokens)" in packed
    # The old avg-only row should not appear.
    assert "isl_avg" not in block
    assert "osl_avg" not in block


def test_render_seq_rows_omitted_when_metrics_absent() -> None:
    from aiperf.records.records_manager import _render_realtime_block

    # _baseline_metrics() doesn't include ISL/OSL; their rows (and the (tokens)
    # suffix) should be skipped rather than rendered as dashes.
    block = _render_realtime_block(
        _baseline_metrics(), _phase_stats(), prev_snapshot=None
    )
    assert "isl" not in block
    assert "osl" not in block
    assert "(tokens)" not in block


def test_render_server_snapshot_line_includes_unique_input_tokens() -> None:
    from aiperf.records.records_manager import _render_realtime_block

    block = _render_realtime_block(
        _baseline_metrics(),
        _phase_stats(),
        prev_snapshot=None,
        server_snapshot={
            "prefix_cache_hit_rate": 68.3,
            "unique_input_tokens_srv": 123456.0,
            "external_prefix_cache_hit_rate": 11.2,
            "kv_cache_usage_pct": 94.5,
            "cpu_kv_cache_usage_pct": 37.0,
            "num_running": 24,
            "num_waiting": 0,
            "input_token_throughput_srv": 98765.0,
            "output_token_throughput_srv": 4321.0,
        },
    )

    assert "srv  " in block
    assert "prefix_cache_hit=68.3%" in block
    assert "unique_in_srv=123,456" in block
    assert "ext_cache_hit=11.2%" in block
    assert "kv_usage=94.5%" in block
    assert "cpu_kv_usage=37.0%" in block
    assert "queue=24r/0w" in block
    assert "tput_in_srv=98,765/s" in block
    assert "tput_out_srv=4,321/s" in block
