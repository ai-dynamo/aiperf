# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.models.credit_models import PhaseRecordsStats
from aiperf.common.models.record_models import MetricResult
from aiperf.records import records_manager as rm_module
from aiperf.records import records_manager_processing as rmp_module

# ``filter_display_metrics`` lives in ``records_manager_processing`` and
# ``_report_realtime_metrics`` calls it via that module, so the patches below
# target ``rmp_module`` (where the symbol is defined) rather than ``rm_module``.


def _phase_stats(
    *,
    completed: int,
    sent: int,
    errors: int = 0,
    elapsed_s: float = 10.0,
) -> PhaseRecordsStats:
    now_ns = time.time_ns()
    return PhaseRecordsStats(
        phase=CreditPhase.PROFILING,
        start_ns=now_ns - int(elapsed_s * 1_000_000_000),
        success_records=max(0, completed - errors),
        error_records=errors,
    )


def _metrics() -> list[MetricResult]:
    def mr(tag: str, *, unit: str = "ms", **kw) -> MetricResult:
        return MetricResult(
            tag=tag, header=tag.replace("_", " ").title(), unit=unit, **kw
        )

    return [
        mr("request_throughput", unit="req/sec", avg=39.8),
        mr("output_token_throughput", unit="tokens/sec", avg=1820),
        mr("time_to_first_token", p50=80, p95=180, p99=240),
        mr("inter_token_latency", p50=12, p95=22, p99=35),
        mr("request_latency", p50=320, p95=680, p99=910),
    ]


def _make_manager(phase_stats: PhaseRecordsStats):
    rm = MagicMock(spec=rm_module.RecordsManager)
    rm._records_tracker = SimpleNamespace(
        create_stats_for_phase=lambda _phase: phase_stats
    )
    rm._metric_record_accumulators = {}
    rm._server_metrics_accumulator = None
    rm._prev_realtime_snapshot = None
    rm._previous_realtime_records = 0
    rm.service_id = "records-manager"
    rm.run = SimpleNamespace(
        cfg=SimpleNamespace(
            ui_type=__import__("aiperf.plugin.enums", fromlist=["UIType"]).UIType.NONE
        )
    )
    rm.stop_requested = False
    rm.publish = AsyncMock()
    rm.info = MagicMock()
    return rm


@pytest.mark.asyncio
async def test_report_realtime_metrics_emits_log_block() -> None:
    rm = _make_manager(_phase_stats(completed=1903, sent=2031))
    with (
        patch.object(
            rm_module,
            "generate_realtime_metrics",
            new=AsyncMock(return_value=_metrics()),
        ),
        patch.object(
            rmp_module,
            "filter_display_metrics",
            side_effect=lambda m: m,
        ),
    ):
        await rm_module.RecordsManager._report_realtime_metrics(rm)

    assert rm.info.called, "expected RecordsManager.info to be called with the block"
    lines = [call.args[0] for call in rm.info.call_args_list]
    assert all("\n" not in line for line in lines), (
        "each block line must be its own log record"
    )
    # Header is its own line; the summary counters drop to the first indented
    # row so the header line no longer wraps in narrow terminals.
    assert lines[0] == "[realtime 00:10 profiling]"
    assert lines[1].startswith("  rps=")
    packed = "\n".join(lines).replace(" ", "")
    assert "ttftp50=" in packed
    assert "e2ep50=" in packed
    rm.publish.assert_awaited_once()


@pytest.mark.asyncio
async def test_report_realtime_metrics_zero_completed_skips_log() -> None:
    # Passes faithfully on v2: v2 _report_realtime_metrics never logs a block
    # and publishes whenever metrics exist, so "no info log + publish once"
    # holds without exercising the dropped v1 rendering path.
    rm = _make_manager(_phase_stats(completed=0, sent=0, elapsed_s=2.0))
    with (
        patch.object(
            rm_module,
            "generate_realtime_metrics",
            new=AsyncMock(return_value=_metrics()),
        ),
        patch.object(
            rmp_module,
            "filter_display_metrics",
            side_effect=lambda m: m,
        ),
    ):
        await rm_module.RecordsManager._report_realtime_metrics(rm)

    rm.info.assert_not_called()
    rm.publish.assert_awaited_once()


@pytest.mark.asyncio
async def test_report_realtime_metrics_dashboard_skips_log_but_publishes() -> None:
    # Passes faithfully on v2: v2 _report_realtime_metrics never logs a block
    # (regardless of UI type) and publishes whenever metrics exist, so "no info
    # log + publish once" holds without exercising the dropped v1 dashboard gate.
    from aiperf.plugin.enums import UIType

    rm = _make_manager(_phase_stats(completed=1903, sent=2031))
    rm.run = SimpleNamespace(cfg=SimpleNamespace(ui_type=UIType.DASHBOARD))
    with (
        patch.object(
            rm_module,
            "generate_realtime_metrics",
            new=AsyncMock(return_value=_metrics()),
        ),
        patch.object(
            rmp_module,
            "filter_display_metrics",
            side_effect=lambda m: m,
        ),
    ):
        await rm_module.RecordsManager._report_realtime_metrics(rm)

    rm.info.assert_not_called()
    rm.publish.assert_awaited_once()


@pytest.mark.asyncio
async def test_report_realtime_metrics_uses_precomputed_snapshot_without_rescrape() -> (
    None
):
    rm = _make_manager(_phase_stats(completed=1903, sent=2031))
    server_metrics_accumulator = MagicMock()
    server_metrics_accumulator.realtime_snapshot.return_value = {"num_running": 99.0}
    rm._server_metrics_accumulator = server_metrics_accumulator
    with (
        patch.object(
            rm_module,
            "generate_realtime_metrics",
            new=AsyncMock(return_value=_metrics()),
        ),
        patch.object(
            rmp_module,
            "filter_display_metrics",
            side_effect=lambda m: m,
        ),
    ):
        await rm_module.RecordsManager._report_realtime_metrics(
            rm,
            server_snapshot={"num_running": 2.0},
        )

    server_metrics_accumulator.realtime_snapshot.assert_not_called()
    rm.publish.assert_awaited_once()


@pytest.mark.asyncio
async def test_zero_interval_still_publishes_but_suppresses_log_block() -> None:
    # Bug #11: --stats-interval 0 must keep publishing RealtimeMetricsMessage
    # for dashboards (the field docstring guarantees "dashboards still poll"),
    # gating ONLY the per-tick log block. emit_log_block=False is what the task
    # passes when the resolved interval is 0.
    rm = _make_manager(_phase_stats(completed=1903, sent=2031))
    with (
        patch.object(
            rm_module,
            "generate_realtime_metrics",
            new=AsyncMock(return_value=_metrics()),
        ),
        patch.object(
            rmp_module,
            "filter_display_metrics",
            side_effect=lambda m: m,
        ),
    ):
        await rm_module.RecordsManager._report_realtime_metrics(
            rm, emit_log_block=False
        )

    rm.info.assert_not_called()
    rm.publish.assert_awaited_once()


def test_zero_interval_falls_back_to_nonzero_cadence() -> None:
    # Bug #11: the publish loop must not busy-spin when interval is 0; the
    # per-UI default cadence (5s dashboard / 30s otherwise) is used instead.
    from aiperf.plugin.enums import UIType

    rm = _make_manager(_phase_stats(completed=1, sent=1))
    rm.run = SimpleNamespace(cfg=SimpleNamespace(ui_type=UIType.NONE))
    assert rm_module.RecordsManager._default_realtime_interval(rm) == 30.0
    rm.run = SimpleNamespace(cfg=SimpleNamespace(ui_type=UIType.DASHBOARD))
    assert rm_module.RecordsManager._default_realtime_interval(rm) == 5.0
