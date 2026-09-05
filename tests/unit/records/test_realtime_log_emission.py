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
async def test_report_realtime_metrics_publishes_server_snapshot_metrics() -> None:
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
            rm,
            server_snapshot={
                "prefix_cache_hit_rate": 42.0,
                "num_running": 2.0,
            },
        )

    published = rm.publish.await_args.args[0]
    published_tags = {metric.tag for metric in published.metrics}
    assert "request_throughput" in published_tags
    assert "prefix_cache_hit_rate" in published_tags
    assert "num_running" in published_tags


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


def _publish_gate_manager(
    *, ui_type, api_port, service_run_type="multiprocessing"
) -> MagicMock:
    rm = MagicMock(spec=rm_module.RecordsManager)
    rm.run = SimpleNamespace(
        cfg=SimpleNamespace(
            ui_type=ui_type,
            runtime=SimpleNamespace(
                api_port=api_port, service_run_type=service_run_type
            ),
        )
    )
    rm._is_kubernetes_run = lambda: str(service_run_type).lower() == "kubernetes"
    return rm


def test_realtime_publish_gate_dashboard_always_enabled() -> None:
    from aiperf.plugin.enums import UIType

    rm = _publish_gate_manager(ui_type=UIType.DASHBOARD, api_port=None)
    with patch.object(rm_module.Environment.UI, "REALTIME_METRICS_ENABLED", False):
        assert rm_module.RecordsManager._realtime_metrics_publish_enabled(rm)


def test_realtime_publish_gate_none_ui_no_consumer_disabled() -> None:
    from aiperf.plugin.enums import UIType

    rm = _publish_gate_manager(ui_type=UIType.NONE, api_port=None)
    with patch.object(rm_module.Environment.UI, "REALTIME_METRICS_ENABLED", False):
        assert not rm_module.RecordsManager._realtime_metrics_publish_enabled(rm)


def test_realtime_publish_gate_none_ui_with_api_port_enabled() -> None:
    # A local API server serves the web dashboard even with --ui-type none.
    from aiperf.plugin.enums import UIType

    rm = _publish_gate_manager(ui_type=UIType.NONE, api_port=9090)
    with patch.object(rm_module.Environment.UI, "REALTIME_METRICS_ENABLED", False):
        assert rm_module.RecordsManager._realtime_metrics_publish_enabled(rm)


def test_realtime_publish_gate_none_ui_under_kubernetes_enabled() -> None:
    from aiperf.plugin.enums import UIType

    rm = _publish_gate_manager(
        ui_type=UIType.NONE, api_port=None, service_run_type="kubernetes"
    )
    with patch.object(rm_module.Environment.UI, "REALTIME_METRICS_ENABLED", False):
        assert rm_module.RecordsManager._realtime_metrics_publish_enabled(rm)


def test_realtime_publish_gate_none_ui_env_override_enabled() -> None:
    from aiperf.plugin.enums import UIType

    rm = _publish_gate_manager(ui_type=UIType.NONE, api_port=None)
    with patch.object(rm_module.Environment.UI, "REALTIME_METRICS_ENABLED", True):
        assert rm_module.RecordsManager._realtime_metrics_publish_enabled(rm)
