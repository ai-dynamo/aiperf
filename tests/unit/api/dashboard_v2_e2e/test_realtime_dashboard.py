# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Golden-path browser coverage for dashboard-v2 realtime rendering."""

from __future__ import annotations

import time
from typing import Any

from playwright.sync_api import expect

from tests.unit.api.dashboard_v2_e2e.harness import DashboardHarness, DashboardScenario
from tests.unit.api.dashboard_v2_e2e.helpers import (
    dashboard_cfg,
    metric_result,
    phase_start_payload,
    realtime_metrics_payload,
)


def _realtime_metrics(
    *, request_count: int = 100, good_count: int = 98
) -> dict[str, Any]:
    return realtime_metrics_payload(
        metric_result(
            "output_token_throughput",
            "Output Tokens/s",
            "tok/s",
            current=1320.0,
            avg=1305.0,
        ),
        metric_result(
            "request_throughput",
            "Requests/s",
            "req/s",
            current=22.4,
            avg=21.8,
        ),
        metric_result(
            "time_to_first_token",
            "TTFT",
            "ms",
            avg=82.0,
            p99=125.0,
        ),
        metric_result(
            "request_latency",
            "Request Latency",
            "ms",
            avg=510.0,
            p99=820.0,
        ),
        metric_result(
            "inter_token_latency",
            "ITL",
            "ms",
            avg=11.0,
            p99=24.0,
        ),
        metric_result(
            "request_count",
            "Requests",
            "",
            current=float(request_count),
            avg=float(request_count),
        ),
        metric_result(
            "good_request_count",
            "Good Requests",
            "",
            current=float(good_count),
            avg=float(good_count),
        ),
        metric_result("goodput", "Goodput", "req/s", current=21.9, avg=21.9),
    )


def _server_endpoint_summaries() -> dict[str, Any]:
    return {
        "http://srv:8000": {
            "metrics": {
                "kv_cache_utilization": {
                    "unit": "ratio",
                    "series": [{"stats": {"avg": 0.92}}],
                },
                "queue_depth": {
                    "unit": "requests",
                    "series": [{"stats": {"avg": 64}}],
                },
                "tokens_total": {
                    "unit": "tokens",
                    "series": [{"stats": {"value": 125000}}],
                },
            }
        }
    }


def _server_metrics_payload() -> dict[str, Any]:
    return {
        "type": "realtime_server_metrics",
        "endpoint_summaries": _server_endpoint_summaries(),
    }


def _worker_group_payload() -> dict[str, Any]:
    return {
        "type": "worker_group_stats",
        "group_id": "worker-group-dashboard-v2-primary",
        "status": "healthy",
        "startup_state": "ready",
        "declared_workers": 2,
        "ready_workers": 2,
        # Real wire shape: WorkerTaskStats serializes only total/completed/failed;
        # in_progress is a non-serialized @property the client derives.
        "task_stats": {"completed": 97, "failed": 1, "total": 101},
        "health": {"cpu_usage": 41.2, "memory_usage": 3_221_225_472},
        "worker_statuses": {
            "worker-dashboard-v2-a": "healthy",
            "worker-dashboard-v2-b": "high_load",
        },
        "worker_startup_states": {
            "worker-dashboard-v2-a": "ready",
            "worker-dashboard-v2-b": "ready",
        },
        "worker_task_stats": {
            "worker-dashboard-v2-a": {"total": 52, "completed": 51, "failed": 0},
            "worker-dashboard-v2-b": {"total": 49, "completed": 46, "failed": 1},
        },
        "worker_health": {
            "worker-dashboard-v2-a": {"cpu_usage": 32.4, "memory_usage": 1_610_612_736},
            "worker-dashboard-v2-b": {"cpu_usage": 87.9, "memory_usage": 1_718_272_000},
        },
    }


def _gpu_telemetry_payload() -> dict[str, Any]:
    endpoint = "http://srv:9400"
    model = "NVIDIA H100 80GB HBM3"
    return {
        "type": "realtime_telemetry_metrics",
        "metrics": [
            metric_result(
                "gpu_power_usage_dcgm_gpu_0",
                f"GPU Power Usage | {endpoint} | GPU 0 | {model}",
                "W",
                current=421.5,
            ),
            metric_result(
                "gpu_utilization_dcgm_gpu_0",
                f"GPU Utilization | {endpoint} | GPU 0 | {model}",
                "%",
                current=96.0,
            ),
            metric_result(
                "gpu_temperature_dcgm_gpu_0",
                f"GPU Temperature | {endpoint} | GPU 0 | {model}",
                "C",
                current=73.0,
            ),
            metric_result(
                "gpu_memory_used_dcgm_gpu_0",
                f"GPU Memory Used | {endpoint} | GPU 0 | {model}",
                "MiB",
                current=64212.0,
            ),
            metric_result(
                "gpu_sm_clock_dcgm_gpu_0",
                f"SM Clock | {endpoint} | GPU 0 | {model}",
                "MHz",
                current=1590.0,
            ),
        ],
    }


def test_phase_cards_status_and_hero_render_from_realtime_payloads(
    dashboard: DashboardHarness,
) -> None:
    """A phase start plus metrics updates the hero, status, and phase card."""
    scenario = DashboardScenario(
        cfg=dashboard_cfg(slos={"request_latency": 1000.0}),
        ws_payloads=[
            phase_start_payload(
                "profiling",
                total_expected_requests=200,
                requests_completed=50,
                start_ns=time.time_ns() - 5_000_000_000,
            ),
            _realtime_metrics(request_count=100, good_count=100),
        ],
    )

    dashboard.goto_dashboard(scenario)
    dashboard.wait_for_boot()

    expect(dashboard.page.locator(".hero-health-label")).to_contain_text("On target")
    expect(dashboard.page.locator(".hero-phase-name")).to_contain_text("profiling")
    expect(dashboard.page.locator(".hero-phase-sub")).to_contain_text("50 / 200")
    expect(dashboard.page.locator(".hero-phase-sub")).to_contain_text("completed")
    phase_card = dashboard.page.locator(".phase-card").filter(has_text="profiling")
    expect(phase_card).to_be_visible()
    expect(phase_card.locator(".phase-badge")).to_contain_text("Running")
    expect(phase_card).to_contain_text("25.0%")
    expect(dashboard.page.locator(".status-dot.connected")).to_be_visible()
    dashboard.assert_no_console_errors()
    dashboard.assert_no_bad_responses()


def test_kpi_tiles_goodput_success_rate_and_chart_render(
    dashboard: DashboardHarness,
) -> None:
    """Realtime metric batches populate KPI tiles and the Chart.js canvas."""
    scenario = DashboardScenario(
        cfg=dashboard_cfg(slos={"request_latency": 1000.0}),
        ws_payloads=[
            _realtime_metrics(request_count=80, good_count=80),
            _realtime_metrics(request_count=100, good_count=98),
        ],
    )

    dashboard.goto_dashboard(scenario)
    dashboard.wait_for_boot()

    metrics_card = dashboard.page.locator(".card").filter(has_text="Realtime Metrics")
    expect(metrics_card).to_contain_text("Output Tokens/s")
    expect(metrics_card).to_contain_text("Requests/s")
    expect(metrics_card).to_contain_text("TTFT")
    expect(metrics_card).to_contain_text("Goodput")
    expect(metrics_card).to_contain_text("2 failed")
    expect(metrics_card).to_contain_text("98.0%")

    chart = dashboard.page.locator(
        'canvas[aria-label="throughput and latency over time"]'
    )
    expect(chart).to_be_visible()
    dashboard.page.wait_for_function(
        """() => {
            const canvas = document.querySelector(
                'canvas[aria-label="throughput and latency over time"]'
            );
            return Boolean(globalThis.Chart?.getChart?.(canvas));
        }"""
    )

    success_scenario = DashboardScenario(
        cfg=dashboard_cfg(slos={}),
        ws_payloads=[
            realtime_metrics_payload(
                metric_result(
                    "request_count",
                    "Requests",
                    "",
                    current=125.0,
                    avg=125.0,
                ),
                metric_result(
                    "error_request_count",
                    "Errors",
                    "",
                    current=1.0,
                    avg=1.0,
                ),
                metric_result(
                    "error_request_rate",
                    "Error Rate",
                    "%",
                    current=0.8,
                    avg=0.8,
                ),
            )
        ],
    )
    dashboard.goto_dashboard(success_scenario)
    dashboard.wait_for_boot()

    success_card = dashboard.page.locator(".kpi-tile").filter(has_text="Success Rate")
    expect(success_card).to_be_visible()
    expect(success_card).to_contain_text("99.20%")
    expect(success_card).to_contain_text("1 errors")
    dashboard.assert_no_console_errors()
    dashboard.assert_no_bad_responses()


def test_full_metrics_tables_render_benchmark_gpu_and_server_rows(
    dashboard: DashboardHarness,
) -> None:
    scenario = DashboardScenario(
        cfg=dashboard_cfg(),
        server_metrics={"endpoint_summaries": _server_endpoint_summaries()},
        ws_payloads=[
            _realtime_metrics(request_count=100, good_count=98),
            _gpu_telemetry_payload(),
            _server_metrics_payload(),
        ],
    )

    dashboard.goto_dashboard(scenario)
    dashboard.wait_for_boot()

    dashboard.page.wait_for_selector(".full-metrics-card", timeout=10_000)

    benchmark = dashboard.page.locator(".full-metrics-card").filter(
        has_text="Full Benchmark Metrics"
    )
    expect(benchmark).to_be_visible()
    expect(benchmark).to_contain_text("Request Latency")
    expect(benchmark).to_contain_text("TTFT")
    expect(benchmark.locator("th")).to_contain_text(
        ["Metric", "avg", "min", "max", "p99", "p90", "p50"]
    )

    gpu = dashboard.page.locator(".full-metrics-card").filter(
        has_text="Full GPU Telemetry Metrics"
    )
    expect(gpu).to_be_visible()
    expect(gpu).to_contain_text("GPU Utilization")
    expect(gpu).to_contain_text("96.00")
    expect(gpu).to_contain_text("SM Clock")

    server = dashboard.page.locator(".full-metrics-card").filter(
        has_text="Full Server Metrics"
    )
    expect(server).to_be_visible()
    expect(server).to_contain_text("http://srv:8000 · kv_cache_utilization")
    expect(server).to_contain_text("ratio")
    expect(server).to_contain_text("0.92")

    dashboard.assert_no_console_errors()
    dashboard.assert_no_bad_responses()


def test_gpu_worker_logs_records_and_server_metrics_render(
    dashboard: DashboardHarness,
) -> None:
    """Operational cards render GPU, worker, records, logs, and server metrics."""
    scenario = DashboardScenario(
        server_metrics={"endpoint_summaries": _server_endpoint_summaries()},
        ws_payloads=[
            _gpu_telemetry_payload(),
            _worker_group_payload(),
            {
                "type": "processing_stats",
                "processing_stats": {
                    "success_records": 97,
                    "error_records": 1,
                    "final_requests_completed": 98,
                    "start_ns": time.time_ns() - 3_000_000_000,
                },
            },
            {
                "type": "all_records_received",
                "final_processing_stats": {
                    "success_records": 99,
                    "error_records": 1,
                    "final_requests_completed": 100,
                    "records_end_ns": time.time_ns(),
                },
            },
            _server_metrics_payload(),
        ],
    )

    dashboard.goto_dashboard(scenario)
    dashboard.wait_for_boot()

    gpu_card = dashboard.page.locator(".gpu-card").filter(has_text="GPU 0")
    expect(gpu_card).to_be_visible()
    expect(gpu_card).to_contain_text("NVIDIA H100 80GB HBM3")
    expect(gpu_card).to_contain_text("Power")
    expect(gpu_card).to_contain_text("421.5 W")
    expect(gpu_card).to_contain_text("SM Clock")

    workers = dashboard.page.locator(".worker-table")
    expect(workers).to_contain_text("primary")
    expect(workers).to_contain_text("2/2 ready")
    expect(workers).to_contain_text("v2-a")
    expect(workers).to_contain_text("high load")
    expect(workers).to_contain_text("97")
    # In-flight is derived client-side as total - completed - failed because
    # the wire WorkerTaskStats never serializes in_progress.
    group_row = dashboard.page.locator(".worker-group-row")
    expect(group_row.locator("td").nth(2)).to_have_text("3")

    # Records totals prove the client read the real wire keys
    # (processing_stats / final_processing_stats), not a nonexistent `stats`.
    records_item = dashboard.page.locator(".status-item").filter(has_text="Records")
    expect(records_item.locator(".status-val")).to_have_text("100")
    expect(dashboard.page.locator(".status-bar")).to_contain_text("complete")

    server_metrics = dashboard.page.locator(".server-metrics-card")
    expect(server_metrics).to_contain_text("http://srv:8000")
    expect(server_metrics).to_contain_text("kv_cache_utilization")
    expect(server_metrics).to_contain_text("saturated")
    expect(server_metrics).to_contain_text("queue_depth")

    log = dashboard.page.locator(".log-pane")
    expect(log).to_contain_text("All records received")
    expect(log).to_contain_text("records")
    dashboard.assert_no_console_errors()
    dashboard.assert_no_bad_responses()
