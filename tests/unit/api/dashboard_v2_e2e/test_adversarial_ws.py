# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial browser coverage for dashboard-v2 WebSocket handling."""

from __future__ import annotations

import time
from typing import Any

import orjson
from playwright.sync_api import expect

from .harness import DashboardHarness, DashboardScenario
from .helpers import (
    dashboard_cfg,
    metric_result,
    phase_start_payload,
    realtime_metrics_payload,
)


def _phase_complete_payload(phase: str) -> dict[str, Any]:
    return {
        "type": "credit_phase_complete",
        "phase": phase,
        "stats": {
            "phase": phase,
            "start_ns": time.time_ns() - 5_000_000_000,
            "requests_end_ns": time.time_ns(),
            "total_expected_requests": 100,
            "final_requests_completed": 100,
            "requests_completed": 100,
        },
    }


def test_unknown_ws_type_logs_once_even_if_repeated(
    dashboard: DashboardHarness,
) -> None:
    """Repeated unknown message types should not flood the dashboard log."""
    unknown = {"type": "definitely_not_a_dashboard_v2_message", "payload": 1}
    scenario = DashboardScenario(ws_payloads=[unknown, unknown, unknown])

    dashboard.goto_dashboard(scenario)
    dashboard.wait_for_boot()

    unknown_entries = dashboard.page.locator(".log-msg").filter(
        has_text="Unknown WS message type: definitely_not_a_dashboard_v2_message"
    )
    expect(unknown_entries).to_have_count(1)
    dashboard.assert_no_console_errors()
    dashboard.assert_no_bad_responses()


def test_hostile_unknown_ws_type_renders_as_text_not_markup(
    dashboard: DashboardHarness,
) -> None:
    """Unknown WS message types with markup must render as inert log text."""
    dialogs: list[str] = []
    dashboard.page.on(
        "dialog", lambda dialog: (dialogs.append(dialog.message), dialog.dismiss())
    )
    hostile_type = (
        '<svg onload="alert(1)"><script>alert(2)</script><img src=x onerror="alert(3)">'
    )
    scenario = DashboardScenario(ws_payloads=[{"type": hostile_type}])

    dashboard.goto_dashboard(scenario)
    dashboard.wait_for_boot()

    log_pane = dashboard.page.locator(".log-pane")
    expect(log_pane).to_contain_text(f"Unknown WS message type: {hostile_type}")
    assert log_pane.locator("svg").count() == 0
    assert log_pane.locator("img").count() == 0
    assert log_pane.locator("script").count() == 0
    assert dialogs == []
    dashboard.assert_no_console_errors()
    dashboard.assert_no_bad_responses()


def test_terminal_phase_complete_is_not_overwritten_by_later_progress(
    dashboard: DashboardHarness,
) -> None:
    """A stale non-terminal progress push must not downgrade a complete phase."""
    phase = "profiling"
    scenario = DashboardScenario(
        ws_payloads=[
            _phase_complete_payload(phase),
            {
                "type": "credit_phase_progress",
                "phase": phase,
                "stats": {
                    "phase": phase,
                    "start_ns": time.time_ns() - 2_000_000_000,
                    "total_expected_requests": 100,
                    "requests_completed": 50,
                },
            },
        ],
    )

    dashboard.goto_dashboard(scenario)
    dashboard.wait_for_boot()

    phase_card = dashboard.page.locator(".phase-card").filter(has_text=phase)
    expect(phase_card).to_be_visible()
    expect(phase_card.locator(".phase-badge")).to_contain_text("Complete")
    expect(phase_card).to_contain_text("100.0%")
    expect(phase_card).to_contain_text("100 / 100")
    dashboard.assert_no_console_errors()
    dashboard.assert_no_bad_responses()


def test_terminal_phase_complete_is_not_overwritten_by_later_failure(
    dashboard: DashboardHarness,
) -> None:
    """A stale failed push must not downgrade a complete phase."""
    phase = "profiling"
    scenario = DashboardScenario(
        ws_payloads=[
            _phase_complete_payload(phase),
            {
                "type": "credit_phase_failed",
                "phase": phase,
                "stats": {
                    "phase": phase,
                    "start_ns": time.time_ns() - 2_000_000_000,
                    "total_expected_requests": 100,
                    "requests_completed": 50,
                },
            },
        ],
    )

    dashboard.goto_dashboard(scenario)
    dashboard.wait_for_boot()

    phase_card = dashboard.page.locator(".phase-card").filter(has_text=phase)
    expect(phase_card).to_be_visible()
    expect(phase_card.locator(".phase-badge")).to_contain_text("Complete")
    expect(phase_card).to_contain_text("100.0%")
    expect(phase_card).to_contain_text("100 / 100")
    dashboard.assert_no_console_errors()
    dashboard.assert_no_bad_responses()


def test_non_finite_metrics_render_fallback_without_nan_or_infinity_leaks(
    dashboard: DashboardHarness,
) -> None:
    """Parsed non-finite JSON numbers should render as fallback text only."""
    payload = {
        "type": "realtime_metrics",
        "metrics": [
            metric_result(
                "output_token_throughput",
                "Output Tokens/s",
                "tok/s",
                current=100.0,
                avg=95.0,
            ),
            metric_result(
                "request_throughput",
                "Requests/s",
                "req/s",
                current=10.0,
                avg=9.5,
            ),
            metric_result("time_to_first_token", "TTFT", "ms"),
        ],
    }
    raw_payload = (
        orjson.dumps(payload)
        .decode()
        .replace('"current":100.0', '"current":1e999')
        .replace(
            '"tag":"time_to_first_token"',
            '"tag":"time_to_first_token","current":1e999,"avg":-1e999,"p99":1e999',
        )
    )
    scenario = DashboardScenario(
        cfg=dashboard_cfg(slos={"time_to_first_token": 200.0}),
        ws_payloads=[raw_payload],
    )

    dashboard.goto_dashboard(scenario)
    dashboard.wait_for_boot()

    metrics_card = dashboard.page.locator(".card").filter(has_text="Realtime Metrics")
    expect(metrics_card).to_be_visible()
    ttft_tile = dashboard.page.locator(".kpi-tile").filter(has_text="TTFT")
    expect(ttft_tile).to_be_visible()
    expect(ttft_tile.locator(".kpi-big-val")).to_contain_text("---")
    page_text = dashboard.page.locator("body").inner_text()
    assert "NaN" not in page_text
    assert "Infinity" not in page_text
    dashboard.assert_no_console_errors()
    dashboard.assert_no_bad_responses()


def test_string_slo_metric_does_not_crash_or_count_as_violation(
    dashboard: DashboardHarness,
) -> None:
    """String metric values are ignored for SLO health instead of crashing the page."""
    scenario = DashboardScenario(
        cfg=dashboard_cfg(slos={"time_to_first_token": 200.0}),
        ws_payloads=[
            realtime_metrics_payload(
                {
                    **metric_result("time_to_first_token", "TTFT", "ms", avg=82.0),
                    "p99": "999",
                }
            )
        ],
    )

    dashboard.goto_dashboard(scenario)
    dashboard.wait_for_boot()

    expect(dashboard.page.locator(".hero-health-label")).to_contain_text("On target")
    expect(dashboard.page.locator(".kpi-tile").filter(has_text="TTFT")).to_be_visible()
    assert not [
        error for error in dashboard.console_errors if error.startswith("[pageerror]")
    ]
    dashboard.assert_no_console_errors()
    dashboard.assert_no_bad_responses()


def test_server_metrics_use_first_finite_series_stat(
    dashboard: DashboardHarness,
) -> None:
    """Server metrics skip empty series stats and show the first finite value."""
    endpoint_summaries = {
        "http://srv:8000": {
            "metrics": {
                "queue_depth": {
                    "unit": "requests",
                    "series": [
                        {"stats": {"avg": None}},
                        {"stats": {"avg": 64}},
                    ],
                }
            }
        }
    }
    scenario = DashboardScenario(
        server_metrics={"endpoint_summaries": endpoint_summaries},
        ws_payloads=[
            {
                "type": "realtime_server_metrics",
                "endpoint_summaries": endpoint_summaries,
            }
        ],
    )

    dashboard.goto_dashboard(scenario)
    dashboard.wait_for_boot()

    server_metrics = dashboard.page.locator(".server-metrics-card")
    expect(server_metrics).to_contain_text("queue_depth")
    expect(server_metrics).to_contain_text("64.00 requests")
    dashboard.assert_no_console_errors()
    dashboard.assert_no_bad_responses()


def test_hostile_worker_ids_and_text_render_as_text_not_markup(
    dashboard: DashboardHarness,
) -> None:
    """Worker identifiers from WS payloads must be escaped by the UI renderer."""
    dialogs: list[str] = []
    dashboard.page.on(
        "dialog", lambda dialog: (dialogs.append(dialog.message), dialog.dismiss())
    )
    hostile_group = 'group-<svg onload="alert(1)">-primary'
    hostile_worker = '<img src=x onerror="alert(1)">worker'
    scenario = DashboardScenario(
        ws_payloads=[
            {
                "type": "worker_group_stats",
                "group_id": hostile_group,
                "status": 'healthy"><script>alert(1)</script>',
                "startup_state": "ready",
                "declared_workers": 1,
                "ready_workers": 1,
                "task_stats": {"total": 1, "completed": 1, "failed": 0},
                "health": {"cpu_usage": 12.0, "memory_usage": 1024},
                "worker_statuses": {hostile_worker: "healthy"},
                "worker_startup_states": {hostile_worker: "ready"},
                "worker_task_stats": {
                    hostile_worker: {"total": 1, "completed": 1, "failed": 0}
                },
                "worker_health": {
                    hostile_worker: {"cpu_usage": 7.0, "memory_usage": 2048}
                },
            }
        ]
    )

    dashboard.goto_dashboard(scenario)
    dashboard.wait_for_boot()

    worker_table = dashboard.page.locator(".worker-table")
    expect(worker_table).to_be_visible()
    expect(worker_table).to_contain_text('<svg onload="alert(1)">')
    expect(worker_table).to_contain_text('<img src=x onerror="alert(1)">worker')
    assert dashboard.page.locator(".worker-table svg").count() == 0
    assert dashboard.page.locator(".worker-table img").count() == 0
    assert dashboard.page.locator(".worker-table script").count() == 0
    assert dialogs == []
    dashboard.assert_no_console_errors()
    dashboard.assert_no_bad_responses()


def test_hostile_config_model_and_phase_names_render_as_text(
    dashboard: DashboardHarness,
) -> None:
    """Config model and phase labels must render as inert text."""
    dialogs: list[str] = []
    dashboard.page.on(
        "dialog", lambda dialog: (dialogs.append(dialog.message), dialog.dismiss())
    )
    hostile_model = '<img src=x onerror="alert(1)">model'
    hostile_phase = '<svg onload="alert(2)">profiling'
    scenario = DashboardScenario(
        cfg=dashboard_cfg(
            models=[hostile_model],
            phases=[
                {
                    "name": "warmup",
                    "type": "concurrency",
                    "requests": 2,
                    "concurrency": 1,
                },
                {
                    "name": hostile_phase,
                    "type": "poisson",
                    "rate": 5,
                    "duration": 10,
                    "concurrency": 2,
                },
            ],
        )
    )

    dashboard.goto_dashboard(scenario)
    dashboard.wait_for_boot()

    config_bar = dashboard.page.locator("#config-bar.visible")
    expect(config_bar).to_contain_text(hostile_model)
    expect(config_bar).to_contain_text(hostile_phase)
    expect(config_bar).to_contain_text("poisson")
    assert config_bar.locator("svg").count() == 0
    assert config_bar.locator("img").count() == 0
    assert config_bar.locator("script").count() == 0
    assert dialogs == []
    dashboard.assert_no_console_errors()
    dashboard.assert_no_bad_responses()


def test_hostile_server_metric_names_and_endpoints_render_as_text(
    dashboard: DashboardHarness,
) -> None:
    """Server metric endpoint, metric name, and unit strings are inert text."""
    dialogs: list[str] = []
    dashboard.page.on(
        "dialog", lambda dialog: (dialogs.append(dialog.message), dialog.dismiss())
    )
    hostile_endpoint = '<svg onload="alert(1)">endpoint'
    hostile_metric = '<img src=x onerror="alert(2)">metric'
    hostile_unit = "<script>alert(3)</script>units"
    endpoint_summaries = {
        hostile_endpoint: {
            "metrics": {
                hostile_metric: {
                    "unit": hostile_unit,
                    "series": [{"stats": {"avg": 12.5}}],
                }
            }
        }
    }
    scenario = DashboardScenario(
        server_metrics={"endpoint_summaries": endpoint_summaries},
        ws_payloads=[
            {
                "type": "realtime_server_metrics",
                "endpoint_summaries": endpoint_summaries,
            }
        ],
    )

    dashboard.goto_dashboard(scenario)
    dashboard.wait_for_boot()

    server_metrics = dashboard.page.locator(".server-metrics-card")
    expect(server_metrics).to_contain_text(hostile_endpoint)
    expect(server_metrics).to_contain_text(hostile_metric)
    expect(server_metrics).to_contain_text(f"12.50 {hostile_unit}")
    assert server_metrics.locator("svg").count() == 0
    assert server_metrics.locator("img").count() == 0
    assert server_metrics.locator("script").count() == 0
    assert dialogs == []
    dashboard.assert_no_console_errors()
    dashboard.assert_no_bad_responses()


def test_server_metrics_empty_ws_update_clears_prior_rows(
    dashboard: DashboardHarness,
) -> None:
    """An empty server-metrics WS update replaces prior rows with the empty state."""
    endpoint_summaries = {
        "http://srv:8000": {
            "metrics": {
                "queue_depth": {
                    "unit": "requests",
                    "series": [{"stats": {"avg": 64}}],
                }
            }
        }
    }
    scenario = DashboardScenario(
        server_metrics={"endpoint_summaries": endpoint_summaries},
        ws_payloads=[
            {
                "type": "realtime_server_metrics",
                "endpoint_summaries": {},
            }
        ],
    )

    dashboard.goto_dashboard(scenario)
    dashboard.wait_for_boot()

    server_metrics = dashboard.page.locator(".server-metrics-card")
    expect(server_metrics).to_contain_text("No server-side metrics reported yet.")
    expect(server_metrics).not_to_contain_text("queue_depth")
    expect(server_metrics).not_to_contain_text("http://srv:8000")
    dashboard.assert_no_console_errors()
    dashboard.assert_no_bad_responses()


def test_stale_reconnect_progress_response_does_not_clobber_newer_ws_phase(
    dashboard: DashboardHarness,
) -> None:
    """A reconnect warm-start progress snapshot must not downgrade newer WS progress."""
    phase = "profiling"
    scenario = DashboardScenario(
        progress={
            "phases": {
                phase: {
                    "phase": phase,
                    "start_ns": time.time_ns() - 4_000_000_000,
                    "total_expected_requests": 100,
                    "requests_completed": 10,
                }
            }
        },
        ws_payloads=[
            phase_start_payload(
                phase,
                total_expected_requests=100,
                requests_completed=75,
                start_ns=time.time_ns() - 2_000_000_000,
            )
        ],
        close_ws_after_payloads=True,
    )

    dashboard.goto_dashboard(scenario)
    dashboard.page.wait_for_selector("#config-bar.visible", timeout=10_000)
    dashboard.page.wait_for_function(
        """() => {
            const text = document.querySelector('.status-bar')?.textContent ?? '';
            return text.includes('Disconnected');
        }""",
        timeout=10_000,
    )

    phase_card = dashboard.page.locator(".phase-card").filter(has_text=phase)
    expect(phase_card).to_be_visible()
    expect(phase_card).to_contain_text("75.0%")
    expect(phase_card).to_contain_text("75 / 100")
    dashboard.assert_no_console_errors()
    dashboard.assert_no_bad_responses()


def test_websocket_close_after_payload_leaves_app_usable(
    dashboard: DashboardHarness,
) -> None:
    """The dashboard should survive the server closing WS after a valid payload."""
    scenario = DashboardScenario(
        ws_payloads=[
            realtime_metrics_payload(
                metric_result(
                    "request_throughput",
                    "Requests/s",
                    "req/s",
                    current=12.0,
                    avg=11.5,
                )
            )
        ],
        close_ws_after_payloads=True,
    )

    dashboard.goto_dashboard(scenario)
    dashboard.page.wait_for_selector("#config-bar.visible", timeout=10_000)
    dashboard.page.wait_for_function(
        """() => {
            const text = document.querySelector('.status-bar')?.textContent ?? '';
            return text.includes('Connected') || text.includes('Disconnected');
        }""",
        timeout=10_000,
    )

    expect(dashboard.page.locator(".topbar")).to_be_visible()
    expect(dashboard.page.locator("#config-bar.visible")).to_be_visible()
    expect(dashboard.page.get_by_text("AIPerf Dashboard").first).to_be_visible()
    expect(dashboard.page.locator("body")).to_contain_text("llama3-8b")
    dashboard.page.get_by_role("button", name="warn+").click()
    expect(dashboard.page.locator(".log-pane")).to_be_visible()
    dashboard.assert_no_console_errors()
    dashboard.assert_no_bad_responses()


def test_websocket_transient_close_preserves_last_live_values(
    dashboard: DashboardHarness,
) -> None:
    """A reconnecting dashboard keeps last phase and metric values visible."""
    phase = "profiling"
    scenario = DashboardScenario(
        ws_payloads=[
            {
                "type": "credit_phase_progress",
                "phase": phase,
                "stats": {
                    "phase": phase,
                    "start_ns": time.time_ns() - 2_000_000_000,
                    "total_expected_requests": 100,
                    "requests_completed": 25,
                },
            },
            realtime_metrics_payload(
                metric_result(
                    "request_throughput",
                    "Requests/s",
                    "req/s",
                    current=12.0,
                    avg=11.5,
                )
            ),
        ],
        close_ws_after_payloads=True,
    )

    dashboard.goto_dashboard(scenario)
    dashboard.page.wait_for_selector("#config-bar.visible", timeout=10_000)
    dashboard.page.wait_for_function(
        """() => {
            const text = document.querySelector('.status-bar')?.textContent ?? '';
            return text.includes('Disconnected');
        }""",
        timeout=10_000,
    )

    expect(dashboard.page.locator(".status-dot.disconnected")).to_be_visible()
    phase_card = dashboard.page.locator(".phase-card").filter(has_text=phase)
    expect(phase_card).to_be_visible()
    expect(phase_card.locator(".phase-badge")).to_contain_text("Running")
    expect(phase_card).to_contain_text("25.0%")
    expect(
        dashboard.page.locator(".kpi-tile").filter(has_text="Requests/s")
    ).to_contain_text("12.00")
    dashboard.assert_no_console_errors()
    dashboard.assert_no_bad_responses()
