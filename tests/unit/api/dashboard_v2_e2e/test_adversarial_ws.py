# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial browser coverage for dashboard-v2 WebSocket handling."""

from __future__ import annotations

import time
from typing import Any

import orjson
from playwright.sync_api import expect

from .harness import DashboardHarness, DashboardScenario
from .helpers import dashboard_cfg, metric_result, realtime_metrics_payload


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
                "task_stats": {"in_progress": 0, "completed": 1, "failed": 0},
                "health": {"cpu_usage": 12.0, "memory_usage": 1024},
                "worker_statuses": {hostile_worker: "healthy"},
                "worker_startup_states": {hostile_worker: "ready"},
                "worker_task_stats": {
                    hostile_worker: {"in_progress": 0, "completed": 1, "failed": 0}
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
