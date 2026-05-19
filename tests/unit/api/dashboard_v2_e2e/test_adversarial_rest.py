# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial REST coverage for dashboard-v2 browser bootstrapping."""

from __future__ import annotations

import time
from typing import Any

import orjson
import pytest
from fastapi.responses import JSONResponse
from playwright.sync_api import expect
from starlette.background import BackgroundTask

from .harness import DashboardHarness, DashboardScenario
from .helpers import dashboard_cfg


def _now_ns() -> int:
    return time.time_ns()


def _visible_text(dashboard: DashboardHarness) -> str:
    return dashboard.page.locator("body").inner_text()


def _assert_real_dashboard_v2_navigation(dashboard: DashboardHarness) -> None:
    assert "/dashboard-v2" in dashboard.page.url


def test_progress_500_keeps_dashboard_usable_and_visible_failure_logged(
    dashboard: DashboardHarness,
) -> None:
    """A failed progress warm-start is visible but does not crash the app."""
    scenario = DashboardScenario()

    def _restore_progress_route() -> None:
        scenario.rest_overrides.pop("/api/progress", None)

    scenario.rest_overrides["/api/progress"] = (
        500,
        JSONResponse(
            {"error": "boom"},
            background=BackgroundTask(_restore_progress_route),
        ),
    )

    dashboard.goto_dashboard(scenario)
    dashboard.wait_for_boot()
    _assert_real_dashboard_v2_navigation(dashboard)

    expect(dashboard.page.locator(".topbar")).to_be_visible()
    expect(dashboard.page.locator(".status-dot.connected")).to_be_visible()
    expect(dashboard.page.locator("#config-bar.visible")).to_contain_text("llama3-8b")
    expect(dashboard.page.locator(".log-pane")).to_contain_text(
        "fetch failed: /api/progress"
    )
    assert len(dashboard.bad_responses) == 1, dashboard.bad_responses
    bad_response = dashboard.bad_responses[0]
    assert bad_response.startswith("500 GET "), dashboard.bad_responses
    assert bad_response.endswith("/api/progress"), dashboard.bad_responses
    assert not [
        error for error in dashboard.console_errors if error.startswith("[pageerror]")
    ]


@pytest.mark.parametrize(
    "endpoint_summaries",
    [
        pytest.param([], id="list-instead-of-map"),
        pytest.param({"http://srv:8000": "not-a-summary"}, id="primitive-endpoint-body"),
        pytest.param({"http://srv:8000": {"metrics": "not-a-map"}}, id="primitive-metrics"),
        pytest.param({"http://srv:8000": {"metrics": {"queue_depth": "not-a-metric"}}}, id="primitive-metric"),
    ],
)  # fmt: skip
def test_malformed_server_metrics_shapes_render_empty_state_without_console_errors(
    dashboard: DashboardHarness,
    endpoint_summaries: Any,
) -> None:
    """Malformed server-metrics payloads degrade to the server metrics empty state."""
    scenario = DashboardScenario(
        server_metrics={"endpoint_summaries": endpoint_summaries}
    )

    dashboard.goto_dashboard(scenario)
    dashboard.wait_for_boot()
    _assert_real_dashboard_v2_navigation(dashboard)

    server_metrics = dashboard.page.locator(".server-metrics-card")
    expect(server_metrics).to_contain_text("No server-side metrics reported yet.")
    assert "not-a-summary" not in _visible_text(dashboard)
    dashboard.assert_no_console_errors()
    dashboard.assert_no_bad_responses()


def test_missing_phase_fields_do_not_create_broken_cards(
    dashboard: DashboardHarness,
) -> None:
    """Malformed progress phases are ignored while valid phases still render."""
    scenario = DashboardScenario(
        progress={
            "phases": {
                "missing-start": {
                    "requests_completed": 7,
                    "total_expected_requests": 10,
                },
                "profiling": {
                    "start_ns": _now_ns() - 2_000_000_000,
                    "requests_completed": 8,
                    "total_expected_requests": 20,
                },
            }
        }
    )

    dashboard.goto_dashboard(scenario)
    dashboard.wait_for_boot()
    _assert_real_dashboard_v2_navigation(dashboard)

    valid_phase = dashboard.page.locator(".phase-card").filter(has_text="profiling")
    expect(valid_phase).to_be_visible()
    expect(valid_phase).to_contain_text("40.0%")
    expect(
        dashboard.page.locator(".phase-card").filter(has_text="missing-start")
    ).to_have_count(0)
    assert "NaN" not in _visible_text(dashboard)
    assert "undefined" not in _visible_text(dashboard)
    dashboard.assert_no_console_errors()
    dashboard.assert_no_bad_responses()


def test_config_response_redacts_api_key_and_keeps_visible_endpoint_fields(
    dashboard: DashboardHarness,
) -> None:
    """The browser receives endpoint config minus secrets, and still renders it."""
    config_payloads: list[dict[str, Any]] = []

    def _capture_config_response(response: Any) -> None:
        if response.url.endswith("/api/config") and response.status == 200:
            config_payloads.append(response.json())

    dashboard.page.on("response", _capture_config_response)
    dashboard.goto_dashboard(DashboardScenario(cfg=dashboard_cfg()))
    dashboard.wait_for_boot()
    _assert_real_dashboard_v2_navigation(dashboard)

    assert config_payloads
    latest_config = config_payloads[-1]
    serialized = orjson.dumps(latest_config, option=orjson.OPT_SORT_KEYS).decode()
    endpoint = latest_config["endpoint"]
    assert "api_key" not in serialized
    assert endpoint["urls"] == ["http://srv:8000/v1/chat/completions"]
    assert endpoint["type"] == "chat"
    assert endpoint["streaming"] is True

    config_bar = dashboard.page.locator("#config-bar.visible")
    expect(config_bar).to_contain_text("chat (streaming)")
    expect(config_bar).to_contain_text("http://srv:8000/v1/chat/completions")
    dashboard.assert_no_console_errors()
    dashboard.assert_no_bad_responses()


def test_config_bar_redacts_endpoint_url_credentials_and_api_key_query(
    dashboard: DashboardHarness,
) -> None:
    """Displayed endpoint URLs omit credentials and secret query values only."""
    scenario = DashboardScenario(
        cfg=dashboard_cfg(
            endpoint_urls=[
                "https://user:SECRET_PASSWORD@api.example.test/v1/chat/completions?api_key=SECRET_TOKEN&region=us-west"
            ]
        )
    )

    dashboard.goto_dashboard(scenario)
    dashboard.wait_for_boot()

    config_bar = dashboard.page.locator("#config-bar.visible")
    expect(config_bar).to_contain_text("https://api.example.test/v1/chat/completions")
    expect(config_bar).to_contain_text("region=us-west")
    assert "SECRET_PASSWORD" not in config_bar.inner_text()
    assert "SECRET_TOKEN" not in config_bar.inner_text()
    dashboard.assert_no_console_errors()
    dashboard.assert_no_bad_responses()
