# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests: the error summary must survive a total-failure run.

``_print_post_benchmark_info_and_metrics`` has two early-return paths that fire
before ``ExporterManager`` is constructed:

* no records at all
* records exist but every request failed

``ExporterManager`` is what drives ``ConsoleErrorExporter``, so before this fix
both paths discarded the already-populated ``error_summary`` and left the
operator with a single aggregate line. That is the least diagnostic output in
precisely the runs that need it most, and it hid the HTTP status code.
"""

from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from aiperf.common.models import (
    ErrorDetails,
    ErrorDetailsCount,
    MetricResult,
    ProfileResults,
)


def make_record() -> MetricResult:
    """A single metric record, enough to get past the empty-records guard."""
    return MetricResult(tag="request_latency", header="Request Latency", unit="ms")


def make_profile_results(*, records, successful, errors, error_summary):
    """Wrap a ProfileResults in the message-shaped object the controller holds."""
    results = ProfileResults(
        records=records,
        completed=successful + errors,
        start_ns=0,
        end_ns=1,
        successful_request_count=successful,
        error_request_count=errors,
        error_summary=error_summary,
    )
    message = MagicMock()
    message.results = results
    return message


def summary(code=404, type="Not Found", message="no such path", count=100):
    return [
        ErrorDetailsCount(
            error_details=ErrorDetails(code=code, type=type, message=message),
            count=count,
        )
    ]


async def run_report(controller) -> str:
    """Invoke the reporting path with every console redirected to a recorder."""
    recorder = Console(record=True, width=200)

    with (
        patch(
            "aiperf.controller.system_controller.Console",
            return_value=recorder,
        ),
        patch(
            "aiperf.controller.system_controller.ExporterManager"
        ) as mock_exporter_manager,
    ):
        await controller._print_post_benchmark_info_and_metrics()

    return recorder.export_text(), mock_exporter_manager


class TestAllRequestsFailed:
    """Records exist, but every request errored."""

    @pytest.fixture
    def controller(self, system_controller):
        system_controller._profile_results = make_profile_results(
            records=[make_record()],
            successful=0,
            errors=100,
            error_summary=summary(),
        )
        system_controller._telemetry_results = None
        system_controller._server_metrics_results = None
        return system_controller

    async def test_error_table_is_printed(self, controller):
        """The regression: the Code/Type/Message/Count table must appear."""
        out, _ = await run_report(controller)

        assert "Error Summary" in out
        for header in ("Code", "Type", "Message", "Count"):
            assert header in out, f"missing column header: {header}"

    async def test_http_status_code_is_visible(self, controller):
        """The status code is the field the operator actually needs."""
        out, _ = await run_report(controller)

        assert "404" in out
        assert "no such path" in out

    async def test_aggregate_line_still_recorded_as_exit_error(self, controller):
        """Exit code behaviour is unchanged: _exit_errors stays non-empty."""
        await run_report(controller)

        assert len(controller._exit_errors) == 1
        assert "All 100 inference" in controller._exit_errors[0].error_details.message

    async def test_message_no_longer_points_at_absent_log_output(self, controller):
        """Per-request details are not logged at INFO, so stop citing them."""
        await run_report(controller)

        message = controller._exit_errors[0].error_details.message
        assert "prior log output" not in message
        assert "error summary table" in message

    async def test_full_exporter_manager_is_not_run(self, controller):
        """Scope guard: we add console output without writing new artifacts.

        Falling through to ExporterManager would newly emit
        profile_export_aiperf.json/csv on runs that previously produced none,
        which is a behaviour change for anything automating over the artifact
        directory.
        """
        _, mock_exporter_manager = await run_report(controller)

        mock_exporter_manager.assert_not_called()


class TestNoRecordsCollected:
    """No records at all, but errors were still tracked."""

    @pytest.fixture
    def controller(self, system_controller):
        system_controller._profile_results = make_profile_results(
            records=[],
            successful=0,
            errors=20,
            error_summary=summary(code=401, type="Unauthorized", message="bad token"),
        )
        system_controller._telemetry_results = None
        system_controller._server_metrics_results = None
        return system_controller

    async def test_error_table_is_printed(self, controller):
        """This path is reached when the accumulator emits no metric records."""
        out, _ = await run_report(controller)

        assert "Error Summary" in out
        assert "401" in out
        assert "bad token" in out

    async def test_exit_error_still_recorded(self, controller):
        await run_report(controller)

        assert len(controller._exit_errors) == 1


class TestNoErrorsAtAll:
    """A clean run must not grow a spurious empty table."""

    @pytest.fixture
    def controller(self, system_controller):
        system_controller._profile_results = make_profile_results(
            records=[],
            successful=0,
            errors=0,
            error_summary=[],
        )
        system_controller._telemetry_results = None
        system_controller._server_metrics_results = None
        return system_controller

    async def test_no_table_when_summary_empty(self, controller):
        out, _ = await run_report(controller)

        assert "Error Summary" not in out
