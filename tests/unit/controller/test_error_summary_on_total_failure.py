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

from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from rich.console import Console

from aiperf.common.models import (
    ErrorDetails,
    ErrorDetailsCount,
    ExitErrorInfo,
    MetricResult,
    ProcessRecordsResult,
    ProfileResults,
)
from aiperf.controller.system_controller import SystemController


def make_record() -> MetricResult:
    """A single metric record, enough to get past the empty-records guard."""
    return MetricResult(tag="request_latency", header="Request Latency", unit="ms")


def make_profile_results(
    *,
    records: list[MetricResult],
    successful: int,
    errors: int,
    error_summary: list[ErrorDetailsCount],
) -> MagicMock:
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


def summary(
    code: int | None = 404,
    error_type: str | None = "Not Found",
    message: str = "no such path",
    count: int = 100,
) -> list[ErrorDetailsCount]:
    return [
        ErrorDetailsCount(
            error_details=ErrorDetails(code=code, type=error_type, message=message),
            count=count,
        )
    ]


async def run_report(controller: SystemController) -> tuple[str, MagicMock]:
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
    def controller(self, system_controller: SystemController) -> SystemController:
        system_controller._profile_results = make_profile_results(
            records=[make_record()],
            successful=0,
            errors=100,
            error_summary=summary(),
        )
        system_controller._telemetry_results = None
        system_controller._server_metrics_results = None
        return system_controller

    async def test_error_table_is_printed(self, controller: SystemController) -> None:
        """The regression: the Code/Type/Message/Count table must appear."""
        out, _ = await run_report(controller)

        assert "Error Summary" in out
        for header in ("Code", "Type", "Message", "Count"):
            assert header in out, f"missing column header: {header}"

    async def test_http_status_code_is_visible(
        self, controller: SystemController
    ) -> None:
        """The status code is the field the operator actually needs."""
        out, _ = await run_report(controller)

        assert "404" in out
        assert "no such path" in out

    async def test_aggregate_line_still_recorded_as_exit_error(
        self, controller: SystemController
    ) -> None:
        """Exit code behaviour is unchanged: _exit_errors stays non-empty."""
        await run_report(controller)

        assert len(controller._exit_errors) == 1
        assert "All 100 inference" in controller._exit_errors[0].error_details.message

    async def test_message_no_longer_points_at_absent_log_output(
        self, controller: SystemController
    ) -> None:
        """Per-request details are not logged at INFO, so stop citing them."""
        await run_report(controller)

        message = controller._exit_errors[0].error_details.message
        assert "prior log output" not in message
        assert "error summary table" in message

    async def test_full_exporter_manager_is_not_run(
        self, controller: SystemController
    ) -> None:
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
    def controller(self, system_controller: SystemController) -> SystemController:
        system_controller._profile_results = make_profile_results(
            records=[],
            successful=0,
            errors=20,
            error_summary=summary(
                code=401, error_type="Unauthorized", message="bad token"
            ),
        )
        system_controller._telemetry_results = None
        system_controller._server_metrics_results = None
        return system_controller

    async def test_error_table_is_printed(self, controller: SystemController) -> None:
        """This path is reached when the accumulator emits no metric records."""
        out, _ = await run_report(controller)

        assert "Error Summary" in out
        assert "401" in out
        assert "bad token" in out

    async def test_exit_error_still_recorded(
        self, controller: SystemController
    ) -> None:
        await run_report(controller)

        assert len(controller._exit_errors) == 1


class TestNoErrorsAtAll:
    """A clean run must not grow a spurious empty table."""

    @pytest.fixture
    def controller(self, system_controller: SystemController) -> SystemController:
        system_controller._profile_results = make_profile_results(
            records=[],
            successful=0,
            errors=0,
            error_summary=[],
        )
        system_controller._telemetry_results = None
        system_controller._server_metrics_results = None
        return system_controller

    async def test_no_table_when_summary_empty(
        self, controller: SystemController
    ) -> None:
        out, _ = await run_report(controller)

        assert "Error Summary" not in out


class TestRenderFailureIsolation:
    """A failure while rendering the table must not hide the exit-error panel.

    Both callers print the exit-error panel and the log-file path immediately
    after the table. Those are the operator's remaining diagnostics on a run
    that already failed, so a rendering error must not take them down with it.
    """

    @pytest.fixture
    def controller(self, system_controller: SystemController) -> SystemController:
        system_controller._profile_results = make_profile_results(
            records=[make_record()],
            successful=0,
            errors=100,
            error_summary=summary(message="body with [/INST] in it"),
        )
        system_controller._telemetry_results = None
        system_controller._server_metrics_results = None
        return system_controller

    async def test_server_markup_renders_and_panel_survives(
        self, controller: SystemController
    ) -> None:
        """End to end: hostile server text renders, and the panel still prints."""
        out, _ = await run_report(controller)

        assert "[/INST]" in out
        assert "Log File" in out

    async def test_panel_survives_arbitrary_exporter_failure(
        self, controller: SystemController
    ) -> None:
        """Even an unrelated crash in the exporter must not eat the panel."""
        recorder = Console(record=True, width=200)
        with (
            patch("aiperf.controller.system_controller.Console", return_value=recorder),
            patch("aiperf.controller.system_controller.ExporterManager"),
            patch(
                "aiperf.controller.system_controller.ConsoleErrorExporter",
                side_effect=RuntimeError("boom"),
            ),
        ):
            await controller._print_post_benchmark_info_and_metrics()

        out = recorder.export_text()
        assert "Log File" in out
        assert len(controller._exit_errors) == 1


class TestPreExistingExitError:
    """A mid-run service failure must not also cost us the error table.

    When something populates ``_exit_errors`` before shutdown (a service failing
    a lifecycle command, or crashing mid-run), ``_stop_system_controller`` takes
    its else branch and never calls
    ``_print_post_benchmark_info_and_metrics``. Without an export on that branch
    a run whose requests *also* failed loses the table on exactly the runs
    carrying two kinds of failure at once.
    """

    @pytest.fixture
    def controller(
        self, system_controller: SystemController, mock_service_manager: AsyncMock
    ) -> SystemController:
        system_controller._profile_results = make_profile_results(
            records=[make_record()],
            successful=0,
            errors=100,
            error_summary=summary(),
        )
        system_controller._telemetry_results = None
        system_controller._server_metrics_results = None
        system_controller._exit_errors = [
            ExitErrorInfo(
                error_details=ErrorDetails(
                    type="SERVICE_ERROR", message="a service died mid-run"
                ),
                operation="service_runtime",
                service_id="worker_1",
            )
        ]
        system_controller.publish = AsyncMock()
        system_controller.service_manager = mock_service_manager
        system_controller.comms = AsyncMock()
        system_controller.proxy_manager = AsyncMock()
        system_controller.ui = AsyncMock()
        return system_controller

    async def _run_shutdown(
        self, controller: SystemController
    ) -> tuple[str, MagicMock]:
        recorder = Console(record=True, width=200)
        with (
            patch("aiperf.controller.system_controller.Console", return_value=recorder),
            patch(
                "aiperf.controller.system_controller.cleanup_global_log_queue",
                new_callable=AsyncMock,
            ),
            patch("aiperf.controller.system_controller.os._exit") as mock_exit,
        ):
            await controller._stop_system_controller()
        return recorder.export_text(), mock_exit

    async def test_error_table_printed_on_the_exit_error_branch(
        self, controller: SystemController
    ) -> None:
        """The regression: this branch printed no table at all."""
        out, _ = await self._run_shutdown(controller)

        assert "Error Summary" in out
        assert "404" in out

    async def test_exit_panel_and_log_path_still_print(
        self, controller: SystemController
    ) -> None:
        out, _ = await self._run_shutdown(controller)

        assert "Log File" in out
        assert "a service died mid-run" in out

    async def test_table_precedes_the_exit_panel(
        self, controller: SystemController
    ) -> None:
        """The exit message says "see the table above", so ordering is load-bearing."""
        out, _ = await self._run_shutdown(controller)

        # Assert presence first: ``str.find`` returns -1 when absent, so a bare
        # ordering comparison would pass vacuously if the table never rendered.
        assert "Error Summary" in out
        assert "Exit Errors" in out
        assert out.index("Error Summary") < out.index("Exit Errors")

    async def test_shutdown_still_reaches_os_exit(
        self, controller: SystemController
    ) -> None:
        """The stop hook must always terminate the process."""
        _, mock_exit = await self._run_shutdown(controller)

        mock_exit.assert_called_once()


class TestAbsentNestedResults:
    """``ProcessRecordsResult.results`` is declared required, but the records
    handler defends against it being absent, so the exporter must too.

    Reaching ``results.error_summary`` on an unset value raises
    ``AttributeError``. The surrounding guard catches it, but then logs a
    misleading "failed to render the error summary" line during shutdown.
    """

    @pytest.fixture
    def controller(self, system_controller: SystemController) -> SystemController:
        system_controller._profile_results = ProcessRecordsResult.model_construct(
            results=None
        )
        system_controller._telemetry_results = None
        system_controller._server_metrics_results = None
        return system_controller

    async def test_returns_quietly_when_nested_results_absent(
        self, controller: SystemController
    ) -> None:
        """No spurious rendering-failure log, and no exception escapes."""
        with patch.object(controller, "exception") as mock_exception:
            await controller._export_error_summary()

        mock_exception.assert_not_called()

    async def test_failure_while_inspecting_results_is_contained(
        self, controller: SystemController
    ) -> None:
        """Isolation must cover the guards, not only the rendering.

        The guards run inside the try for this reason: an exception raised
        while inspecting the results would otherwise escape and be swallowed by
        the caller's broad shutdown guard, taking the exit-error panel and the
        log-file path down with it.
        """
        exploding = MagicMock()
        type(exploding).results = PropertyMock(side_effect=RuntimeError("boom"))
        controller._profile_results = exploding

        await controller._export_error_summary()
