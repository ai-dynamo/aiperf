# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A service that fails during start-up must exit non-zero, not silently.

``AIPerfLifecycleMixin._fail()`` reports a start-up failure by raising
``asyncio.CancelledError`` -- a ``BaseException``, not an ``Exception``. The
lifecycle block in ``bootstrap_and_run_service`` guards with
``except Exception``, so that error escapes it, the
``contextlib.suppress(asyncio.CancelledError)`` around the event loop swallows
it, and ``_exit_if_service_failed`` never runs.

The worker then exits cleanly without reporting a service error and the
controller times out waiting for a registration that never comes: 35s and a
misleading "No workers registered with the credit router" instead of 1s and the
real cause.

This is the same root cause fixed for ``endpoint_signer`` in ``b7e1a8af5``, but
on the *default* path -- ``BaseTransport.__init__`` attaches the request signer
as a child lifecycle, so it runs on every benchmark, while ``endpoint_signer``
only runs with ``--reset-kv-cache`` or a readiness timeout.
"""

import asyncio

import pytest

from aiperf.common.bootstrap import bootstrap_and_run_service
from aiperf.common.enums import LifecycleState
from aiperf.common.hooks import on_init
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.config.flags.cli_config import CLIConfig
from tests.harness import mock_plugin
from tests.unit.common.conftest import DummyService
from tests.unit.conftest import make_run_from_cli


class _LifecycleFailingService(DummyService):
    """Fails start-up through the real mixin, not a simulation of it.

    Calling ``_fail`` rather than raising ``CancelledError`` directly is the
    point: a hand-rolled raise would not set ``FAILED`` or populate
    ``_exit_errors``, and the test could then pass against a fix that only
    happened to catch the exception type.
    """

    service_type = "test_lifecycle_failing"

    async def start(self):
        await self._fail(ValueError("No AWS credentials found"))


class TestServiceStartupFailureExitsNonZero:
    @pytest.fixture(autouse=True)
    def setup_bootstrap_mocks(
        self,
        mock_psutil_process,
        mock_setup_child_process_logging,
        register_dummy_services,
    ):
        pass

    def test_a_lifecycle_start_failure_exits_non_zero(
        self,
        service_config_no_uvloop: CLIConfig,
        cli_config: CLIConfig,
        mock_log_queue,
    ) -> None:
        run = make_run_from_cli(cli_config)

        with (
            mock_plugin(
                "service",
                "test_lifecycle_failing",
                _LifecycleFailingService,
                metadata={"required": False, "auto_start": False, "disable_gc": False},
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            bootstrap_and_run_service(
                "test_lifecycle_failing",
                run=run,
                log_queue=mock_log_queue,
                service_id="test_lifecycle_failing",
            )

        assert exc_info.value.code == 1

    def test_the_failure_is_recorded_on_the_service(
        self,
        service_config_no_uvloop: CLIConfig,
        cli_config: CLIConfig,
        mock_log_queue,
    ) -> None:
        """``_exit_if_service_failed`` keys off ``FAILED`` and ``_exit_errors``,
        both of which ``_fail`` populates before raising. Pinning them here
        keeps the test above honest: it must exit non-zero *because* the
        failure was seen, not because some unrelated error escaped."""
        run = make_run_from_cli(cli_config)
        captured: list[_LifecycleFailingService] = []

        class _Capturing(_LifecycleFailingService):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                captured.append(self)

        with (
            mock_plugin(
                "service",
                "test_lifecycle_failing",
                _Capturing,
                metadata={"required": False, "auto_start": False, "disable_gc": False},
            ),
            pytest.raises(SystemExit),
        ):
            bootstrap_and_run_service(
                "test_lifecycle_failing",
                run=run,
                log_queue=mock_log_queue,
                service_id="test_lifecycle_failing",
            )

        assert captured, "service was never constructed"
        service = captured[0]
        assert service.state == LifecycleState.FAILED
        assert service._exit_errors, "the underlying error was not recorded"
        assert "No AWS credentials found" in str(service._exit_errors[0])

    def test_genuine_cancellation_is_not_turned_into_a_failure_exit(
        self,
        service_config_no_uvloop: CLIConfig,
        cli_config: CLIConfig,
        mock_log_queue,
    ) -> None:
        """A cancel that is not a lifecycle failure must stay silent.

        Without this, "catch CancelledError" could be satisfied by treating
        every cancellation as a failed service, which would turn an ordinary
        Ctrl-C shutdown into a spurious non-zero exit.
        """

        class _CancelledButNotFailed(DummyService):
            service_type = "test_plain_cancel"

            async def start(self):
                raise asyncio.CancelledError("shutting down")

        run = make_run_from_cli(cli_config)

        with mock_plugin(
            "service",
            "test_plain_cancel",
            _CancelledButNotFailed,
            metadata={"required": False, "auto_start": False, "disable_gc": False},
        ):
            bootstrap_and_run_service(
                "test_plain_cancel",
                run=run,
                log_queue=mock_log_queue,
                service_id="test_plain_cancel",
            )


class _FailingChild(AIPerfLifecycleMixin):
    """Stands in for SigV4RequestSigner: a child lifecycle whose @on_init hook
    cannot complete. Fails through the real `_fail`, so it raises
    CancelledError exactly as the signer does."""

    @on_init
    async def _boom(self) -> None:
        raise ValueError("No AWS credentials found")


class _ServiceWithFailingChild(DummyService):
    """Mirrors BaseTransport.__init__, which attaches the request signer as a
    child lifecycle -- so this is the path every benchmark takes, not the
    --reset-kv-cache path that `endpoint_signer` covers."""

    service_type = "test_failing_child"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attach_child_lifecycle(_FailingChild())


class TestChildLifecycleFailureIsSurfaced:
    """The signer is a *child* of the transport, so its `_fail` sets the
    child's state, not the service's.

    The parent's own transition handler guards with `except Exception`, and the
    child raises CancelledError -- a BaseException -- so the parent never
    reaches `_fail` and never becomes FAILED. Everything upstream then sees a
    healthy service that simply stopped, and the process exits 0.
    """

    @pytest.fixture(autouse=True)
    def setup_bootstrap_mocks(
        self,
        mock_psutil_process,
        mock_setup_child_process_logging,
        register_dummy_services,
    ):
        pass

    def test_a_failing_child_lifecycle_fails_the_parent(
        self,
        service_config_no_uvloop: CLIConfig,
        cli_config: CLIConfig,
        mock_log_queue,
    ) -> None:
        run = make_run_from_cli(cli_config)
        captured: list[_ServiceWithFailingChild] = []

        class _Capturing(_ServiceWithFailingChild):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                captured.append(self)

        with (
            mock_plugin(
                "service",
                "test_failing_child",
                _Capturing,
                metadata={"required": False, "auto_start": False, "disable_gc": False},
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            bootstrap_and_run_service(
                "test_failing_child",
                run=run,
                log_queue=mock_log_queue,
                service_id="test_failing_child",
            )

        assert exc_info.value.code == 1
        assert captured[0].state == LifecycleState.FAILED
        assert "No AWS credentials found" in str(captured[0]._exit_errors)


class TestStartupFailureIsReportedToTheController:
    """A service that opts in publishes its start-up failure as SERVICE_ERROR
    before the lifecycle tears its comms down.

    Workers need this: they are not in ``required_services`` in multi-process
    mode, so a worker that dies before registering is otherwise invisible to
    the controller -- it waits out ``PhaseOrchestrator``'s 30s credit-router
    timeout and reports that instead of the worker's own error. Publishing lets
    the controller both see the real cause and decide whether any worker is
    left.

    Opt-in rather than universal: optional collectors (GPU telemetry, server
    metrics) failing at start-up must stay a degraded run, and every
    SERVICE_ERROR the controller receives lands in its exit errors.
    """

    @pytest.fixture(autouse=True)
    def setup_bootstrap_mocks(
        self,
        mock_psutil_process,
        mock_setup_child_process_logging,
        register_dummy_services,
    ):
        pass

    @staticmethod
    def _run_and_capture(service_cls, run, mock_log_queue) -> list:
        published: list = []

        class _Recording(service_cls):
            async def publish(self, message, *args, **kwargs):
                published.append(message)

        with (
            mock_plugin(
                "service",
                "test_reporting",
                _Recording,
                metadata={"required": False, "auto_start": False, "disable_gc": False},
            ),
            pytest.raises(SystemExit),
        ):
            bootstrap_and_run_service(
                "test_reporting",
                run=run,
                log_queue=mock_log_queue,
                service_id="test_reporting",
            )
        return published

    def test_an_opted_in_service_publishes_its_startup_failure(
        self,
        service_config_no_uvloop: CLIConfig,
        cli_config: CLIConfig,
        mock_log_queue,
    ) -> None:
        from aiperf.common.messages import BaseServiceErrorMessage

        class _Reporting(_ServiceWithFailingChild):
            reports_startup_failure = True

        published = self._run_and_capture(
            _Reporting, make_run_from_cli(cli_config), mock_log_queue
        )

        errors = [m for m in published if isinstance(m, BaseServiceErrorMessage)]
        assert len(errors) == 1, published
        assert errors[0].service_id == "test_reporting"
        # The child's underlying cause, not "Failed for <service>": this string
        # is what ends up as the headline in the exit-errors panel.
        assert "No AWS credentials found" in errors[0].error.message

    def test_a_service_that_has_not_opted_in_stays_silent(
        self,
        service_config_no_uvloop: CLIConfig,
        cli_config: CLIConfig,
        mock_log_queue,
    ) -> None:
        """The default must not change: an optional collector failing at
        start-up would otherwise land in the controller's exit errors and turn
        a degraded run into a failed one."""
        from aiperf.common.messages import BaseServiceErrorMessage

        published = self._run_and_capture(
            _ServiceWithFailingChild, make_run_from_cli(cli_config), mock_log_queue
        )

        assert not [m for m in published if isinstance(m, BaseServiceErrorMessage)]

    def test_workers_opt_in(self) -> None:
        """Pins the one production opt-in, so removing it is a visible change
        rather than a silent return to the 30s credit-router timeout."""
        from aiperf.workers.worker import Worker

        assert Worker.reports_startup_failure is True
