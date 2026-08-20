# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SystemController's required-service set and proxy ownership in Kubernetes mode.

Both behaviors are cluster-only contracts that unit tests of the
multiprocessing path cannot observe.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.enums import MessageType
from aiperf.controller.system_controller import SystemController
from aiperf.plugin.enums import ServiceRunType, ServiceType


def _build_controller(benchmark_run, run_type: ServiceRunType) -> SystemController:
    """Construct a SystemController with every external dependency mocked."""
    benchmark_run.cfg.runtime.service_run_type = run_type

    def mock_get_class(protocol, name):
        return lambda **kwargs: AsyncMock()

    with (
        patch(
            "aiperf.controller.system_controller.plugins.get_class",
            side_effect=mock_get_class,
        ),
        patch("aiperf.controller.system_controller.ProxyManager") as mock_proxy,
        patch(
            "aiperf.common.mixins.communication_mixin.plugins.get_class",
            side_effect=mock_get_class,
        ),
    ):  # fmt: skip
        mock_proxy.return_value = AsyncMock()
        controller = SystemController(run=benchmark_run, service_id="test_controller")
        controller._proxy_manager_call = mock_proxy.call_args
        return controller


def test_kubernetes_requires_worker_group_manager_not_worker_manager(
    benchmark_run,
) -> None:
    """No container runs a WorkerManager in Kubernetes; requiring one deadlocks startup.

    Regression: the controller waited out the full registration timeout for
    ``worker_manager`` and then killed the control plane, even though every
    real service (including both WorkerGroupManagers) had registered.
    """
    controller = _build_controller(benchmark_run, ServiceRunType.KUBERNETES)

    assert ServiceType.WORKER_MANAGER not in controller.required_services
    assert controller.required_services[ServiceType.WORKER_GROUP_MANAGER] == 1
    assert controller.required_services[ServiceType.WORKER] == 1


def test_multiprocessing_still_requires_worker_manager(benchmark_run) -> None:
    """The non-Kubernetes required-service set must be untouched."""
    controller = _build_controller(benchmark_run, ServiceRunType.MULTIPROCESSING)

    assert controller.required_services[ServiceType.WORKER_MANAGER] == 1
    assert ServiceType.WORKER_GROUP_MANAGER not in controller.required_services
    assert ServiceType.WORKER not in controller.required_services


@pytest.mark.parametrize("sidecar_enabled", [True, False])
def test_event_bus_proxy_skipped_when_sidecar_owns_it(
    benchmark_run, sidecar_enabled: bool
) -> None:
    """The controller must not re-bind an event bus a sidecar container already binds."""
    from aiperf.kubernetes.environment import K8sEnvironment

    original = K8sEnvironment.EVENT_BUS_SIDECAR_ENABLED
    K8sEnvironment.EVENT_BUS_SIDECAR_ENABLED = sidecar_enabled
    try:
        controller = _build_controller(benchmark_run, ServiceRunType.KUBERNETES)
    finally:
        K8sEnvironment.EVENT_BUS_SIDECAR_ENABLED = original

    _, kwargs = controller._proxy_manager_call
    assert kwargs["enable_event_bus"] is not sidecar_enabled


def test_multiprocessing_always_hosts_the_event_bus_proxy(benchmark_run) -> None:
    """The multiprocessing path keeps hosting all three proxies unconditionally."""
    from aiperf.kubernetes.environment import K8sEnvironment

    original = K8sEnvironment.EVENT_BUS_SIDECAR_ENABLED
    K8sEnvironment.EVENT_BUS_SIDECAR_ENABLED = True
    try:
        controller = _build_controller(benchmark_run, ServiceRunType.MULTIPROCESSING)
    finally:
        K8sEnvironment.EVENT_BUS_SIDECAR_ENABLED = original

    _, kwargs = controller._proxy_manager_call
    assert kwargs["enable_event_bus"] is True


@pytest.mark.asyncio
async def test_kubernetes_export_writes_ready_marker_and_publishes(
    benchmark_run, tmp_path
) -> None:
    """Regression: results sat on disk and the AIPerfJob never left Initializing.

    The results sidecar refuses to serve top-level artifacts until
    ``.aiperf_results_ready.json`` exists, and ProgressRouter only reports
    is_complete after ResultsExportedMessage. Neither was ever produced.
    """
    controller = _build_controller(benchmark_run, ServiceRunType.KUBERNETES)
    controller.run.cfg.artifacts.dir = tmp_path
    controller.publish = AsyncMock()

    with patch(
        "aiperf.kubernetes.completion_signal.signal_benchmark_complete",
        new_callable=AsyncMock,
    ) as signal_complete:
        await controller._announce_results_exported()

    assert (tmp_path / ".aiperf_results_ready.json").exists()
    published = controller.publish.await_args.args[0]
    assert published.message_type == MessageType.RESULTS_EXPORTED
    signal_complete.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_kubernetes_export_runtime_begins_processing_before_file_export(
    benchmark_run, tmp_path
) -> None:
    """The controller must close stale readiness before any exporter mutates files."""
    controller = _build_controller(benchmark_run, ServiceRunType.KUBERNETES)
    controller.run.cfg.artifacts.dir = tmp_path
    controller._profile_results = MagicMock()
    controller._profile_results.results.records = [MagicMock()]
    controller._profile_results.results.successful_request_count = 1
    controller._profile_results.results.error_request_count = 0
    controller._inject_accuracy_results_into_records = MagicMock()
    controller._surface_export_failures = MagicMock(return_value=False)
    controller._print_cli_command = MagicMock()
    controller._print_benchmark_duration = MagicMock()
    controller._print_exported_file_infos = MagicMock()
    controller._print_log_file_info = MagicMock()
    order: list[str] = []

    exporter_manager = MagicMock()

    async def export_data() -> list[object]:
        order.append("export")
        return []

    exporter_manager.export_data = AsyncMock(side_effect=export_data)
    exporter_manager.export_console = AsyncMock()
    controller._run_kubernetes_auto_plot = AsyncMock(
        side_effect=lambda: order.append("plot")
    )
    controller._announce_results_exported = AsyncMock(
        side_effect=lambda: order.append("announce")
    )
    console = MagicMock()
    console.width = 120

    with (
        patch(
            "aiperf.controller.system_controller.write_processing_marker",
            side_effect=lambda _path: order.append("processing"),
        ),
        patch(
            "aiperf.controller.system_controller.ExporterManager",
            return_value=exporter_manager,
        ),
        patch("aiperf.controller.system_controller.Console", return_value=console),
    ):
        await controller._print_post_benchmark_info_and_metrics()

    assert order == ["processing", "export", "plot", "announce"]


@pytest.mark.asyncio
async def test_kubernetes_initialization_closes_stale_readiness_before_services_start(
    benchmark_run, tmp_path
) -> None:
    """A restarted controller must hide the prior attempt before doing new work."""
    controller = _build_controller(benchmark_run, ServiceRunType.KUBERNETES)
    controller.run.cfg.artifacts.dir = tmp_path
    controller.setup_signal_handlers = MagicMock()
    controller.service_manager.initialize = AsyncMock()
    stale_ready = tmp_path / ".aiperf_results_ready.json"
    stale_ready.write_text('{"ready":true}')

    await controller._initialize_system_controller()

    assert not stale_ready.exists()
    assert (tmp_path / ".aiperf_results_processing.json").is_file()
    controller.service_manager.initialize.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_kubernetes_failed_reentry_closes_stale_readiness_before_validation(
    benchmark_run, tmp_path
) -> None:
    """A failed second export attempt must not expose an earlier attempt's files."""
    controller = _build_controller(benchmark_run, ServiceRunType.KUBERNETES)
    controller.run.cfg.artifacts.dir = tmp_path
    controller._profile_results = None
    stale_ready = tmp_path / ".aiperf_results_ready.json"
    stale_ready.write_text('{"ready":true}')

    await controller._print_post_benchmark_info_and_metrics()

    assert not stale_ready.exists()
    assert (tmp_path / ".aiperf_results_processing.json").is_file()
    # Reporting is centralized in ``_stop_system_controller`` so a degraded run
    # can export AND print its errors; this path only records them.
    assert [e.operation for e in controller._exit_errors] == ["export_results"]


@pytest.mark.asyncio
async def test_kubernetes_completion_notification_follows_durable_readiness(
    benchmark_run, tmp_path
) -> None:
    """The operator must not harvest until the marker and local API state commit."""
    controller = _build_controller(benchmark_run, ServiceRunType.KUBERNETES)
    controller.run.cfg.artifacts.dir = tmp_path
    order: list[str] = []

    assert await controller._begin_results_export_transaction() is True
    assert (tmp_path / ".aiperf_results_processing.json").is_file()

    async def publish(_message: object) -> None:
        assert (tmp_path / ".aiperf_results_ready.json").is_file()
        assert not (tmp_path / ".aiperf_results_processing.json").exists()
        order.append("publish")

    controller.publish = AsyncMock(side_effect=publish)

    async def signal_complete() -> bool:
        assert (tmp_path / ".aiperf_results_ready.json").is_file()
        assert not (tmp_path / ".aiperf_results_processing.json").exists()
        order.append("operator")
        return True

    with patch(
        "aiperf.kubernetes.completion_signal.signal_benchmark_complete",
        side_effect=signal_complete,
    ):
        await controller._announce_results_exported()

    assert order == ["publish", "operator"]


@pytest.mark.asyncio
async def test_kubernetes_processing_marker_failure_blocks_export_transaction(
    benchmark_run, tmp_path
) -> None:
    controller = _build_controller(benchmark_run, ServiceRunType.KUBERNETES)
    controller.run.cfg.artifacts.dir = tmp_path

    with patch(
        "aiperf.controller.system_controller.write_processing_marker",
        side_effect=OSError("read-only volume"),
    ):
        started = await controller._begin_results_export_transaction()

    assert started is False
    assert controller._export_failed is True
    assert controller._exit_errors[-1].operation == "export:ResultsProcessingMarker"


@pytest.mark.asyncio
async def test_multiprocessing_export_writes_no_marker(benchmark_run, tmp_path) -> None:
    """The local path has no sidecar to hand off to."""
    controller = _build_controller(benchmark_run, ServiceRunType.MULTIPROCESSING)
    controller.run.cfg.artifacts.dir = tmp_path
    controller.publish = AsyncMock()

    with patch(
        "aiperf.kubernetes.completion_signal.signal_benchmark_complete",
        new_callable=AsyncMock,
    ) as signal_complete:
        await controller._announce_results_exported()

    assert not (tmp_path / ".aiperf_results_ready.json").exists()
    signal_complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_kubernetes_local_export_failure_withholds_ready_marker(
    benchmark_run, tmp_path
) -> None:
    """A failed local writer cannot expose a partial artifact set as ready."""
    controller = _build_controller(benchmark_run, ServiceRunType.KUBERNETES)
    controller.run.cfg.artifacts.dir = tmp_path
    controller._export_failed = True
    controller.publish = AsyncMock()

    with patch(
        "aiperf.kubernetes.completion_signal.signal_benchmark_complete",
        new_callable=AsyncMock,
    ) as signal_complete:
        await controller._announce_results_exported()

    assert not (tmp_path / ".aiperf_results_ready.json").exists()
    controller.publish.assert_not_awaited()
    signal_complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_kubernetes_ready_marker_failure_blocks_completion_notification(
    benchmark_run, tmp_path
) -> None:
    """The durable marker must commit before completion is announced."""
    controller = _build_controller(benchmark_run, ServiceRunType.KUBERNETES)
    controller.run.cfg.artifacts.dir = tmp_path
    controller.publish = AsyncMock()

    with (
        patch(
            "aiperf.controller.system_controller.write_ready_marker",
            side_effect=OSError("disk full"),
        ),
        patch(
            "aiperf.kubernetes.completion_signal.signal_benchmark_complete",
            new_callable=AsyncMock,
        ) as signal_complete,
    ):
        await controller._announce_results_exported()

    controller.publish.assert_not_awaited()
    signal_complete.assert_not_awaited()
    assert controller._exit_errors[-1].operation == "export:ResultsReadyMarker"


@pytest.mark.asyncio
async def test_export_survives_cancelled_error_from_stopped_bus(
    benchmark_run, tmp_path
) -> None:
    """Regression: the AIPerfJob hung in Running forever after a clean benchmark.

    The stop hook now exports before ``comms.stop()``, but a concurrent bus
    failure can still raise ``asyncio.CancelledError``. The durable marker is
    authoritative and must survive that best-effort live notification failure.
    """
    controller = _build_controller(benchmark_run, ServiceRunType.KUBERNETES)
    controller.run.cfg.artifacts.dir = tmp_path
    controller.publish = AsyncMock(side_effect=asyncio.CancelledError("Socket stopped"))

    with patch(
        "aiperf.kubernetes.completion_signal.signal_benchmark_complete",
        new_callable=AsyncMock,
    ) as signal_complete:
        await controller._announce_results_exported()

    assert (tmp_path / ".aiperf_results_ready.json").exists()
    signal_complete.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_kubernetes_auto_plot_runs_before_readiness(
    benchmark_run, tmp_path
) -> None:
    controller = _build_controller(benchmark_run, ServiceRunType.KUBERNETES)
    controller.run.cfg.artifacts.dir = tmp_path
    controller.run.cfg.artifacts.auto_plot = True
    controller.run.plot = None

    with patch(
        "aiperf.plot.auto_plot.run_auto_plot_async", new_callable=AsyncMock
    ) as auto_plot:
        await controller._run_kubernetes_auto_plot()

    auto_plot.assert_awaited_once_with(
        artifact_dir=tmp_path,
        plot_required=False,
        plot_envelope=None,
    )
    assert not (tmp_path / ".aiperf_results_ready.json").exists()


@pytest.mark.asyncio
async def test_kubernetes_required_auto_plot_failure_withholds_readiness(
    benchmark_run, tmp_path
) -> None:
    controller = _build_controller(benchmark_run, ServiceRunType.KUBERNETES)
    controller.run.cfg.artifacts.dir = tmp_path
    controller.run.cfg.artifacts.auto_plot = True
    controller.run.cfg.artifacts.plot_required = True
    controller.publish = AsyncMock()

    with patch(
        "aiperf.plot.auto_plot.run_auto_plot_async",
        new=AsyncMock(side_effect=RuntimeError("plot failed")),
    ):
        await controller._run_kubernetes_auto_plot()
    with patch(
        "aiperf.kubernetes.completion_signal.signal_benchmark_complete",
        new_callable=AsyncMock,
    ) as signal_complete:
        await controller._announce_results_exported()

    assert controller._export_failed is True
    assert controller._exit_errors[-1].operation == "auto_plot"
    assert not (tmp_path / ".aiperf_results_ready.json").exists()
    controller.publish.assert_not_awaited()
    signal_complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_multiprocessing_does_not_run_controller_auto_plot(
    benchmark_run, tmp_path
) -> None:
    controller = _build_controller(benchmark_run, ServiceRunType.MULTIPROCESSING)
    controller.run.cfg.artifacts.dir = tmp_path
    controller.run.cfg.artifacts.auto_plot = True

    with patch(
        "aiperf.plot.auto_plot.run_auto_plot_async", new_callable=AsyncMock
    ) as auto_plot:
        await controller._run_kubernetes_auto_plot()

    auto_plot.assert_not_awaited()
