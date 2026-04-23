# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Internal setup helpers for ``GPUTelemetryManager``.

Split out of ``manager.py`` to keep the lifecycle class focused on hooks
and message handlers while isolating pure helpers, processor-factory
logic, and endpoint configuration routines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.common.environment import Environment
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.gpu_telemetry.constants import PYNVML_SOURCE_IDENTIFIER
from aiperf.gpu_telemetry.dcgm_collector import DCGMTelemetryCollector
from aiperf.gpu_telemetry.protocols import (
    GPUTelemetryAccumulatorProtocol,
    GPUTelemetryProcessorProtocol,
)
from aiperf.plugin import plugins
from aiperf.plugin.enums import (
    GPUTelemetryCollectorType,
    GPUTelemetryProcessorType,
    PluginType,
)

if TYPE_CHECKING:
    from aiperf.gpu_telemetry.manager import GPUTelemetryManager


def normalize_dcgm_url(url: str) -> str:
    """Ensure DCGM URL ends with /metrics endpoint.

    Args:
        url: Base URL or full metrics URL

    Returns:
        str: URL ending with /metrics
    """
    url = url.rstrip("/")
    if not url.endswith("/metrics"):
        url = f"{url}/metrics"
    return url


def compute_endpoints_for_display(
    reachable_defaults: list[str],
    user_provided_endpoints: list[str],
) -> list[str]:
    """Compute which DCGM endpoints should be displayed to the user.

    Filters endpoints for clean console output based on user configuration
    and reachability. This intentional filtering prevents cluttering the UI
    with unreachable default endpoints that the user didn't explicitly configure.

    Args:
        reachable_defaults: List of default DCGM endpoints that are reachable
        user_provided_endpoints: List of user-configured endpoints (excluding defaults)

    Returns:
        List of endpoint URLs to display in console/export output:
        - reachable_defaults if any defaults are reachable
        - user_provided_endpoints + reachable_defaults if custom endpoints and defaults reachable
        - user_provided_endpoints if user configured but no defaults reachable
        - Empty list if no reachable defaults and user did not configure telemetry
    """
    if reachable_defaults and user_provided_endpoints:
        return list(user_provided_endpoints) + reachable_defaults
    elif reachable_defaults:
        return reachable_defaults
    elif user_provided_endpoints:
        return user_provided_endpoints
    return []


def create_processors(
    manager: GPUTelemetryManager,
) -> tuple[
    list[GPUTelemetryProcessorProtocol],
    GPUTelemetryAccumulatorProtocol | None,
]:
    """Instantiate all registered GPU telemetry processors for ``manager``.

    Returns the list of created processors plus the accumulator instance
    (or None if no accumulator plugin is registered). Failures to
    instantiate an individual processor are logged and skipped so one bad
    plugin cannot disable telemetry entirely.
    """
    processors: list[GPUTelemetryProcessorProtocol] = []
    accumulator: GPUTelemetryAccumulatorProtocol | None = None

    for entry in plugins.iter_entries(PluginType.GPU_TELEMETRY_PROCESSOR):
        try:
            ProcessorClass = plugins.get_class(
                PluginType.GPU_TELEMETRY_PROCESSOR, entry.name
            )
            processor = ProcessorClass(
                service_id=manager.service_id,
                run=manager.run,
                pub_client=manager.pub_client,
            )
            manager.attach_child_lifecycle(processor)
            processors.append(processor)
            if entry.name == GPUTelemetryProcessorType.GPU_TELEMETRY_ACCUMULATOR:
                accumulator = processor
            manager.debug(
                f"Created GPU telemetry processor: {entry.name}: "
                f"{processor.__class__.__name__}"
            )
        except PostProcessorDisabled:
            manager.debug(
                f"GPU telemetry processor {entry.name} is disabled and will not be used"
            )
        except Exception as e:  # noqa: BLE001 - per-plugin; skip bad processor and continue
            manager.error(f"Failed to create GPU telemetry processor {entry.name}: {e}")

    return processors, accumulator


async def configure_pynvml_collector(manager: GPUTelemetryManager) -> None:
    """Configure a single PyNVML collector for local GPU monitoring."""
    manager.debug("GPU Telemetry: Configuring pynvml collector")

    try:
        CollectorClass = plugins.get_class(
            PluginType.GPU_TELEMETRY_COLLECTOR,
            GPUTelemetryCollectorType.PYNVML,
        )

        collector_id = "pynvml_collector"
        collector = CollectorClass(
            collection_interval=manager._collection_interval,
            record_callback=manager._on_telemetry_records,
            error_callback=manager._on_telemetry_error,
            collector_id=collector_id,
        )

        is_available = await collector.is_url_reachable()
        if is_available:
            manager._collectors[PYNVML_SOURCE_IDENTIFIER] = collector
            manager.debug("GPU Telemetry: pynvml collector configured successfully")
            await manager._send_telemetry_status(
                enabled=True,
                reason=None,
                endpoints_configured=[PYNVML_SOURCE_IDENTIFIER],
                endpoints_reachable=[PYNVML_SOURCE_IDENTIFIER],
            )
        else:
            manager.warning("GPU Telemetry: pynvml not available or no GPUs found")
            await manager._send_telemetry_status(
                enabled=False,
                reason="pynvml not available or no GPUs found",
                endpoints_configured=[PYNVML_SOURCE_IDENTIFIER],
                endpoints_reachable=[],
            )
    except RuntimeError as e:
        # pynvml package not installed
        manager.error(f"GPU Telemetry: {e}")
        await manager._send_telemetry_status(
            enabled=False,
            reason=str(e),
            endpoints_configured=[],
            endpoints_reachable=[],
        )
    except Exception as e:  # noqa: BLE001 - fault-tolerant telemetry
        manager.error(f"GPU Telemetry: Failed to configure pynvml collector: {e}")
        await manager._send_telemetry_status(
            enabled=False,
            reason=f"pynvml configuration failed: {e}",
            endpoints_configured=[],
            endpoints_reachable=[],
        )


async def configure_dcgm_collectors(manager: GPUTelemetryManager) -> None:
    """Configure DCGM collectors for HTTP-based GPU telemetry."""
    for dcgm_url in manager._dcgm_endpoints:
        manager.debug(f"GPU Telemetry: Testing reachability of {dcgm_url}")
        collector_id = f"collector_{dcgm_url.replace(':', '_').replace('/', '_')}"
        collector = DCGMTelemetryCollector(
            dcgm_url=dcgm_url,
            collection_interval=manager._collection_interval,
            record_callback=manager._on_telemetry_records,
            error_callback=manager._on_telemetry_error,
            collector_id=collector_id,
        )

        try:
            is_reachable = await collector.is_url_reachable()
            if is_reachable:
                manager._collectors[dcgm_url] = collector
                manager.debug(f"GPU Telemetry: DCGM endpoint {dcgm_url} is reachable")
            else:
                manager.debug(
                    f"GPU Telemetry: DCGM endpoint {dcgm_url} is not reachable"
                )
        except Exception as e:  # noqa: BLE001 - per-endpoint; skip unreachable and continue
            manager.error(f"GPU Telemetry: Exception testing {dcgm_url}: {e}")

    # Determine which defaults are reachable for display filtering
    reachable_endpoints = list(manager._collectors.keys())
    reachable_defaults = [
        ep for ep in Environment.GPU.DEFAULT_DCGM_ENDPOINTS if ep in reachable_endpoints
    ]
    endpoints_for_display = compute_endpoints_for_display(
        reachable_defaults, manager._user_provided_endpoints
    )

    if not manager._collectors:
        # Telemetry manager shutdown occurs in _on_start_profiling to prevent hang
        await manager._send_telemetry_status(
            enabled=False,
            reason="no DCGM endpoints reachable",
            endpoints_configured=endpoints_for_display,
            endpoints_reachable=[],
        )
        return

    # Phase 2: Capture baseline metrics before profiling starts
    manager.info("GPU Telemetry: Capturing baseline metrics...")
    for dcgm_url, collector in manager._collectors.items():
        try:
            await collector.initialize()
            await collector.collect_and_process_metrics()
            manager.debug(f"GPU Telemetry: Captured baseline from {dcgm_url}")
        except Exception as e:  # noqa: BLE001 - per-endpoint; skip baseline failure and continue
            manager.warning(
                f"GPU Telemetry: Failed to capture baseline from {dcgm_url}: {e}"
            )

    await manager._send_telemetry_status(
        enabled=True,
        reason=None,
        endpoints_configured=endpoints_for_display,
        endpoints_reachable=reachable_endpoints,
    )
