# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plugin registry with lazy loading and priority-based conflict resolution.

Conflict resolution: higher priority wins; equal priority: external beats built-in.

The heavy lifting lives in two siblings:
  - `_registry.py` — the `_PluginRegistry` class (state, discovery, lookup).
  - `metadata.py`  — typed metadata helper functions.

This module is the public facade: it owns the singleton `_registry` instance,
exposes module-level functions as stable attribute names, and hosts the
auto-generated TYPE_CHECKING `@overload` stubs (see generate_plugin_artifacts.py).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, TypeAlias

# Re-export the registry class so `from aiperf.plugin.plugins import _PluginRegistry`
# keeps working for tests and direct consumers.
from aiperf.plugin._registry import _PluginRegistry
from aiperf.plugin.metadata import (
    get_dataset_loader_metadata,
    get_endpoint_metadata,
    get_metadata,
    get_plot_metadata,
    get_public_dataset_loader_metadata,
    get_service_metadata,
    get_transport_metadata,
    get_typed_metadata,
    is_trace_dataset,
)
from aiperf.plugin.types import PluginEntry

if TYPE_CHECKING:
    from aiperf.plugin.enums import PluginType, PluginTypeStr

    CategoryT: TypeAlias = PluginType | PluginTypeStr
else:
    CategoryT: TypeAlias = str


# Re-export metadata helpers to keep `plugins.get_endpoint_metadata` etc. accessible.
# (also used by tests that patch `aiperf.plugin.plugins.get_endpoint_metadata`)
__all__ = [
    "_PluginRegistry",
    "create_enum",
    "find_registered_name",
    "get_category_metadata",
    "get_class",
    "get_dataset_loader_metadata",
    "get_endpoint_metadata",
    "get_entry",
    "get_metadata",
    "get_package_metadata",
    "get_plot_metadata",
    "get_public_dataset_loader_metadata",
    "get_service_metadata",
    "get_transport_metadata",
    "get_typed_metadata",
    "has_entry",
    "is_internal_category",
    "is_trace_dataset",
    "iter_all",
    "iter_entries",
    "list_categories",
    "list_entries",
    "list_packages",
    "load_manifest",
    "register",
    "reset_registry",
    "unregister",
    "validate_all",
]


# ==============================================================================
# Overloaded functions
# ==============================================================================
if TYPE_CHECKING:
    # <generated-imports>
    # fmt: off
    # ruff: noqa: I001
    from aiperf.accuracy.protocols import AccuracyBenchmarkProtocol, AccuracyGraderProtocol
    from aiperf.api.routers.base_router import BaseRouter
    from aiperf.common.protocols import CommunicationClientProtocol, CommunicationProtocol, ServiceProtocol
    from aiperf.controller.protocols import ServiceManagerProtocol
    from aiperf.dataset.composer.base import BaseDatasetComposer
    from aiperf.dataset.protocols import CustomDatasetLoaderProtocol, DatasetBackingStoreProtocol, DatasetClientStoreProtocol, DatasetSamplingStrategyProtocol, PublicDatasetLoaderProtocol
    from aiperf.endpoints.protocols import EndpointProtocol
    from aiperf.exporters.protocols import ConsoleExporterProtocol, DataExporterProtocol
    from aiperf.gpu_telemetry.protocols import GPUTelemetryCollectorProtocol
    from aiperf.plot.core.plot_type_handlers import PlotTypeHandlerProtocol
    from aiperf.plugin.enums import APIRouterType, AccuracyBenchmarkType, AccuracyGraderType, ArrivalPattern, CommClientType, CommunicationBackend, ComposerType, ConsoleExporterType, CustomDatasetType, DataExporterType, DatasetBackingStoreType, DatasetClientStoreType, DatasetSamplingStrategy, EndpointType, GPUTelemetryCollectorType, GPUTelemetryProcessorType, PlotType, PluginType, PluginTypeStr, PublicDatasetType, RampType, RecordProcessorType, ResultsProcessorType, ServerMetricsProcessorType, ServiceRunType, ServiceType, TimingMode, TransportType, UIType, URLSelectionStrategy, ZMQProxyType
    from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor
    from aiperf.post_processors.protocols import RecordProcessorProtocol
    from aiperf.timing.intervals import IntervalGeneratorProtocol
    from aiperf.timing.ramping import RampStrategyProtocol
    from aiperf.timing.strategies.core import TimingStrategyProtocol
    from aiperf.timing.url_samplers import URLSelectionStrategyProtocol
    from aiperf.transports.base_transports import TransportProtocol
    from aiperf.ui.protocols import AIPerfUIProtocol
    from aiperf.zmq.zmq_proxy_base import BaseZMQProxy
    from typing import Literal, overload
    # </generated-imports>
    # <generated-overloads>
    @overload
    def get_class(category: Literal[PluginType.API_ROUTER, "api_router"], name_or_class_path: APIRouterType | str) -> type[BaseRouter]: ...
    @overload
    def iter_all(category: Literal[PluginType.API_ROUTER, "api_router"]) -> Iterator[tuple[PluginEntry, type[BaseRouter]]]: ...
    @overload
    def get_class(category: Literal[PluginType.TIMING_STRATEGY, "timing_strategy"], name_or_class_path: TimingMode | str) -> type[TimingStrategyProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.TIMING_STRATEGY, "timing_strategy"]) -> Iterator[tuple[PluginEntry, type[TimingStrategyProtocol]]]: ...
    @overload
    def get_class(category: Literal[PluginType.ARRIVAL_PATTERN, "arrival_pattern"], name_or_class_path: ArrivalPattern | str) -> type[IntervalGeneratorProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.ARRIVAL_PATTERN, "arrival_pattern"]) -> Iterator[tuple[PluginEntry, type[IntervalGeneratorProtocol]]]: ...
    @overload
    def get_class(category: Literal[PluginType.RAMP, "ramp"], name_or_class_path: RampType | str) -> type[RampStrategyProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.RAMP, "ramp"]) -> Iterator[tuple[PluginEntry, type[RampStrategyProtocol]]]: ...
    @overload
    def get_class(category: Literal[PluginType.DATASET_BACKING_STORE, "dataset_backing_store"], name_or_class_path: DatasetBackingStoreType | str) -> type[DatasetBackingStoreProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.DATASET_BACKING_STORE, "dataset_backing_store"]) -> Iterator[tuple[PluginEntry, type[DatasetBackingStoreProtocol]]]: ...
    @overload
    def get_class(category: Literal[PluginType.DATASET_CLIENT_STORE, "dataset_client_store"], name_or_class_path: DatasetClientStoreType | str) -> type[DatasetClientStoreProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.DATASET_CLIENT_STORE, "dataset_client_store"]) -> Iterator[tuple[PluginEntry, type[DatasetClientStoreProtocol]]]: ...
    @overload
    def get_class(category: Literal[PluginType.DATASET_SAMPLER, "dataset_sampler"], name_or_class_path: DatasetSamplingStrategy | str) -> type[DatasetSamplingStrategyProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.DATASET_SAMPLER, "dataset_sampler"]) -> Iterator[tuple[PluginEntry, type[DatasetSamplingStrategyProtocol]]]: ...
    @overload
    def get_class(category: Literal[PluginType.DATASET_COMPOSER, "dataset_composer"], name_or_class_path: ComposerType | str) -> type[BaseDatasetComposer]: ...
    @overload
    def iter_all(category: Literal[PluginType.DATASET_COMPOSER, "dataset_composer"]) -> Iterator[tuple[PluginEntry, type[BaseDatasetComposer]]]: ...
    @overload
    def get_class(category: Literal[PluginType.CUSTOM_DATASET_LOADER, "custom_dataset_loader"], name_or_class_path: CustomDatasetType | str) -> type[CustomDatasetLoaderProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.CUSTOM_DATASET_LOADER, "custom_dataset_loader"]) -> Iterator[tuple[PluginEntry, type[CustomDatasetLoaderProtocol]]]: ...
    @overload
    def get_class(category: Literal[PluginType.PUBLIC_DATASET_LOADER, "public_dataset_loader"], name_or_class_path: PublicDatasetType | str) -> type[PublicDatasetLoaderProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.PUBLIC_DATASET_LOADER, "public_dataset_loader"]) -> Iterator[tuple[PluginEntry, type[PublicDatasetLoaderProtocol]]]: ...
    @overload
    def get_class(category: Literal[PluginType.ENDPOINT, "endpoint"], name_or_class_path: EndpointType | str) -> type[EndpointProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.ENDPOINT, "endpoint"]) -> Iterator[tuple[PluginEntry, type[EndpointProtocol]]]: ...
    @overload
    def get_class(category: Literal[PluginType.TRANSPORT, "transport"], name_or_class_path: TransportType | str) -> type[TransportProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.TRANSPORT, "transport"]) -> Iterator[tuple[PluginEntry, type[TransportProtocol]]]: ...
    @overload
    def get_class(category: Literal[PluginType.RECORD_PROCESSOR, "record_processor"], name_or_class_path: RecordProcessorType | str) -> type[RecordProcessorProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.RECORD_PROCESSOR, "record_processor"]) -> Iterator[tuple[PluginEntry, type[RecordProcessorProtocol]]]: ...
    @overload
    def get_class(category: Literal[PluginType.RESULTS_PROCESSOR, "results_processor"], name_or_class_path: ResultsProcessorType | str) -> type[BaseMetricsProcessor]: ...
    @overload
    def iter_all(category: Literal[PluginType.RESULTS_PROCESSOR, "results_processor"]) -> Iterator[tuple[PluginEntry, type[BaseMetricsProcessor]]]: ...
    @overload
    def get_class(category: Literal[PluginType.GPU_TELEMETRY_PROCESSOR, "gpu_telemetry_processor"], name_or_class_path: GPUTelemetryProcessorType | str) -> type[BaseMetricsProcessor]: ...
    @overload
    def iter_all(category: Literal[PluginType.GPU_TELEMETRY_PROCESSOR, "gpu_telemetry_processor"]) -> Iterator[tuple[PluginEntry, type[BaseMetricsProcessor]]]: ...
    @overload
    def get_class(category: Literal[PluginType.SERVER_METRICS_PROCESSOR, "server_metrics_processor"], name_or_class_path: ServerMetricsProcessorType | str) -> type[BaseMetricsProcessor]: ...
    @overload
    def iter_all(category: Literal[PluginType.SERVER_METRICS_PROCESSOR, "server_metrics_processor"]) -> Iterator[tuple[PluginEntry, type[BaseMetricsProcessor]]]: ...
    @overload
    def get_class(category: Literal[PluginType.ACCURACY_GRADER, "accuracy_grader"], name_or_class_path: AccuracyGraderType | str) -> type[AccuracyGraderProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.ACCURACY_GRADER, "accuracy_grader"]) -> Iterator[tuple[PluginEntry, type[AccuracyGraderProtocol]]]: ...
    @overload
    def get_class(category: Literal[PluginType.ACCURACY_BENCHMARK, "accuracy_benchmark"], name_or_class_path: AccuracyBenchmarkType | str) -> type[AccuracyBenchmarkProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.ACCURACY_BENCHMARK, "accuracy_benchmark"]) -> Iterator[tuple[PluginEntry, type[AccuracyBenchmarkProtocol]]]: ...
    @overload
    def get_class(category: Literal[PluginType.DATA_EXPORTER, "data_exporter"], name_or_class_path: DataExporterType | str) -> type[DataExporterProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.DATA_EXPORTER, "data_exporter"]) -> Iterator[tuple[PluginEntry, type[DataExporterProtocol]]]: ...
    @overload
    def get_class(category: Literal[PluginType.CONSOLE_EXPORTER, "console_exporter"], name_or_class_path: ConsoleExporterType | str) -> type[ConsoleExporterProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.CONSOLE_EXPORTER, "console_exporter"]) -> Iterator[tuple[PluginEntry, type[ConsoleExporterProtocol]]]: ...
    @overload
    def get_class(category: Literal[PluginType.UI, "ui"], name_or_class_path: UIType | str) -> type[AIPerfUIProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.UI, "ui"]) -> Iterator[tuple[PluginEntry, type[AIPerfUIProtocol]]]: ...
    @overload
    def get_class(category: Literal[PluginType.URL_SELECTION_STRATEGY, "url_selection_strategy"], name_or_class_path: URLSelectionStrategy | str) -> type[URLSelectionStrategyProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.URL_SELECTION_STRATEGY, "url_selection_strategy"]) -> Iterator[tuple[PluginEntry, type[URLSelectionStrategyProtocol]]]: ...
    @overload
    def get_class(category: Literal[PluginType.SERVICE, "service"], name_or_class_path: ServiceType | str) -> type[ServiceProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.SERVICE, "service"]) -> Iterator[tuple[PluginEntry, type[ServiceProtocol]]]: ...
    @overload
    def get_class(category: Literal[PluginType.SERVICE_MANAGER, "service_manager"], name_or_class_path: ServiceRunType | str) -> type[ServiceManagerProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.SERVICE_MANAGER, "service_manager"]) -> Iterator[tuple[PluginEntry, type[ServiceManagerProtocol]]]: ...
    @overload
    def get_class(category: Literal[PluginType.COMMUNICATION, "communication"], name_or_class_path: CommunicationBackend | str) -> type[CommunicationProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.COMMUNICATION, "communication"]) -> Iterator[tuple[PluginEntry, type[CommunicationProtocol]]]: ...
    @overload
    def get_class(category: Literal[PluginType.COMMUNICATION_CLIENT, "communication_client"], name_or_class_path: CommClientType | str) -> type[CommunicationClientProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.COMMUNICATION_CLIENT, "communication_client"]) -> Iterator[tuple[PluginEntry, type[CommunicationClientProtocol]]]: ...
    @overload
    def get_class(category: Literal[PluginType.ZMQ_PROXY, "zmq_proxy"], name_or_class_path: ZMQProxyType | str) -> type[BaseZMQProxy]: ...
    @overload
    def iter_all(category: Literal[PluginType.ZMQ_PROXY, "zmq_proxy"]) -> Iterator[tuple[PluginEntry, type[BaseZMQProxy]]]: ...
    @overload
    def get_class(category: Literal[PluginType.PLOT, "plot"], name_or_class_path: PlotType | str) -> type[PlotTypeHandlerProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.PLOT, "plot"]) -> Iterator[tuple[PluginEntry, type[PlotTypeHandlerProtocol]]]: ...
    @overload
    def get_class(category: Literal[PluginType.GPU_TELEMETRY_COLLECTOR, "gpu_telemetry_collector"], name_or_class_path: GPUTelemetryCollectorType | str) -> type[GPUTelemetryCollectorProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.GPU_TELEMETRY_COLLECTOR, "gpu_telemetry_collector"]) -> Iterator[tuple[PluginEntry, type[GPUTelemetryCollectorProtocol]]]: ...
    @overload
    def get_class(category: PluginType | PluginTypeStr, name_or_class_path: str) -> type: ...
    # fmt: on
    # </generated-overloads>


# ==============================================================================
# Module-Level Singleton
# ==============================================================================
# This pattern follows the random_generator module design.
# Usage:
#   from aiperf.plugin import plugins
#   from aiperf.plugin.enums import PluginType
#   EndpointClass = plugins.get_class(PluginType.ENDPOINT, 'openai')
#   endpoint = EndpointClass(...)
# ==============================================================================

# Create singleton instance at module load.
# Rationale: module-level singleton is the documented public access pattern for the
# plugin registry; all callers use `from aiperf.plugin import plugins`.
_registry = _PluginRegistry()

# ==============================================================================
# Public API: Module-Level Functions
# ==============================================================================

# Core lookup
get_class = _registry.get_class
get_entry = _registry.get_entry
has_entry = _registry.has_entry

# Iteration
iter_all = _registry.iter_all
iter_entries = _registry.iter_entries

# Listing
list_categories = _registry.list_categories
list_entries = _registry.list_entries
list_packages = _registry.list_packages

# Metadata
get_category_metadata = _registry.get_category_metadata
get_package_metadata = _registry.get_package_metadata

# Utilities
create_enum = _registry.create_enum
find_registered_name = _registry.find_registered_name
is_internal_category = _registry.is_internal_category
validate_all = _registry.validate_all

# Registration (for plugins and tests)
register = _registry.register
unregister = _registry.unregister
load_manifest = _registry.load_manifest
reset_registry = _registry.reset_registry
