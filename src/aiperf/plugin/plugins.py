# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plugin registry with lazy loading and priority-based conflict resolution.

Conflict resolution: higher priority wins; equal priority: external beats built-in.
"""

from __future__ import annotations

from collections.abc import Iterator

from typing import TYPE_CHECKING, Any, TypeAlias

from ruamel.yaml import YAML

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.plugin.extensible_enums import _normalize_name
from aiperf.plugin.schema.schemas import (
    ConvergenceCriterionMetadata,
    CustomDatasetLoaderMetadata,
    EndpointMetadata,
    GPUTelemetryCollectorMetadata,
    PlotMetadata,
    PublicDatasetLoaderMetadata,
    SearchPlannerMetadata,
    ServiceMetadata,
    TransportMetadata,
)
from aiperf.plugin.types import (
    PluginEntry,
)
from aiperf.plugin._registry import _PluginRegistry


# Alias for category normalization - same as plugin name normalization
_normalize_category = _normalize_name

_logger = AIPerfLogger(__name__)
_yaml = YAML(typ="safe")

# Type alias to reduce repetition throughout the module
if TYPE_CHECKING:
    from aiperf.plugin.enums import PluginType, PluginTypeStr

    CategoryT: TypeAlias = PluginType | PluginTypeStr
else:
    CategoryT: TypeAlias = str


# ==============================================================================
# Registry Class
# ==============================================================================


# ==============================================================================
# Overloaded functions
# ==============================================================================
if TYPE_CHECKING:
    # <generated-imports>
    # fmt: off
    # ruff: noqa: I001
    from aiperf.accuracy.protocols import AccuracyBenchmarkProtocol, AccuracyGraderProtocol
    from aiperf.api.routers.base_router import BaseRouter
    from aiperf.common.accumulator_protocols import AccumulatorProtocol, AnalyzerProtocol, StreamExporterProtocol
    from aiperf.common.protocols import CommunicationClientProtocol, CommunicationProtocol, ServiceProtocol
    from aiperf.controller.protocols import ServiceManagerProtocol
    from aiperf.dataset.composer.base import BaseDatasetComposer
    from aiperf.dataset.protocols import CustomDatasetLoaderProtocol, DatasetBackingStoreProtocol, DatasetClientStoreProtocol, DatasetSamplingStrategyProtocol, PublicDatasetLoaderProtocol
    from aiperf.endpoints.protocols import EndpointProtocol
    from aiperf.exporters.protocols import ConsoleExporterProtocol, DataExporterProtocol
    from aiperf.gpu_telemetry.protocols import GPUTelemetryCollectorProtocol
    from aiperf.orchestrator.convergence.base import ConvergenceCriterion
    from aiperf.orchestrator.search_planner.base import SearchPlanner
    from aiperf.plot.core.plot_type_handlers import PlotTypeHandlerProtocol
    from aiperf.plugin.enums import APIRouterType, AccumulatorType, AccuracyBenchmarkType, AccuracyGraderType, AnalyzerType, ArrivalPattern, CommClientType, CommunicationBackend, ComposerType, ConsoleExporterType, ConvergenceCriterionType, CustomDatasetType, DataExporterType, DatasetBackingStoreType, DatasetClientStoreType, DatasetSamplingStrategy, EndpointType, GPUTelemetryCollectorType, GPUTelemetryProcessorType, PlotType, PluginType, PluginTypeStr, PublicDatasetType, RampType, RecordProcessorType, ResultsProcessorType, SearchPlannerType, SearchRecipePostProcessType, SearchRecipeType, ServerMetricsProcessorType, ServiceRunType, ServiceType, StreamExporterType, TimingMode, TransportType, UIType, URLSelectionStrategy, ZMQProxyType
    from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor
    from aiperf.post_processors.protocols import RecordProcessorProtocol
    from aiperf.search_recipes._base import SearchRecipe
    from aiperf.search_recipes.post_process import PostProcessHandler
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
    def get_class(category: Literal[PluginType.SEARCH_RECIPE, "search_recipe"], name_or_class_path: SearchRecipeType | str) -> type[SearchRecipe]: ...
    @overload
    def iter_all(category: Literal[PluginType.SEARCH_RECIPE, "search_recipe"]) -> Iterator[tuple[PluginEntry, type[SearchRecipe]]]: ...
    @overload
    def get_class(category: Literal[PluginType.SEARCH_RECIPE_POST_PROCESS, "search_recipe_post_process"], name_or_class_path: SearchRecipePostProcessType | str) -> type[PostProcessHandler]: ...
    @overload
    def iter_all(category: Literal[PluginType.SEARCH_RECIPE_POST_PROCESS, "search_recipe_post_process"]) -> Iterator[tuple[PluginEntry, type[PostProcessHandler]]]: ...
    @overload
    def get_class(category: Literal[PluginType.CONVERGENCE_CRITERION, "convergence_criterion"], name_or_class_path: ConvergenceCriterionType | str) -> type[ConvergenceCriterion]: ...
    @overload
    def iter_all(category: Literal[PluginType.CONVERGENCE_CRITERION, "convergence_criterion"]) -> Iterator[tuple[PluginEntry, type[ConvergenceCriterion]]]: ...
    @overload
    def get_class(category: Literal[PluginType.SEARCH_PLANNER, "search_planner"], name_or_class_path: SearchPlannerType | str) -> type[SearchPlanner]: ...
    @overload
    def iter_all(category: Literal[PluginType.SEARCH_PLANNER, "search_planner"]) -> Iterator[tuple[PluginEntry, type[SearchPlanner]]]: ...
    @overload
    def get_class(category: Literal[PluginType.ACCUMULATOR, "accumulator"], name_or_class_path: AccumulatorType | str) -> type[AccumulatorProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.ACCUMULATOR, "accumulator"]) -> Iterator[tuple[PluginEntry, type[AccumulatorProtocol]]]: ...
    @overload
    def get_class(category: Literal[PluginType.STREAM_EXPORTER, "stream_exporter"], name_or_class_path: StreamExporterType | str) -> type[StreamExporterProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.STREAM_EXPORTER, "stream_exporter"]) -> Iterator[tuple[PluginEntry, type[StreamExporterProtocol]]]: ...
    @overload
    def get_class(category: Literal[PluginType.ANALYZER, "analyzer"], name_or_class_path: AnalyzerType | str) -> type[AnalyzerProtocol]: ...
    @overload
    def iter_all(category: Literal[PluginType.ANALYZER, "analyzer"]) -> Iterator[tuple[PluginEntry, type[AnalyzerProtocol]]]: ...
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

# Create singleton instance at module load
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


# ==============================================================================
# Metadata Helpers
# ==============================================================================


def get_metadata(category: CategoryT, name: str) -> dict[str, Any]:
    """Get raw metadata dict for a plugin.

    Args:
        category: Plugin category.
        name: Plugin name.

    Returns:
        Metadata dict from plugins.yaml.
    """
    return get_entry(category, name).metadata


def get_endpoint_metadata(name: str) -> EndpointMetadata:
    """Get typed metadata for an endpoint plugin.

    Args:
        name: Endpoint plugin name (e.g., 'chat', 'completions').

    Returns:
        Validated EndpointMetadata instance.
    """
    return get_entry("endpoint", name).get_typed_metadata(EndpointMetadata)


def get_transport_metadata(name: str) -> TransportMetadata:
    """Get typed metadata for a transport plugin.

    Args:
        name: Transport plugin name (e.g., 'http').

    Returns:
        Validated TransportMetadata instance.
    """
    return get_entry("transport", name).get_typed_metadata(TransportMetadata)


def get_plot_metadata(name: str) -> PlotMetadata:
    """Get typed metadata for a plot plugin.

    Args:
        name: Plot plugin name (e.g., 'scatter', 'histogram').

    Returns:
        Validated PlotMetadata instance.
    """
    return get_entry("plot", name).get_typed_metadata(PlotMetadata)


def get_service_metadata(name: str) -> ServiceMetadata:
    """Get typed metadata for a service plugin.

    Args:
        name: Service plugin name (e.g., 'worker', 'timing_manager').

    Returns:
        Validated ServiceMetadata instance.
    """
    return get_entry("service", name).get_typed_metadata(ServiceMetadata)


def get_dataset_loader_metadata(name: str) -> CustomDatasetLoaderMetadata:
    """Get typed metadata for a custom dataset loader plugin.

    Args:
        name: Dataset loader plugin name (e.g., 'mooncake_trace', 'bailian_trace').

    Returns:
        Validated CustomDatasetLoaderMetadata instance.
    """
    return get_entry("custom_dataset_loader", name).get_typed_metadata(
        CustomDatasetLoaderMetadata
    )


def get_public_dataset_loader_metadata(name: str) -> PublicDatasetLoaderMetadata:
    """Get typed metadata for a public dataset loader plugin.

    Args:
        name: Public dataset loader plugin name (e.g., 'aimo', 'sharegpt').

    Returns:
        Validated PublicDatasetLoaderMetadata instance.
    """
    return get_entry("public_dataset_loader", name).get_typed_metadata(
        PublicDatasetLoaderMetadata
    )


def get_gpu_telemetry_collector_metadata(
    name: str,
) -> GPUTelemetryCollectorMetadata:
    """Get typed metadata for a GPU telemetry collector plugin.

    Args:
        name: Collector plugin name (e.g., 'dcgm', 'pynvml', 'amdsmi').

    Returns:
        Validated GPUTelemetryCollectorMetadata instance.
    """
    return get_entry("gpu_telemetry_collector", name).get_typed_metadata(
        GPUTelemetryCollectorMetadata
    )


def is_trace_dataset(name: str) -> bool:
    """Check if a custom dataset loader is a trace-format dataset.

    Args:
        name: Dataset loader plugin name (e.g., 'mooncake_trace', 'single_turn').

    Returns:
        True if the loader handles trace-format datasets.
    """
    return get_dataset_loader_metadata(name).is_trace


def get_convergence_criterion_metadata(name: str) -> ConvergenceCriterionMetadata:
    """Get typed metadata for a convergence criterion plugin.

    Args:
        name: Convergence criterion plugin name (e.g., 'ci_width', 'cv', 'distribution').

    Returns:
        Validated ConvergenceCriterionMetadata instance.
    """
    return get_entry("convergence_criterion", name).get_typed_metadata(
        ConvergenceCriterionMetadata
    )


def get_search_planner_metadata(name: str) -> SearchPlannerMetadata:
    """Get typed metadata for a search planner plugin.

    Args:
        name: Search planner plugin name (e.g., 'bayesian').

    Returns:
        Validated SearchPlannerMetadata instance.
    """
    return get_entry("search_planner", name).get_typed_metadata(SearchPlannerMetadata)


# Mapping of categories to their metadata classes (for categories with typed metadata)
_CATEGORY_METADATA_CLASSES: dict[str, type] = {
    "endpoint": EndpointMetadata,
    "transport": TransportMetadata,
    "plot": PlotMetadata,
    "service": ServiceMetadata,
    "custom_dataset_loader": CustomDatasetLoaderMetadata,
    "convergence_criterion": ConvergenceCriterionMetadata,
    "search_planner": SearchPlannerMetadata,
    "gpu_telemetry_collector": GPUTelemetryCollectorMetadata,
}


def get_typed_metadata(category: CategoryT, name: str) -> Any:
    """Get typed metadata for any plugin that has a registered metadata class.

    This is a generic helper that automatically uses the correct metadata class
    based on the category. For categories without a registered metadata class,
    returns the raw metadata dict.

    Args:
        category: Plugin category (e.g., PluginType.ENDPOINT or "endpoint").
            Supports dash/underscore normalized matching.
        name: Plugin name within the category.

    Returns:
        Validated metadata instance if the category has a metadata class,
        otherwise the raw metadata dict.

    Example:
        >>> endpoint_meta = get_typed_metadata(PluginType.ENDPOINT, "chat")
        >>> print(endpoint_meta.streaming)  # Typed access
        True
    """
    category = _normalize_category(category)
    entry = get_entry(category, name)
    if metadata_cls := _CATEGORY_METADATA_CLASSES.get(category):
        return entry.get_typed_metadata(metadata_cls)

    # Fall back to raw metadata dict
    return entry.metadata
