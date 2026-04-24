# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed metadata helpers for plugin registry (moved out of plugins.py for file-size).

Provides typed accessors over the raw metadata dicts stored on `PluginEntry`.
Re-exported from `aiperf.plugin.plugins` so existing callers keep working.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

from aiperf.plugin.schema.schemas import (
    CustomDatasetLoaderMetadata,
    EndpointMetadata,
    PlotMetadata,
    PublicDatasetLoaderMetadata,
    ServiceMetadata,
    TransportMetadata,
)

if TYPE_CHECKING:
    from aiperf.plugin.enums import PluginType, PluginTypeStr

    CategoryT: TypeAlias = PluginType | PluginTypeStr
else:
    CategoryT: TypeAlias = str


def _normalize_category(category: CategoryT) -> str:
    """Normalize category for lookup. Lazy import to avoid circular dependency."""
    from aiperf.plugin.extensible_enums import _normalize_name

    return _normalize_name(category)


def _get_entry(category: CategoryT, name: str):
    """Lazy accessor for the registry facade to avoid import cycles at module init."""
    from aiperf.plugin import plugins as _plugins

    return _plugins.get_entry(category, name)


def get_metadata(category: CategoryT, name: str) -> dict[str, Any]:
    """Get raw metadata dict for a plugin.

    Args:
        category: Plugin category.
        name: Plugin name.

    Returns:
        Metadata dict from plugins.yaml.
    """
    return _get_entry(category, name).metadata


def get_endpoint_metadata(name: str) -> EndpointMetadata:
    """Get typed metadata for an endpoint plugin.

    Args:
        name: Endpoint plugin name (e.g., 'chat', 'completions').

    Returns:
        Validated EndpointMetadata instance.
    """
    return _get_entry("endpoint", name).get_typed_metadata(EndpointMetadata)


def get_transport_metadata(name: str) -> TransportMetadata:
    """Get typed metadata for a transport plugin.

    Args:
        name: Transport plugin name (e.g., 'http').

    Returns:
        Validated TransportMetadata instance.
    """
    return _get_entry("transport", name).get_typed_metadata(TransportMetadata)


def get_plot_metadata(name: str) -> PlotMetadata:
    """Get typed metadata for a plot plugin.

    Args:
        name: Plot plugin name (e.g., 'scatter', 'histogram').

    Returns:
        Validated PlotMetadata instance.
    """
    return _get_entry("plot", name).get_typed_metadata(PlotMetadata)


def get_service_metadata(name: str) -> ServiceMetadata:
    """Get typed metadata for a service plugin.

    Args:
        name: Service plugin name (e.g., 'worker', 'timing_manager').

    Returns:
        Validated ServiceMetadata instance.
    """
    return _get_entry("service", name).get_typed_metadata(ServiceMetadata)


def get_dataset_loader_metadata(name: str) -> CustomDatasetLoaderMetadata:
    """Get typed metadata for a custom dataset loader plugin.

    Args:
        name: Dataset loader plugin name (e.g., 'mooncake_trace', 'bailian_trace').

    Returns:
        Validated CustomDatasetLoaderMetadata instance.
    """
    return _get_entry("custom_dataset_loader", name).get_typed_metadata(
        CustomDatasetLoaderMetadata
    )


def get_public_dataset_loader_metadata(name: str) -> PublicDatasetLoaderMetadata:
    """Get typed metadata for a public dataset loader plugin.

    Args:
        name: Public dataset loader plugin name (e.g., 'aimo', 'sharegpt').

    Returns:
        Validated PublicDatasetLoaderMetadata instance.
    """
    return _get_entry("public_dataset_loader", name).get_typed_metadata(
        PublicDatasetLoaderMetadata
    )


def is_trace_dataset(name: str) -> bool:
    """Check if a custom dataset loader is a trace-format dataset.

    Args:
        name: Dataset loader plugin name (e.g., 'mooncake_trace', 'single_turn').

    Returns:
        True if the loader handles trace-format datasets.
    """
    return get_dataset_loader_metadata(name).is_trace


# Mapping of categories to their metadata classes (for categories with typed metadata)
_CATEGORY_METADATA_CLASSES: dict[str, type] = {
    "endpoint": EndpointMetadata,
    "transport": TransportMetadata,
    "plot": PlotMetadata,
    "service": ServiceMetadata,
    "custom_dataset_loader": CustomDatasetLoaderMetadata,
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
    entry = _get_entry(category, name)
    if metadata_cls := _CATEGORY_METADATA_CLASSES.get(category):
        return entry.get_typed_metadata(metadata_cls)

    # Fall back to raw metadata dict
    return entry.metadata
