# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pydantic models for plugin YAML schema validation.

Defines the structure of categories.yaml (extension points) and plugins.yaml
(implementations). Used for JSON Schema generation (IDE autocomplete/validation)
and runtime validation.

To regenerate JSON schemas: python tools/generate_plugin_artifacts.py --schemas
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# Plugins YAML Schema (plugins.yaml)
# =============================================================================
# These models define the structure of plugins.yaml, which registers concrete
# plugin implementations. The PluginsManifest is the root, containing category
# sections that map type names to PluginSpec entries.
# =============================================================================


class PluginsManifest(BaseModel):
    """Root model for plugins.yaml file.

    This file registers plugin implementations for AIPerf. Each top-level section
    corresponds to a category from categories.yaml and maps type names to their
    implementing classes.

    Note: Package metadata (name, version, author) comes from pyproject.toml
    via importlib.metadata, not from this file.

    Example:
    ```yaml
    schema_version: "1.0"

    endpoint:
      my_custom:
        class: my_package.endpoints.my_custom:MyCustomEndpoint
        description: Custom endpoint for my API.
        metadata:
          endpoint_path: /v1/generate
          supports_streaming: true
          produces_tokens: true
          tokenizes_input: true
          metrics_title: My Custom Metrics

    custom_dataset_loader:
      my_jsonl:
        class: my_package.dataset_loaders.my_jsonl:MyJSONLDatasetLoader
        description: Custom dataset loader for my API.
    ```
    """

    # Plugin categories are stored as additional fields
    model_config = ConfigDict(extra="allow")

    schema_version: str = Field(
        default="1.0",
        description="Version of the plugins.yaml schema format. Use '1.0' for current format.",
    )


class PluginSpec(BaseModel):
    """Specification for a plugin implementation.

    Each plugin entry maps a type name (like 'chat' or 'completions') to a Python
    class that implements the category's protocol. The type name becomes an enum
    member (e.g., EndpointType.CHAT) used for configuration and API selection.

    Example::

        chat:
          class: aiperf.endpoints.openai_chat:ChatEndpoint
          description: |
            OpenAI Chat Completions endpoint. Supports multi-modal inputs
            and streaming responses.
          metadata:
            endpoint_path: /v1/chat/completions
            supports_streaming: true
    """

    model_config = ConfigDict(populate_by_name=True)

    class_: str = Field(
        alias="class",
        description=(
            "Python class that implements this plugin entry. "
            "Use 'module.path:ClassName' format, e.g., 'aiperf.endpoints.openai_chat:ChatEndpoint'."
        ),
    )
    description: str = Field(
        default="",
        description="Brief explanation of what this plugin type does and when to use it.",
    )
    priority: int = Field(
        default=0,
        description=(
            "Conflict resolution priority. When multiple packages register the same type name, "
            "the one with higher priority wins. Use 0 for normal plugins, higher values to "
            "override built-in implementations."
        ),
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Category-specific configuration for this plugin type. "
            "The allowed fields depend on the category's metadata_class in categories.yaml."
        ),
    )


# =============================================================================
# Categories YAML Schema (categories.yaml)
# =============================================================================
# These models define the structure of categories.yaml, which declares all
# extension points in AIPerf. The CategoriesManifest is the root, containing
# CategorySpec entries that define protocols, enums, and optional metadata schemas.
# =============================================================================


class CategoriesManifest(BaseModel):
    """Root model for categories.yaml file.

    This file defines all plugin extension points in AIPerf. Each category
    specifies a protocol interface and generates an enum for type selection.

    Example:
    ```yaml
    schema_version: "1.0"

    endpoint:
      protocol: aiperf.endpoints.protocols:EndpointProtocol
      metadata_class: aiperf.plugin.schema.schemas:EndpointMetadata
      enum: EndpointType
      description: HTTP endpoint handlers for LLM APIs

    service:
      protocol: aiperf.common.protocols:ServiceProtocol
      metadata_class: aiperf.plugin.schema.schemas:ServiceMetadata
      enum: ServiceType
      description: Service plugins for AIPerf.

    ...
    ```
    """

    # Categories are stored as additional fields beyond schema_version
    model_config = ConfigDict(extra="allow")

    schema_version: str = Field(
        default="1.0",
        description="Version of the categories.yaml schema format. Used for backwards compatibility.",
    )


class CategorySpec(BaseModel):
    """Specification for a plugin category (extension point).

    Categories define extension points in AIPerf where plugins can provide
    custom implementations. The protocol defines the interface, and the enum
    provides type-safe selection in configuration files and APIs.

    Example::

        endpoint:
          protocol: aiperf.endpoints.protocols:EndpointProtocol
          metadata_class: aiperf.plugin.schema.schemas:EndpointMetadata
          enum: EndpointType
          description: HTTP endpoint handlers for different LLM APIs.
    """

    protocol: str = Field(
        description=(
            "The interface that plugins in this category must implement. "
            "Use 'module.path:ClassName' format, e.g., 'aiperf.endpoints.protocols:EndpointProtocol'."
        )
    )
    metadata_class: str | None = Field(
        default=None,
        description=(
            "Optional class path for category-specific metadata. "
            "When set, plugins can include typed metadata fields validated against this schema. "
            "Use 'module.path:ClassName' format."
        ),
    )
    enum: str = Field(
        description=(
            "Name of the enum that will be auto-generated from registered plugins. "
            "This enum is used in config files and APIs to select plugin types, "
            "e.g., 'EndpointType' generates EndpointType.CHAT, EndpointType.COMPLETIONS, etc."
        )
    )
    description: str = Field(
        description="Brief explanation of what this category is for and when to use it."
    )
    internal: bool = Field(
        default=False,
        description=(
            "Set to true for infrastructure categories not meant for end users. "
            "Internal categories are hidden from documentation and plugin listings."
        ),
    )


# =============================================================================
# Plugin Metadata Classes
# =============================================================================
# These classes define typed metadata schemas for specific plugin categories.
# They are referenced by categories.yaml via the `metadata_class` field and
# used to validate the `metadata` field in plugins.yaml entries.
#
# When adding a new metadata class:
# 1. Define the Pydantic model here with Field descriptions
# 2. Reference it in categories.yaml: metadata_class: aiperf.plugin.schema.schemas:YourMetadata
# 3. Run `python tools/generate_plugin_artifacts.py --schemas` to update JSON schemas
# 4. Plugins can then include typed metadata validated against this schema
# =============================================================================


class EndpointMetadata(BaseModel):
    """Metadata schema for endpoint plugins.

    Defines API capabilities, paths, and multimodal support for endpoint implementations.
    Used by the framework to route requests, configure streaming, and enable
    multimodal inputs/outputs (images, audio, video).

    Referenced by: categories.yaml endpoint.metadata_class
    Used in: plugins.yaml endpoint entries
    """

    metrics_title: str | None = Field(
        ..., description="Display title for metrics dashboard."
    )
    endpoint_path: str | None = Field(
        ..., description="API path (e.g., /v1/chat/completions)."
    )
    streaming_path: str | None = Field(
        default=None,
        description="Streaming API path if different from the endpoint path (e.g., /generate_stream).",
    )
    service_kind: str = Field(
        default="openai",
        description="The service kind of the endpoint (used for artifact naming).",
    )
    supports_streaming: bool = Field(
        ..., description="Whether endpoint supports streaming responses."
    )
    tokenizes_input: bool = Field(
        ..., description="Whether endpoint tokenizes text inputs."
    )
    produces_tokens: bool = Field(
        ..., description="Whether endpoint produces token-based output."
    )
    supports_audio: bool = Field(
        default=False, description="Whether endpoint accepts audio input."
    )
    supports_images: bool = Field(
        default=False, description="Whether endpoint accepts image input."
    )
    supports_videos: bool = Field(
        default=False, description="Whether endpoint accepts video input."
    )
    produces_audio: bool = Field(
        default=False, description="Whether endpoint produces audio-based outputs."
    )
    produces_images: bool = Field(
        default=False, description="Whether endpoint produces image-based outputs."
    )
    produces_videos: bool = Field(
        default=False, description="Whether endpoint produces video-based outputs."
    )
    requires_polling: bool = Field(
        default=False,
        description="Whether endpoint uses async job polling (submit job, poll for status, retrieve result).",
    )


class TransportMetadata(BaseModel):
    """Metadata schema for transport plugins.

    Defines network layer configuration including transport type and URL schemes.
    Used by the framework to auto-detect appropriate transport based on URL scheme.

    Referenced by: categories.yaml transport.metadata_class
    Used in: plugins.yaml transport entries
    """

    transport_type: str = Field(
        description="Transport type identifier for this transport"
    )
    url_schemes: list[str] = Field(
        default_factory=list,
        description="URL schemes this transport handles (for auto-detection and validation).",
    )


class PlotMetadata(BaseModel):
    """Metadata schema for plot plugins.

    Defines display properties and categorization for visualization handlers.
    Used by the framework to group plots and generate selection interfaces.

    Referenced by: categories.yaml plot.metadata_class
    Used in: plugins.yaml plot entries
    """

    display_name: str = Field(description="Human-readable name for UI display.")
    category: str = Field(
        description="Plot category (per_request, aggregated, combined, comparison)."
    )


class ServiceMetadata(BaseModel):
    """Metadata schema for service plugins.

    Defines lifecycle and runtime configuration for AIPerf distributed services.
    Used by SystemController to manage service startup order and optimize
    latency-sensitive services (timing, workers) by disabling garbage collection.

    Referenced by: categories.yaml service.metadata_class
    Used in: plugins.yaml service entries
    """

    required: bool = Field(
        description="Whether the service is required for benchmark execution."
    )
    auto_start: bool = Field(
        description="Whether the service is automatically started by the system controller."
    )
    disable_gc: bool = Field(
        default=False,
        description="Whether to disable garbage collection in the service for timing-critical operations.",
    )
    replicable: bool = Field(
        default=False,
        description="Whether the service can have multiple instances running in parallel.",
    )
