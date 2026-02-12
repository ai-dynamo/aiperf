<!--
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# AIPerf Plugin System

The AIPerf plugin system provides a flexible, extensible architecture for customizing benchmark behavior. It uses YAML-based configuration with lazy loading, priority-based conflict resolution, and dynamic enum generation.

## Table of Contents

- [Overview](#overview)
  - [Terminology](#terminology)
  - [Key Components](#key-components)
- [Architecture](#architecture)
- [Plugin Categories](#plugin-categories)
- [Using Plugins](#using-plugins)
- [Creating Custom Plugins](#creating-custom-plugins)
- [Plugin Configuration](#plugin-configuration)
- [CLI Commands](#cli-commands)
- [Advanced Topics](#advanced-topics)

## Overview

The plugin system enables:

- **Extensibility**: Add custom endpoints, exporters, and timing strategies without modifying core code
- **Lazy Loading**: Classes load on first access, avoiding circular imports
- **Conflict Resolution**: Higher priority plugins override lower priority ones
- **Type Safety**: Auto-generated enums provide IDE autocomplete
- **Validation**: Validate plugins without importing them

### Terminology

| Term | Description | Code Type |
|------|-------------|-----------|
| **Registry** | Global singleton holding all plugins | `_PluginRegistry` |
| **Package** | Python package providing plugins | `PackageInfo` |
| **Manifest** | `plugins.yaml` declaring plugins | `PluginsManifest` |
| **Category** | Plugin type (e.g., `endpoint`, `transport`) | `PluginType` enum |
| **Entry** | Single registered plugin (name, class_path, priority, metadata) | `PluginEntry` |
| **Class** | Python class implementing a plugin (lazy-loaded) | `type` |
| **Metadata** | Typed configuration (e.g., `EndpointMetadata`) | Pydantic model |

**Hierarchy:**

```
Registry (singleton)
└── Package (1+) ─── discovered via entry points
    └── Manifest (1+ per package) ─── plugins.yaml files
        └── Category (1+)
            └── Entry (1+) ─── PluginEntry
                ├── Class ─── lazy-loaded Python class
                └── Metadata ─── optional typed config
```

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| Plugin Registry | `src/aiperf/plugin/plugins.py` | Singleton managing discovery and loading |
| Plugin Entry | `src/aiperf/plugin/types.py` | Lazy-loading entry with metadata |
| Categories | `src/aiperf/plugin/categories.yaml` | Category definitions with protocols |
| Built-in Plugins | `src/aiperf/plugin/plugins.yaml` | Built-in plugin registrations |
| Schemas | `src/aiperf/plugin/schema/schemas.py` | Pydantic models for validation |
| Enums | `src/aiperf/plugin/enums.py` | Auto-generated enums from registry |
| CLI | `src/aiperf/cli_commands/plugins_cli.py` | Plugin exploration commands |

## Architecture

### Discovery Flow

```
Entry Points → plugins.yaml → Pydantic Validation → Registry
                                                      ↓
                              get_class() → Import Module → Cache
```

| Phase | Action |
|-------|--------|
| 1. Discovery | Scan `aiperf.plugins` entry points for `plugins.yaml` files |
| 2. Loading | Parse YAML, validate with Pydantic, register with conflict resolution |
| 3. Access | `get_class()` imports module, caches class for reuse |

### Registry Singleton Pattern

The plugin registry follows the singleton pattern with module-level exports:

```python
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType

# Get a plugin class by name
EndpointClass = plugins.get_class(PluginType.ENDPOINT, "chat")

# Iterate all plugins in a category
for entry, cls in plugins.iter_all(PluginType.ENDPOINT):
    print(f"{entry.name}: {entry.description}")
```

## Plugin Categories

AIPerf supports 22 plugin categories organized by function:

### Timing Categories

| Category | Enum | Description |
|----------|------|-------------|
| `timing_strategy` | `TimingMode` | Request scheduling strategies (fixed schedule, request rate, user-centric) |
| `arrival_pattern` | `ArrivalPattern` | Inter-arrival time distributions (constant, Poisson, gamma, concurrency burst) |
| `ramp` | `RampType` | Value ramping strategies (linear, exponential, Poisson) |

### Dataset Categories

| Category | Enum | Description |
|----------|------|-------------|
| `dataset_backing_store` | `DatasetBackingStoreType` | Server-side dataset storage |
| `dataset_client_store` | `DatasetClientStoreType` | Worker-side dataset access |
| `dataset_sampler` | `DatasetSamplingStrategy` | Sampling strategies (random, sequential, shuffle) |
| `dataset_composer` | `ComposerType` | Dataset generation (synthetic, custom, rankings) |
| `custom_dataset_loader` | `CustomDatasetType` | JSONL format loaders |

### Endpoint and Transport Categories

| Category | Enum | Description |
|----------|------|-------------|
| `endpoint` | `EndpointType` | API endpoint implementations (chat, completions, embeddings, etc.) |
| `transport` | `TransportType` | Network transport (HTTP via aiohttp) |

### Processing Categories

| Category | Enum | Description |
|----------|------|-------------|
| `record_processor` | `RecordProcessorType` | Per-record metric computation |
| `results_processor` | `ResultsProcessorType` | Aggregated results computation |
| `data_exporter` | `DataExporterType` | File format exporters (CSV, JSON, Parquet) |
| `console_exporter` | `ConsoleExporterType` | Terminal output exporters |

### UI and Selection Categories

| Category | Enum | Description |
|----------|------|-------------|
| `ui` | `UIType` | UI implementations (dashboard, simple, none) |
| `url_selection_strategy` | `URLSelectionStrategy` | Request distribution (round-robin) |

### Infrastructure Categories (Internal)

| Category | Enum | Description |
|----------|------|-------------|
| `service` | `ServiceType` | Core AIPerf services |
| `service_manager` | `ServiceRunType` | Service orchestration (multiprocessing, Kubernetes) |
| `communication` | `CommunicationBackend` | ZMQ backends (IPC, TCP) |
| `communication_client` | `CommClientType` | Socket patterns (PUB, SUB, PUSH, PULL) |
| `zmq_proxy` | `ZMQProxyType` | Message routing proxies |

### Visualization Category

| Category | Enum | Description |
|----------|------|-------------|
| `plot` | `PlotType` | Chart types (scatter, histogram, timeline, etc.) |

## Using Plugins

```python
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType, EndpointType

# Get class by name, enum, or full path
ChatEndpoint = plugins.get_class(PluginType.ENDPOINT, "chat")
ChatEndpoint = plugins.get_class(PluginType.ENDPOINT, EndpointType.CHAT)
ChatEndpoint = plugins.get_class(PluginType.ENDPOINT, "aiperf.endpoints.openai_chat:ChatEndpoint")

# Iterate plugins
for entry, cls in plugins.iter_all(PluginType.ENDPOINT):
    print(f"{entry.name}: {entry.class_path}")

# Get metadata (raw dict or typed)
metadata = plugins.get_metadata("endpoint", "chat")
endpoint_meta = plugins.get_endpoint_metadata("chat")  # Returns EndpointMetadata
```

| Function | Returns | Use Case |
|----------|---------|----------|
| `get_class(category, name)` | `type` | Get plugin class |
| `iter_all(category)` | `Iterator[tuple[PluginEntry, type]]` | List all plugins |
| `get_metadata(category, name)` | `dict` | Raw metadata |
| `get_endpoint_metadata(name)` | `EndpointMetadata` | Typed endpoint config |
| `get_transport_metadata(name)` | `TransportMetadata` | Typed transport config |
| `get_plot_metadata(name)` | `PlotMetadata` | Typed plot config |
| `get_service_metadata(name)` | `ServiceMetadata` | Typed service config |

## Creating Custom Plugins

**Quick Start** (4 steps):

| Step | File | Action |
|------|------|--------|
| 1 | `my_endpoint.py` | Create class extending `BaseEndpoint` |
| 2 | `plugins.yaml` | Register with class path, description, and metadata |
| 3 | `pyproject.toml` | Add entry point: `my-package = "my_package:plugins.yaml"` |
| 4 | Terminal | `uv pip install -e . && aiperf plugins endpoint my_custom` |

### Minimal Endpoint Example

```python
# my_package/endpoints/custom_endpoint.py
class MyCustomEndpoint(BaseEndpoint):
    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        turn = request_info.turns[-1]
        return {"prompt": turn.texts[0].contents[0] if turn.texts else ""}

    def parse_response(self, response: InferenceServerResponse) -> ParsedResponse | None:
        if json_obj := response.get_json():
            return ParsedResponse(perf_ns=response.perf_ns, data=TextResponseData(text=json_obj.get("text", "")))
        return None
```

```yaml
# my_package/plugins.yaml
schema_version: "1.0"
endpoint:
  my_custom:
    class: my_package.endpoints.custom_endpoint:MyCustomEndpoint
    description: Custom endpoint for my API.
    metadata: { endpoint_path: /v1/generate, supports_streaming: true, produces_tokens: true, tokenizes_input: true }
```

> **Note**: Extend base classes (`BaseEndpoint`, etc.) to get logging, helpers, and default implementations. Only implement core methods.

## Plugin Configuration

### categories.yaml Schema

Defines plugin categories with their protocols and metadata schemas:

```yaml
schema_version: "1.0"

endpoint:
  protocol: aiperf.endpoints.protocols:EndpointProtocol
  metadata_class: aiperf.plugin.schema.schemas:EndpointMetadata
  enum: EndpointType
  description: |
    Endpoints define how to format requests and parse responses for different APIs.
  internal: false  # Set to true for infrastructure categories
```

### plugins.yaml Schema

Registers plugin implementations:

```yaml
schema_version: "1.0"

endpoint:
  chat:
    class: aiperf.endpoints.openai_chat:ChatEndpoint
    description: OpenAI Chat Completions endpoint.
    priority: 0  # Higher priority wins conflicts
    metadata:
      endpoint_path: /v1/chat/completions
      supports_streaming: true
      produces_tokens: true
      tokenizes_input: true
      metrics_title: LLM Metrics
```

### Metadata Schemas

Category-specific metadata is validated against Pydantic models in `aiperf.plugin.schema.schemas`:

| Model | Key Fields |
|-------|------------|
| `EndpointMetadata` | `endpoint_path`, `supports_streaming`, `produces_tokens`, `tokenizes_input` |
| `TransportMetadata` | `transport_type`, `url_schemes` |
| `PlotMetadata` | `display_name`, `category` |
| `ServiceMetadata` | `required`, `auto_start`, `disable_gc` |

## CLI Commands

| Command | Output |
|---------|--------|
| `aiperf plugins` | All categories with registered plugins |
| `aiperf plugins endpoint` | All endpoint types with descriptions |
| `aiperf plugins endpoint chat` | Details: class path, package, metadata |
| `aiperf plugins --packages` | Installed packages with versions |
| `aiperf plugins --validate` | Validates ordering, paths, class existence |

```bash
$ aiperf plugins endpoint chat
╭───────────────── endpoint:chat ─────────────────╮
│ Type: chat                                      │
│ Category: endpoint                              │
│ Package: aiperf                                 │
│ Class: aiperf.endpoints.openai_chat:ChatEndpoint│
│                                                 │
│ OpenAI Chat Completions endpoint for LLM APIs. │
╰─────────────────────────────────────────────────╯
```

## Advanced Topics

### Conflict Resolution

| Priority | Rule |
|----------|------|
| 1 | Higher `priority` value wins |
| 2 | External packages beat built-in (equal priority) |
| 3 | First registered wins (with warning) |

> **Tip**: Shadowed plugins remain accessible via full class path: `plugins.get_class("endpoint", "my_pkg.endpoints:MyEndpoint")`

### API Reference

```python
# Runtime registration (testing)
plugins.register("endpoint", "test", TestEndpoint, priority=10)
plugins.reset_registry()  # Reset to initial state

# Dynamic enum generation
MyEndpointType = plugins.create_enum(PluginType.ENDPOINT, "MyEndpointType", module=__name__)

# Validation without importing
errors = plugins.validate_all(check_class=True)  # {category: [(name, error), ...]}

# Reverse lookup
name = plugins.find_registered_name(PluginType.ENDPOINT, ChatEndpoint)  # "chat"

# Package metadata
pkg = plugins.get_package_metadata("aiperf")  # PackageInfo(version, author, ...)
```

> **Type Safety**: `get_class()` returns typed results (e.g., `type[EndpointProtocol]`) with IDE autocomplete.

## Built-in Plugins Reference

### Endpoints

| Name | Class | Description |
|------|-------|-------------|
| `chat` | `ChatEndpoint` | OpenAI Chat Completions API |
| `completions` | `CompletionsEndpoint` | OpenAI Completions API |
| `embeddings` | `EmbeddingsEndpoint` | OpenAI Embeddings API |
| `image_generation` | `ImageGenerationEndpoint` | OpenAI Image Generation API |
| `huggingface_generate` | `HuggingFaceGenerateEndpoint` | HuggingFace TGI |
| `cohere_rankings` | `CohereRankingsEndpoint` | Cohere Reranking API |
| `hf_tei_rankings` | `HFTeiRankingsEndpoint` | HuggingFace TEI Rankings |
| `nim_embeddings` | `NIMEmbeddingsEndpoint` | NVIDIA NIM Embeddings |
| `nim_rankings` | `NIMRankingsEndpoint` | NVIDIA NIM Rankings |
| `solido_rag` | `SolidoEndpoint` | Solido RAG Pipeline |
| `template` | `TemplateEndpoint` | Template for custom endpoints |

### Timing Strategies

| Name | Class | Description |
|------|-------|-------------|
| `fixed_schedule` | `FixedScheduleStrategy` | Send requests at exact timestamps |
| `request_rate` | `RequestRateStrategy` | Send requests at specified rate |
| `user_centric_rate` | `UserCentricStrategy` | Each session acts as separate user |

### Arrival Patterns

| Name | Class | Description |
|------|-------|-------------|
| `constant` | `ConstantIntervalGenerator` | Fixed intervals between requests |
| `poisson` | `PoissonIntervalGenerator` | Poisson process arrivals |
| `gamma` | `GammaIntervalGenerator` | Gamma distribution with tunable smoothness |
| `concurrency_burst` | `ConcurrencyBurstIntervalGenerator` | Send ASAP up to concurrency limit |

### Dataset Composers

| Name | Class | Description |
|------|-------|-------------|
| `synthetic` | `SyntheticDatasetComposer` | Generate synthetic conversations |
| `custom` | `CustomDatasetComposer` | Load from JSONL files |
| `synthetic_rankings` | `SyntheticRankingsDatasetComposer` | Generate ranking tasks |

### UI Types

| Name | Class | Description |
|------|-------|-------------|
| `dashboard` | `AIPerfDashboardUI` | Rich terminal dashboard |
| `simple` | `TQDMProgressUI` | Simple tqdm progress bar |
| `none` | `NoUI` | Headless execution |

## Troubleshooting

### Plugin Not Found

```
TypeNotFoundError: Type 'my_plugin' not found for category 'endpoint'.
```

**Solutions**:
1. Verify the plugin is registered in `plugins.yaml`
2. Check the entry point is defined in `pyproject.toml`
3. Reinstall the package: `uv pip install -e .`
4. Run `aiperf plugins --validate` to check for errors

### Module Import Errors

```
ImportError: Failed to import module for endpoint:my_plugin
```

**Solutions**:
1. Verify the class path format: `module.path:ClassName`
2. Check all dependencies are installed
3. Verify the module is importable: `python -c "import module.path"`

### Class Not Found

```
AttributeError: Class 'MyClass' not found
```

**Solutions**:
1. Verify the class name matches exactly (case-sensitive)
2. Ensure the class is exported from the module
3. Run `aiperf plugins --validate` for detailed error

### Conflict Resolution Issues

If your plugin is being shadowed by another:

1. Use higher priority: `priority: 10` in `plugins.yaml`
2. Access by full class path: `plugins.get_class("endpoint", "my_pkg.endpoints:MyEndpoint")`
3. Check `aiperf plugins --packages` to see which packages are loaded
