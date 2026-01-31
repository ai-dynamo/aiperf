<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Creating Your First AIPerf Plugin

This tutorial walks you through creating a custom AIPerf endpoint plugin from scratch. By the end, you'll have a working plugin package that can benchmark any custom API.

## What You'll Build

We'll create a plugin for a hypothetical "Echo API" that returns the input text with some metadata. This simple example demonstrates all the core concepts you need to build more complex plugins.

## Prerequisites

- Python 3.10+
- AIPerf installed (`uv pip install aiperf`)
- Basic understanding of Python async/await and Pydantic

## Key Concepts

Before diving in, understand the plugin system terminology:

| Term | What It Is |
|------|------------|
| **Package** | Your Python package that provides plugins (e.g., `my-aiperf-plugins`) |
| **Manifest** | The `plugins.yaml` file declaring your plugins |
| **Category** | A type of plugin (e.g., `endpoint`, `transport`, `timing_strategy`) |
| **Entry** | A single registered plugin within a category |
| **Class** | The Python class implementing your plugin |
| **Metadata** | Configuration describing your plugin's capabilities |

**What you're building:**

```
Package (my-aiperf-plugins)
└── Manifest (plugins.yaml)
    └── Category (endpoint)
        └── Entry (echo)
            ├── Class (EchoEndpoint)
            └── Metadata (supports_streaming: true, ...)
```

For complete plugin system documentation, see the [Plugin System Reference](./plugin-system.md).

## Project Structure

Create a new directory for your plugin package:

```
my-aiperf-plugins/
├── pyproject.toml
├── src/
│   └── my_plugins/
│       ├── __init__.py
│       ├── plugins.yaml
│       └── endpoints/
│           ├── __init__.py
│           └── echo_endpoint.py
└── tests/
    └── test_echo_endpoint.py
```

## Step 1: Create the Project Files

### pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my-aiperf-plugins"
version = "0.1.0"
description = "Custom AIPerf plugins for my use case"
requires-python = ">=3.10"
dependencies = [
    "aiperf",
]

[project.entry-points."aiperf.plugins"]
my-plugins = "my_plugins:plugins.yaml"

[tool.hatch.build.targets.wheel]
packages = ["src/my_plugins"]
```

The key part is the `[project.entry-points."aiperf.plugins"]` section - this tells AIPerf where to find your plugin manifest.

### src/my_plugins/__init__.py

```python
"""My custom AIPerf plugins."""
```

### src/my_plugins/endpoints/__init__.py

```python
"""Custom endpoint implementations."""

from my_plugins.endpoints.echo_endpoint import EchoEndpoint

__all__ = ["EchoEndpoint"]
```

## Step 2: Create the Endpoint Class

### src/my_plugins/endpoints/echo_endpoint.py

Your endpoint needs two methods: `format_payload()` and `parse_response()`.

```python
"""Echo endpoint for demonstration purposes."""
from __future__ import annotations
from typing import Any

from aiperf.common.models import ParsedResponse, RequestInfo, TextResponseData
from aiperf.common.protocols import InferenceServerResponse
from aiperf.endpoints.base_endpoint import BaseEndpoint


class EchoEndpoint(BaseEndpoint):
    """Echo endpoint that sends text and receives it back."""

    # ─────────────────────────────────────────────────────────────────────────
    # REQUIRED: Format outgoing request
    # ─────────────────────────────────────────────────────────────────────────
    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        turn = request_info.turns[-1]
        return {
            "text": turn.texts[0].contents[0] if turn.texts else "",
            "model": turn.model or self.model_endpoint.primary_model_name,
            "max_tokens": turn.max_tokens,
            "stream": self.model_endpoint.endpoint.streaming,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # REQUIRED: Parse incoming response
    # ─────────────────────────────────────────────────────────────────────────
    def parse_response(self, response: InferenceServerResponse) -> ParsedResponse | None:
        if json_obj := response.get_json():
            if text := json_obj.get("echo") or json_obj.get("text"):
                return ParsedResponse(perf_ns=response.perf_ns, data=TextResponseData(text=text))
            # Fallback: auto-detect common response formats
            if data := self.auto_detect_and_extract(json_obj):
                return ParsedResponse(perf_ns=response.perf_ns, data=data)
        if text := response.get_text():
            return ParsedResponse(perf_ns=response.perf_ns, data=TextResponseData(text=text))
        return None
```

> **What's happening**: `format_payload()` converts AIPerf's `RequestInfo` into your API's format. `parse_response()` extracts the response text into a `ParsedResponse`.

## Step 3: Create the Plugin Manifest

### src/my_plugins/plugins.yaml

```yaml
schema_version: "1.0"

# Register your endpoint
# Note: Package metadata (name, version, author) comes from pyproject.toml,
# not from this file. AIPerf reads it via importlib.metadata.
endpoint:
  echo:
    class: my_plugins.endpoints.echo_endpoint:EchoEndpoint
    description: |
      Echo endpoint for testing. Sends text to an Echo API and receives it back.
      Useful for testing connectivity and basic benchmarking.
    metadata:
      endpoint_path: /echo
      supports_streaming: true
      produces_tokens: true
      tokenizes_input: true
      metrics_title: Echo Metrics
```

## Step 4: Install Your Plugin

From your plugin directory:

```bash
uv pip install -e .
```

## Step 5: Verify Installation

Check that AIPerf discovers your plugin:

```bash
# List all plugins - your echo endpoint should appear
aiperf plugins endpoint

# View details about your endpoint
aiperf plugins endpoint echo

# Validate your plugin
aiperf plugins --validate
```

You should see output like:

```
Endpoint Types
┌──────────────┬──────────────────────────────────────────────────────────────┐
│ Type         │ Description                                                  │
├──────────────┼──────────────────────────────────────────────────────────────┤
│ chat         │ OpenAI Chat Completions endpoint...                          │
│ echo         │ Echo endpoint for testing. Sends text to an Echo API...      │
│ ...          │ ...                                                          │
└──────────────┴──────────────────────────────────────────────────────────────┘
```

## Step 6: Use Your Plugin

Now you can use your endpoint with AIPerf:

```bash
# Basic usage
aiperf profile \
  --model echo-model \
  --url http://localhost:8000/echo \
  --endpoint-type echo \
  --synthetic-input-tokens-mean 100 \
  --request-count 10

# With custom configuration
aiperf profile \
  --model echo-model \
  --url http://localhost:8000/echo \
  --endpoint-type echo \
  --extra-inputs echo_prefix:"[ECHO] " \
  --synthetic-input-tokens-mean 100 \
  --concurrency 4 \
  --request-count 100
```

## Step 7: Add Tests

### tests/test_echo_endpoint.py

```python
"""Tests for the Echo endpoint."""
import pytest
from my_plugins.endpoints.echo_endpoint import EchoEndpoint


class TestEchoEndpoint:
    def test_format_payload(self, mock_model_endpoint, mock_request_info):
        endpoint = EchoEndpoint(model_endpoint=mock_model_endpoint)
        payload = endpoint.format_payload(mock_request_info)
        assert "text" in payload and "model" in payload

    def test_parse_response(self, mock_model_endpoint, mock_response):
        endpoint = EchoEndpoint(model_endpoint=mock_model_endpoint)
        result = endpoint.parse_response(mock_response)
        assert result is not None and result.data.text
```

> **Fixtures**: Create `conftest.py` with `mock_model_endpoint`, `mock_request_info`, and `mock_response` fixtures. See AIPerf's test utilities for examples.

## Understanding the Code

### Component Summary

| Component | What It Does | You Provide |
|-----------|--------------|-------------|
| `BaseEndpoint` | Logging, `auto_detect_and_extract()`, config access | Inherit from it |
| `format_payload()` | Converts `RequestInfo` → API request | Your API format |
| `parse_response()` | Converts API response → `ParsedResponse` | Your parsing logic |

### Data Flow

```
RequestInfo.turns[-1]  →  format_payload()  →  HTTP Request  →  Your API
                                                                    ↓
ParsedResponse         ←  parse_response()  ←  HTTP Response ←────┘
```

### Response Types

| Type | Use Case | Key Field |
|------|----------|-----------|
| `TextResponseData` | LLM completions | `text: str` |
| `EmbeddingResponseData` | Embeddings | `embeddings: list[list[float]]` |
| `RankingsResponseData` | Reranking | `rankings: list[dict[str, Any]]` |

### Metadata Fields

| Field | Required | Purpose |
|-------|----------|---------|
| `endpoint_path` | Yes | Default API path (e.g., `/v1/chat/completions`) |
| `supports_streaming` | Yes | SSE streaming support |
| `produces_tokens` | Yes | Enables token metrics |
| `tokenizes_input` | Yes | Enables input tokenization |
| `metrics_title` | No | Dashboard display name |

## Next Steps

| Goal | Action |
|------|--------|
| **Multiple endpoints** | Add more entries under `endpoint:` in `plugins.yaml` |
| **Other plugin types** | Use same pattern for `timing_strategy`, `data_exporter`, `dataset_composer` |
| **Publish** | `python -m build && twine upload dist/*` to PyPI |

## Troubleshooting

### Plugin not found

```
TypeNotFoundError: Type 'echo' not found for category 'endpoint'.
```

**Solutions:**
1. Ensure `uv pip install -e .` completed successfully
2. Check the entry point in `pyproject.toml` matches your package structure
3. Run `aiperf plugins --validate` to check for errors

### Import errors

```
ImportError: Failed to import module for endpoint:echo
```

**Solutions:**
1. Verify the class path format: `module.path:ClassName`
2. Check all imports in your endpoint file work: `python -c "from my_plugins.endpoints.echo_endpoint import EchoEndpoint"`
3. Ensure all dependencies are installed

### Response parsing fails

**Solutions:**
1. Use `-vv` flag to see raw responses in debug logs
2. Check that your `parse_response` handles your API's actual response format
3. Use `auto_detect_and_extract()` as a fallback for unknown formats

## Reference

- [Plugin System Documentation](./plugin-system.md) - Complete plugin system reference
- [Template Endpoint Tutorial](../tutorials/template-endpoint.md) - Using templates for custom payloads
