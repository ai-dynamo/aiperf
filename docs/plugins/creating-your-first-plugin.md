<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Creating Your First AIPerf Plugin

This tutorial walks you through creating a custom AIPerf endpoint plugin from scratch. By the end, you'll have a working plugin package that can benchmark any custom API.

> [!TIP]
> **Contributing directly to AIPerf?** The endpoint class (Step 2) and manifest format (Step 3) are the same, but you can skip the external packaging:
> - Add your class under `src/aiperf/` instead of a separate package
> - Register it in the existing `src/aiperf/plugin/plugins.yaml` instead of creating a new one
> - Skip: Project Structure, Step 1 (pyproject.toml/entry points), Step 4 (install)

## What You'll Build

We'll create a plugin for a hypothetical "Echo API" that returns the input text with some metadata. This simple example demonstrates all the core concepts you need to build more complex plugins.

## Prerequisites

- Python 3.10+
- AIPerf installed (`pip install aiperf`)
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

```text
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

```bash
PKG=my-aiperf-plugins
SRC=$PKG/src/my_plugins

mkdir -p $SRC/endpoints $PKG/tests
touch $PKG/pyproject.toml \
      $PKG/echo_server.py \
      $SRC/__init__.py \
      $SRC/plugins.yaml \
      $SRC/endpoints/__init__.py \
      $SRC/endpoints/echo_endpoint.py \
      $PKG/tests/test_echo_endpoint.py
tree $PKG
cd $PKG
```

You should see:

```text
my-aiperf-plugins/
├── echo_server.py
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

Now fill in each file in the steps below.

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

from aiperf.common.models import ParsedResponse, RequestInfo, TextResponseData, InferenceServerResponse
from aiperf.endpoints.base_endpoint import BaseEndpoint


class EchoEndpoint(BaseEndpoint):
    """Echo endpoint that sends text and receives it back."""

    # ─────────────────────────────────────────────────────────────────────────
    # REQUIRED: Format outgoing request
    # ─────────────────────────────────────────────────────────────────────────
    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        turn = request_info.turns[-1]
        model_endpoint = request_info.model_endpoint
        texts = [content for text in turn.texts for content in text.contents if content]
        return {
            "text": texts[0] if texts else "",
            "model": turn.model or model_endpoint.primary_model_name,
            "max_tokens": turn.max_tokens,
            "stream": model_endpoint.endpoint.streaming,
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
# yaml-language-server: $schema=https://raw.githubusercontent.com/ai-dynamo/aiperf/refs/heads/main/src/aiperf/plugin/schema/plugins.schema.json
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

From your plugin directory, install into the **same Python environment** where AIPerf is installed. AIPerf discovers plugins via entry points, which only works when both packages share the same environment.

```bash
pip install -e .
```

You should see:

```text
Successfully installed my-aiperf-plugins-0.1.0
```

> **Important**: If you use `uv`, virtual environments, or conda, make sure you activate the environment where AIPerf is installed before running `pip install`.

## Step 5: Verify Installation

Confirm both packages are installed in the same environment:

```bash
pip show aiperf my-aiperf-plugins
```

You should see both packages listed in the same environment:

```text
Name: aiperf
Version: 0.6.0
Location: ...
Requires: ...
Required-by: my-aiperf-plugins
---
Name: my-aiperf-plugins
Version: 0.1.0
Location: ...
Requires: aiperf
Required-by:
```

Check that AIPerf discovers your plugin:

```bash
# List all plugins - your echo endpoint should appear
aiperf plugins endpoint
```

You should see your plugin in the table:

```text
Endpoint Types
┌──────────────┬──────────────────────────────────────────────────────────────┐
│ Type         │ Description                                                  │
├──────────────┼──────────────────────────────────────────────────────────────┤
│ chat         │ OpenAI Chat Completions endpoint...                          │
│ ...          │ ...                                                          │
│ echo         │ Echo endpoint for testing. Sends text to an Echo API...      │
└──────────────┴──────────────────────────────────────────────────────────────┘
```

```bash
# View details about your endpoint
aiperf plugins endpoint echo
```

You should see:

```text
╭──────────────────────────── endpoint:echo ─────────────────────────────╮
│ Type: echo                                                             │
│ Category: endpoint                                                     │
│ Package: my-plugins                                                    │
│ Class: my_plugins.endpoints.echo_endpoint:EchoEndpoint                 │
│                                                                        │
│ Echo endpoint for testing. Sends text to an Echo API and receives it   │
│ back. Useful for testing connectivity and basic benchmarking.          │
╰────────────────────────────────────────────────────────────────────────╯
```

```bash
# Validate your plugin
aiperf plugins --validate
```

You should see:

```text
Validating plugins...

✓ Class paths

All checks passed
```

## Step 6: Create a Test Server

To test your plugin end-to-end, create a minimal Echo API server. Save this as `echo_server.py` in your project root:

```python
"""Minimal Echo API server for testing the EchoEndpoint plugin."""
from __future__ import annotations

import asyncio

import cyclopts
import orjson
import uvicorn
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse, StreamingResponse

app = FastAPI()
cli = cyclopts.App()

@app.post("/echo", response_model=None)
async def echo(body: dict) -> ORJSONResponse | StreamingResponse:
    echo_text = f"[echo] {body.get('text', '')}"
    model = body.get("model", "echo-model")

    if not body.get("stream"):
        return ORJSONResponse({"echo": echo_text, "model": model})

    async def sse():
        for i, word in enumerate(echo_text.split()):
            chunk = orjson.dumps({"echo": word if i == 0 else f" {word}", "model": model})
            yield b"data: " + chunk + b"\n\n"
            await asyncio.sleep(0.02)
        yield b"data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


@cli.default
def main(host: str = "127.0.0.1", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    cli()
```

Start the server:

```bash
pip install fastapi uvicorn orjson cyclopts
python echo_server.py &
```

You should see:

```text
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

## Step 7: Use Your Plugin

With the test server running, use your endpoint with AIPerf:

```bash
# Basic usage (endpoint_path: /echo from metadata is appended automatically)
aiperf profile \
  --model echo-model \
  --url http://localhost:8000 \
  --endpoint-type echo \
  --tokenizer gpt2 \
  --synthetic-input-tokens-mean 100 \
  --request-count 10

# With custom configuration
aiperf profile \
  --model echo-model \
  --url http://localhost:8000 \
  --endpoint-type echo \
  --tokenizer gpt2 \
  --extra-inputs echo_prefix:"[ECHO] " \
  --synthetic-input-tokens-mean 100 \
  --concurrency 4 \
  --request-count 100
```

You should see:

```text
                            NVIDIA AIPerf | Echo Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━┓
┃                           Metric ┃       avg ┃    min ┃    max ┃    p99 ┃  std ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━┩
│             Request Latency (ms) │      2.05 │   0.29 │  15.42 │  14.18 │ 4.47 │
│  Output Sequence Length (tokens) │    104.00 │ 104.00 │ 104.00 │ 104.00 │ 0.00 │
│   Input Sequence Length (tokens) │    100.00 │ 100.00 │ 100.00 │ 100.00 │ 0.00 │
│          Output Token Throughput │ 40,850.61 │    N/A │    N/A │    N/A │  N/A │
│                     (tokens/sec) │           │        │        │        │      │
│               Request Throughput │    392.79 │    N/A │    N/A │    N/A │  N/A │
│                   (requests/sec) │           │        │        │        │      │
│         Request Count (requests) │     10.00 │    N/A │    N/A │    N/A │  N/A │
└──────────────────────────────────┴───────────┴────────┴────────┴────────┴──────┘
```

## Step 8: Add Tests

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

```text
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
| `endpoint_path` | Yes (nullable) | Default API path (e.g., `/v1/chat/completions`) |
| `supports_streaming` | Yes | SSE streaming support |
| `produces_tokens` | Yes | Enables token metrics |
| `tokenizes_input` | Yes | Enables input tokenization |
| `metrics_title` | Yes | Dashboard display name (nullable) |

## Next Steps

| Goal | Action |
|------|--------|
| **Multiple endpoints** | Add more entries under `endpoint:` in `plugins.yaml` |
| **Other plugin types** | Use same pattern for `timing_strategy`, `data_exporter`, `dataset_composer` |
| **Publish** | `python -m build && twine upload dist/*` to PyPI |

## Troubleshooting

### Plugin not found

```text
TypeNotFoundError: Type 'echo' not found for category 'endpoint'.
```

**Solutions:**
1. Ensure `pip install -e .` completed successfully
2. Check the entry point in `pyproject.toml` matches your package structure
3. Run `aiperf plugins --validate` to check for errors

### Import errors

```text
ImportError: Failed to import module for endpoint:echo from 'my_plugins.endpoints.echo_endpoint:EchoEndpoint'
Reason: ...
Tip: Check that the module is installed and importable
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
