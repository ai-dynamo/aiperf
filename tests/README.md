<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# AIPerf Tests Directory

This directory contains multiple test suites for the AIPerf project, organized into multiple categories and supporting infrastructure.

## Directory Structure

The directory structure is as follows:

```
tests/
├── aiperf_mock_server/
├── ci/
│   └── test_docs_end_to_end/
├── integration/
└── unit/
    ├── server/
    └── ...
```

## Test Suites

### [`aiperf_mock_server/`](aiperf_mock_server/)
A standalone mock server application used for testing AIPerf against simulated endpoints. This is a separate Python package with its own `pyproject.toml` and can be installed independently. See [`aiperf_mock_server/README.md`](aiperf_mock_server/README.md) for detailed information.

### [`ci/`](ci/)
CI/CD-specific test utilities and scripts. These tests are designed to run exclusively on Gitlab and have access to real GPU servers.

### [`ci/test_docs_end_to_end/`](ci/test_docs_end_to_end/)
End-to-end tests for the documentation, running the AIPerf commands from the documentation and verifying the output against actual Dynamo and vLLM servers. These tests are designed to run exclusively on Gitlab and have access to real GPU servers.

### [`integration/`](integration/)
End-to-end integration tests that verify complete workflows and interactions between components using real aiperf command lines. These tests are designed to run exclusively with the mock server and are executed on GitHub Actions. See [`integration/README.md`](integration/README.md) for detailed information.

### [`unit/`](unit/)
Unit tests for individual modules and components. The directory structure mirrors the [`src/aiperf/`](../src/aiperf/) directory as closely as possible, with each subdirectory containing tests for the corresponding source module.

### [`unit/server/`](unit/server/)
Unit tests exclusively for the aiperf-mock-server. See [`aiperf_mock_server/README.md`](aiperf_mock_server/README.md) for detailed information.

## Auto-Selected Markers

Several test suites are excluded by default via `-m 'not k8s and not gpu ...'` in `pyproject.toml` `addopts`. When you target one of these paths directly, the corresponding markers are automatically enabled so tests run instead of being silently skipped.

| Path | Markers enabled |
|---|---|
| `tests/kubernetes/` | `k8s`, `gpu`, `vllm`, `dynamo` |
| `tests/kubernetes/gpu/` | `k8s`, `gpu`, `vllm`, `dynamo` |
| `tests/kubernetes/gpu/vllm/` | `k8s`, `gpu`, `vllm` |
| `tests/kubernetes/gpu/dynamo/` | `k8s`, `gpu`, `dynamo` |
| `tests/integration/` | `integration` |
| `tests/component_integration/` | `component_integration` |

Matching is hierarchical: targeting a child path enables its own markers plus all parent markers. Targeting a parent path enables markers for all children underneath it.

```bash
# These just work -- no need to pass -m manually:
uv run pytest tests/kubernetes/
uv run pytest tests/kubernetes/gpu/vllm/test_foo.py
uv run pytest tests/integration/

# Equivalent to manually overriding the marker expression:
uv run pytest tests/kubernetes/ -m 'not integration and not component_integration'
```

The logic lives in `tests/conftest.py::pytest_configure`. To add a new auto-selected path, add an entry to `_PATH_MARKER_MAP`.
