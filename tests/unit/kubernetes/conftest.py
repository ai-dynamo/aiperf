# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for Kubernetes module tests.

Heavy lifting lives in ``tests.harness.k8s``; this file exposes those
builders as pytest fixtures and re-exports helper functions for backward
compatibility with direct imports from test files.
"""

from typing import Any

import pytest

from tests.harness.k8s import (
    build_completed_jobset,
    build_failed_jobset,
    build_mock_api,
    build_pending_pod,
    build_running_jobset,
    build_sample_config,
    build_sample_jobset,
    build_sample_pod,
    build_sample_pod_template,
    build_sample_run,
    build_succeeded_pod,
    create_api_exception,
    create_jobset_list_response,
    patch_api_accessors,
)

# Re-export helper functions so existing ``from tests.unit.kubernetes.conftest import ...``
# statements continue to work without changes.
__all__ = [
    "create_api_exception",
    "create_jobset_list_response",
    "patch_api_accessors",
]


# =============================================================================
# Mock ApiClient Fixtures
# =============================================================================


@pytest.fixture
def mock_api():
    """Mock kubernetes_asyncio ApiClient."""
    return build_mock_api()


# =============================================================================
# Sample Data Fixtures
# =============================================================================


@pytest.fixture
def sample_jobset() -> dict[str, Any]:
    """Create a sample JobSet dict for testing."""
    return build_sample_jobset()


@pytest.fixture
def sample_running_jobset(sample_jobset) -> dict[str, Any]:
    """Create a sample running JobSet."""
    return build_running_jobset(sample_jobset)


@pytest.fixture
def sample_completed_jobset(sample_jobset) -> dict[str, Any]:
    """Create a sample completed JobSet."""
    return build_completed_jobset(sample_jobset)


@pytest.fixture
def sample_failed_jobset(sample_jobset) -> dict[str, Any]:
    """Create a sample failed JobSet."""
    return build_failed_jobset(sample_jobset)


@pytest.fixture
def sample_pod() -> dict[str, Any]:
    """Create a sample Pod dict (raw K8s API format) for testing."""
    return build_sample_pod()


@pytest.fixture
def sample_pending_pod(sample_pod) -> dict[str, Any]:
    """Create a sample pending Pod."""
    return build_pending_pod(sample_pod)


@pytest.fixture
def sample_succeeded_pod(sample_pod) -> dict[str, Any]:
    """Create a sample succeeded Pod."""
    return build_succeeded_pod(sample_pod)


# =============================================================================
# Config Fixtures
# =============================================================================


@pytest.fixture
def sample_config():
    """Create a minimal AIPerfConfig for testing."""
    return build_sample_config()


@pytest.fixture
def sample_run(sample_config):
    """Create a minimal BenchmarkRun for testing."""
    return build_sample_run(sample_config)


@pytest.fixture
def sample_pod_template():
    """Create a sample PodTemplateConfig for testing."""
    return build_sample_pod_template()
