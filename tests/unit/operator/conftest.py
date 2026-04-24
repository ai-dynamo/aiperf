# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for operator module tests.

Heavy lifting lives in ``tests.harness.operator``; this file exposes those
builders as pytest fixtures.
"""

from typing import Any

import pytest

from tests.harness.operator import (
    build_full_aiperfjob_spec,
    build_high_concurrency_spec,
    build_minimal_aiperfjob_spec,
    build_mock_aiohttp_session,
    build_mock_http_response,
    build_progress_response_complete,
    build_progress_response_running,
    build_progress_response_with_error,
    build_sample_body,
    build_sample_conditions_list,
)

# =============================================================================
# Kubernetes Resource Body Fixtures
# =============================================================================


@pytest.fixture
def sample_body() -> dict[str, Any]:
    """Create a sample Kubernetes resource body for event testing."""
    return build_sample_body()


# =============================================================================
# AIPerfJob Spec Fixtures (flat — no userConfig wrapper)
# =============================================================================


@pytest.fixture
def minimal_aiperfjob_spec() -> dict[str, Any]:
    """Create a minimal flat AIPerfJob spec for testing."""
    return build_minimal_aiperfjob_spec()


@pytest.fixture
def full_aiperfjob_spec() -> dict[str, Any]:
    """Create a complete flat AIPerfJob spec with all options."""
    return build_full_aiperfjob_spec()


@pytest.fixture
def aiperfjob_spec_high_concurrency() -> dict[str, Any]:
    """Create a flat AIPerfJob spec with high concurrency for worker scaling tests."""
    return build_high_concurrency_spec()


# =============================================================================
# Progress API Response Fixtures
# =============================================================================


@pytest.fixture
def progress_api_response_running() -> dict[str, Any]:
    """Create a progress API response for a running job."""
    return build_progress_response_running()


@pytest.fixture
def progress_api_response_complete() -> dict[str, Any]:
    """Create a progress API response for a completed job."""
    return build_progress_response_complete()


@pytest.fixture
def progress_api_response_with_error() -> dict[str, Any]:
    """Create a progress API response with an error."""
    return build_progress_response_with_error()


# =============================================================================
# Condition Fixtures
# =============================================================================


@pytest.fixture
def sample_conditions_list() -> list[dict[str, Any]]:
    """Create a sample conditions list from Kubernetes status."""
    return build_sample_conditions_list()


# =============================================================================
# Mock HTTP Session Fixtures
# =============================================================================


@pytest.fixture
def mock_aiohttp_session():
    """Create a mock aiohttp ClientSession."""
    return build_mock_aiohttp_session()


@pytest.fixture
def mock_http_response():
    """Create a mock HTTP response factory."""
    return build_mock_http_response
