# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures and helpers for endpoint tests.

The pure builders (config / run / request-info / response) live in
``tests.harness.endpoint_helpers`` so tests outside this package (e.g.
``tests/unit/workers``) can reuse them without a cross-package conftest
import. They are re-exported here so existing
``from tests.unit.endpoints.conftest import ...`` sites keep working.
"""

from unittest.mock import MagicMock, patch

import pytest

from tests.harness.endpoint_helpers import (
    _MINIMAL_CONFIG_KWARGS,
    _config_from_model_endpoint,
    _wrap_model_endpoint,
    _wrap_run,
    create_config,
    create_endpoint_with_mock_transport,
    create_mock_response,
    create_model_endpoint,
    create_request_info,
)

__all__ = [
    "_MINIMAL_CONFIG_KWARGS",
    "_config_from_model_endpoint",
    "_wrap_model_endpoint",
    "_wrap_run",
    "create_config",
    "create_endpoint_with_mock_transport",
    "create_mock_response",
    "create_model_endpoint",
    "create_request_info",
]


@pytest.fixture
def mock_transport_plugin():
    """Mock the plugin transport class to return a MagicMock."""
    with patch("aiperf.plugin.plugins.get_class") as mock:
        mock.return_value = MagicMock
        yield mock
