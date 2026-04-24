# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for EndpointConfig URL support (replaces old EndpointInfo tests)."""

import pytest

from aiperf.config.endpoint import EndpointConfig


class TestEndpointConfigMultiURL:
    """Tests for EndpointConfig multi-URL support."""

    def test_single_url_custom(self):
        """Custom single URL should work."""
        config = EndpointConfig(urls=["http://custom-server:8000"])
        assert config.urls == ["http://custom-server:8000"]

    def test_multiple_urls(self):
        """Multiple URLs should be stored correctly."""
        urls = ["http://server1:8000", "http://server2:8000", "http://server3:8000"]
        config = EndpointConfig(urls=urls)
        assert config.urls == urls

    def test_urls_must_have_at_least_one(self):
        """urls must have at least one entry."""
        with pytest.raises(ValueError):
            EndpointConfig(urls=[])
