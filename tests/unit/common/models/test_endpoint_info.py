# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for EndpointInfo multi-URL support."""

import pytest

from aiperf.common.config import EndpointDefaults
from aiperf.common.models.model_endpoint_info import EndpointInfo


class TestEndpointInfoMultiURL:
    """Tests for EndpointInfo multi-URL support."""

    def test_single_url_default(self):
        """Default should be single URL."""
        info = EndpointInfo()
        assert info.base_urls == [EndpointDefaults.URL]
        assert info.base_url == EndpointDefaults.URL

    def test_single_url_custom(self):
        """Custom single URL should work."""
        info = EndpointInfo(base_urls=["http://custom-server:8000"])
        assert info.base_urls == ["http://custom-server:8000"]
        assert info.base_url == "http://custom-server:8000"

    def test_multiple_urls(self):
        """Multiple URLs should be stored correctly."""
        urls = ["http://server1:8000", "http://server2:8000", "http://server3:8000"]
        info = EndpointInfo(base_urls=urls)
        assert info.base_urls == urls
        assert info.base_url == "http://server1:8000"  # Backward compat

    def test_base_urls_must_have_at_least_one(self):
        """base_urls must have at least one entry."""
        with pytest.raises(ValueError):
            EndpointInfo(base_urls=[])


class TestEndpointInfoGetUrl:
    """Tests for EndpointInfo.get_url() method."""

    def test_get_url_none_returns_first(self):
        """get_url(None) should return first URL."""
        urls = ["http://server1:8000", "http://server2:8000"]
        info = EndpointInfo(base_urls=urls)
        assert info.get_url(None) == "http://server1:8000"

    def test_get_url_index_0(self):
        """get_url(0) should return first URL."""
        urls = ["http://server1:8000", "http://server2:8000"]
        info = EndpointInfo(base_urls=urls)
        assert info.get_url(0) == "http://server1:8000"

    def test_get_url_index_1(self):
        """get_url(1) should return second URL."""
        urls = ["http://server1:8000", "http://server2:8000"]
        info = EndpointInfo(base_urls=urls)
        assert info.get_url(1) == "http://server2:8000"

    def test_get_url_wrap_around(self):
        """get_url should wrap around with modulo."""
        urls = ["http://server1:8000", "http://server2:8000", "http://server3:8000"]
        info = EndpointInfo(base_urls=urls)
        # Index 3 should wrap to 0
        assert info.get_url(3) == "http://server1:8000"
        # Index 4 should wrap to 1
        assert info.get_url(4) == "http://server2:8000"
        # Index 5 should wrap to 2
        assert info.get_url(5) == "http://server3:8000"

    def test_get_url_single_url_always_returns_same(self):
        """Single URL should always return the same URL regardless of index."""
        info = EndpointInfo(base_urls=["http://localhost:8000"])
        for i in range(10):
            assert info.get_url(i) == "http://localhost:8000"
