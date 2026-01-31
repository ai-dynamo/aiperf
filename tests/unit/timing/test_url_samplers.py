# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType, URLSelectionStrategy
from aiperf.timing.url_samplers import RoundRobinURLSampler


class TestRoundRobinURLSampler:
    """Unit tests for RoundRobinURLSampler."""

    def test_init_with_single_url(self):
        """Single URL should work without issues."""
        sampler = RoundRobinURLSampler(urls=["http://localhost:8000"])
        assert sampler.urls == ["http://localhost:8000"]

    def test_init_with_multiple_urls(self):
        """Multiple URLs should be stored correctly."""
        urls = ["http://server1:8000", "http://server2:8000", "http://server3:8000"]
        sampler = RoundRobinURLSampler(urls=urls)
        assert sampler.urls == urls

    def test_init_with_empty_urls_raises(self):
        """Empty URLs list should raise ValueError."""
        with pytest.raises(ValueError, match="URLs list cannot be empty"):
            RoundRobinURLSampler(urls=[])

    def test_next_url_index_single_url(self):
        """Single URL should always return index 0."""
        sampler = RoundRobinURLSampler(urls=["http://localhost:8000"])
        for _ in range(10):
            assert sampler.next_url_index() == 0

    def test_next_url_index_round_robin_order(self):
        """Multiple URLs should cycle in round-robin order."""
        urls = ["http://server1:8000", "http://server2:8000", "http://server3:8000"]
        sampler = RoundRobinURLSampler(urls=urls)

        # First cycle
        assert sampler.next_url_index() == 0
        assert sampler.next_url_index() == 1
        assert sampler.next_url_index() == 2

        # Second cycle (wrap-around)
        assert sampler.next_url_index() == 0
        assert sampler.next_url_index() == 1
        assert sampler.next_url_index() == 2

    def test_next_url_index_wrap_around(self):
        """Index should wrap around correctly at list boundary."""
        urls = ["http://server1:8000", "http://server2:8000"]
        sampler = RoundRobinURLSampler(urls=urls)

        # Collect many indices
        indices = [sampler.next_url_index() for _ in range(10)]

        # Should alternate between 0 and 1
        expected = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        assert indices == expected

    def test_kwargs_ignored(self):
        """Extra kwargs should be ignored for factory compatibility."""
        sampler = RoundRobinURLSampler(
            urls=["http://localhost:8000"], extra_arg=True, another_arg="test"
        )
        assert sampler.urls == ["http://localhost:8000"]


class TestURLSelectionStrategyPlugin:
    """Unit tests for URL selection strategy plugin system."""

    def test_plugin_creates_round_robin(self):
        """Plugin system should create RoundRobinURLSampler for ROUND_ROBIN strategy."""
        urls = ["http://server1:8000", "http://server2:8000"]
        SamplerClass = plugins.get_class(
            PluginType.URL_SELECTION_STRATEGY, URLSelectionStrategy.ROUND_ROBIN
        )
        sampler = SamplerClass(urls=urls)
        assert isinstance(sampler, RoundRobinURLSampler)
        assert sampler.urls == urls

    def test_plugin_creates_round_robin_by_string(self):
        """Plugin system should accept string strategy name."""
        urls = ["http://server1:8000", "http://server2:8000"]
        SamplerClass = plugins.get_class(
            PluginType.URL_SELECTION_STRATEGY, "round_robin"
        )
        sampler = SamplerClass(urls=urls)
        assert isinstance(sampler, RoundRobinURLSampler)
        assert sampler.urls == urls

    def test_plugin_with_enum_value(self):
        """Plugin system should accept URLSelectionStrategy enum value."""
        urls = ["http://server1:8000"]
        SamplerClass = plugins.get_class(
            PluginType.URL_SELECTION_STRATEGY, URLSelectionStrategy.ROUND_ROBIN
        )
        sampler = SamplerClass(urls=urls)
        assert isinstance(sampler, RoundRobinURLSampler)
        assert sampler.urls == urls
