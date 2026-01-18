# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for plot cache module.

Tests for caching functionality in aiperf.plot.dashboard.cache module.
"""

import time

import plotly.graph_objects as go
import pytest

import aiperf.plot.dashboard.cache as cache_module
from aiperf.plot.constants import PlotTheme
from aiperf.plot.dashboard.cache import (
    CachedPlot,
    CacheKey,
    PlotCache,
    compute_config_hash,
    compute_runs_hash,
    get_plot_cache,
)


@pytest.fixture
def sample_figure() -> go.Figure:
    """Create a sample Plotly figure for testing."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))
    return fig


@pytest.fixture
def cache_key_dark() -> CacheKey:
    """Create a sample cache key with dark theme."""
    return CacheKey(
        plot_id="test-plot",
        config_hash="abc123def456",
        runs_hash="runs123",
        theme=PlotTheme.DARK,
    )


@pytest.fixture
def cache_key_light() -> CacheKey:
    """Create a sample cache key with light theme."""
    return CacheKey(
        plot_id="test-plot",
        config_hash="abc123def456",
        runs_hash="runs123",
        theme=PlotTheme.LIGHT,
    )


@pytest.fixture
def fresh_plot_cache() -> PlotCache:
    """Create a fresh PlotCache instance with small size for testing."""
    return PlotCache(max_plots_per_theme=5)


@pytest.fixture(autouse=True)
def reset_global_cache():
    """Reset the global cache singleton before and after each test."""
    cache_module._PLOT_CACHE = None
    yield
    cache_module._PLOT_CACHE = None


class TestCacheKey:
    """Tests for CacheKey dataclass."""

    def test_frozen_immutable(self, cache_key_dark):
        """Test that CacheKey is frozen and immutable."""
        with pytest.raises(AttributeError):
            cache_key_dark.plot_id = "new-id"

    def test_equality_for_same_values(self):
        """Test that CacheKeys with same values are equal."""
        key1 = CacheKey(
            plot_id="plot1",
            config_hash="hash1",
            runs_hash="runs1",
            theme=PlotTheme.DARK,
        )
        key2 = CacheKey(
            plot_id="plot1",
            config_hash="hash1",
            runs_hash="runs1",
            theme=PlotTheme.DARK,
        )
        assert key1 == key2

    def test_inequality_for_different_themes(self):
        """Test that CacheKeys with different themes are not equal."""
        key_dark = CacheKey(
            plot_id="plot1",
            config_hash="hash1",
            runs_hash="runs1",
            theme=PlotTheme.DARK,
        )
        key_light = CacheKey(
            plot_id="plot1",
            config_hash="hash1",
            runs_hash="runs1",
            theme=PlotTheme.LIGHT,
        )
        assert key_dark != key_light

    def test_inequality_for_different_plot_id(self):
        """Test that CacheKeys with different plot_id are not equal."""
        key1 = CacheKey(
            plot_id="plot1",
            config_hash="hash1",
            runs_hash="runs1",
            theme=PlotTheme.DARK,
        )
        key2 = CacheKey(
            plot_id="plot2",
            config_hash="hash1",
            runs_hash="runs1",
            theme=PlotTheme.DARK,
        )
        assert key1 != key2

    def test_hashable_can_be_dict_key(self, cache_key_dark, sample_figure):
        """Test that CacheKey can be used as dictionary key."""
        cache_dict = {}
        cache_dict[cache_key_dark] = sample_figure
        assert cache_dict[cache_key_dark] == sample_figure


class TestCachedPlot:
    """Tests for CachedPlot dataclass."""

    def test_stores_figure(self, sample_figure):
        """Test that CachedPlot stores the figure."""
        cached = CachedPlot(figure=sample_figure)
        assert cached.figure == sample_figure

    def test_has_timestamps(self, sample_figure):
        """Test that CachedPlot has created_at and last_accessed timestamps."""
        cached = CachedPlot(figure=sample_figure)
        assert isinstance(cached.created_at, float)
        assert isinstance(cached.last_accessed, float)
        assert cached.created_at > 0
        assert cached.last_accessed > 0

    def test_touch_updates_last_accessed(self, sample_figure):
        """Test that touch() updates last_accessed timestamp."""
        cached = CachedPlot(figure=sample_figure)
        original_last_accessed = cached.last_accessed
        time.sleep(0.01)
        cached.touch()
        assert cached.last_accessed > original_last_accessed

    def test_touch_does_not_change_created_at(self, sample_figure):
        """Test that touch() does NOT change created_at timestamp."""
        cached = CachedPlot(figure=sample_figure)
        original_created_at = cached.created_at
        time.sleep(0.01)
        cached.touch()
        assert cached.created_at == original_created_at


class TestPlotCacheInit:
    """Tests for PlotCache initialization."""

    def test_init_with_default_max_plots(self):
        """Test __init__ with default max_plots_per_theme=50."""
        cache = PlotCache()
        assert cache._max_plots_per_theme == 50
        assert len(cache._cache) == 0
        assert cache._hits == 0
        assert cache._misses == 0

    def test_init_with_custom_max_plots(self):
        """Test __init__ with custom max_plots."""
        cache = PlotCache(max_plots_per_theme=10)
        assert cache._max_plots_per_theme == 10


class TestPlotCacheBasicOperations:
    """Tests for PlotCache get and set operations."""

    def test_get_returns_none_for_missing_key(self, fresh_plot_cache, cache_key_dark):
        """Test that get() returns None for missing key."""
        result = fresh_plot_cache.get(cache_key_dark)
        assert result is None

    def test_set_and_get_store_and_retrieve_figure(
        self, fresh_plot_cache, cache_key_dark, sample_figure
    ):
        """Test that set() and get() store and retrieve figure."""
        fresh_plot_cache.set(cache_key_dark, sample_figure)
        result = fresh_plot_cache.get(cache_key_dark)
        assert result == sample_figure

    def test_get_updates_last_accessed(
        self, fresh_plot_cache, cache_key_dark, sample_figure
    ):
        """Test that get() updates last_accessed (touch)."""
        fresh_plot_cache.set(cache_key_dark, sample_figure)
        cached_plot = fresh_plot_cache._cache[cache_key_dark]
        original_last_accessed = cached_plot.last_accessed
        time.sleep(0.01)
        fresh_plot_cache.get(cache_key_dark)
        assert cached_plot.last_accessed > original_last_accessed

    def test_get_increments_hits_counter_on_hit(
        self, fresh_plot_cache, cache_key_dark, sample_figure
    ):
        """Test that get() increments hits counter on hit."""
        fresh_plot_cache.set(cache_key_dark, sample_figure)
        assert fresh_plot_cache._hits == 0
        fresh_plot_cache.get(cache_key_dark)
        assert fresh_plot_cache._hits == 1
        fresh_plot_cache.get(cache_key_dark)
        assert fresh_plot_cache._hits == 2

    def test_get_increments_misses_counter_on_miss(
        self, fresh_plot_cache, cache_key_dark
    ):
        """Test that get() increments misses counter on miss."""
        assert fresh_plot_cache._misses == 0
        fresh_plot_cache.get(cache_key_dark)
        assert fresh_plot_cache._misses == 1
        fresh_plot_cache.get(cache_key_dark)
        assert fresh_plot_cache._misses == 2

    def test_separate_storage_by_theme(
        self, fresh_plot_cache, cache_key_dark, cache_key_light
    ):
        """Test separate storage by theme (dark vs light)."""
        fig_dark = go.Figure()
        fig_dark.add_trace(go.Scatter(x=[1, 2], y=[1, 2], name="dark"))
        fig_light = go.Figure()
        fig_light.add_trace(go.Scatter(x=[3, 4], y=[3, 4], name="light"))

        fresh_plot_cache.set(cache_key_dark, fig_dark)
        fresh_plot_cache.set(cache_key_light, fig_light)

        result_dark = fresh_plot_cache.get(cache_key_dark)
        result_light = fresh_plot_cache.get(cache_key_light)

        assert result_dark == fig_dark
        assert result_light == fig_light
        assert result_dark != result_light


class TestPlotCacheInvalidation:
    """Tests for PlotCache invalidation methods."""

    def test_invalidate_plot_removes_all_versions(
        self, fresh_plot_cache, sample_figure
    ):
        """Test that invalidate_plot() removes all versions of a plot (both themes)."""
        key_dark = CacheKey(
            plot_id="plot1",
            config_hash="hash1",
            runs_hash="runs1",
            theme=PlotTheme.DARK,
        )
        key_light = CacheKey(
            plot_id="plot1",
            config_hash="hash1",
            runs_hash="runs1",
            theme=PlotTheme.LIGHT,
        )
        key_other = CacheKey(
            plot_id="plot2",
            config_hash="hash1",
            runs_hash="runs1",
            theme=PlotTheme.DARK,
        )

        fresh_plot_cache.set(key_dark, sample_figure)
        fresh_plot_cache.set(key_light, sample_figure)
        fresh_plot_cache.set(key_other, sample_figure)

        fresh_plot_cache.invalidate_plot("plot1")

        assert fresh_plot_cache.get(key_dark) is None
        assert fresh_plot_cache.get(key_light) is None
        assert fresh_plot_cache.get(key_other) == sample_figure

    def test_invalidate_plot_with_nonexistent_id(self, fresh_plot_cache):
        """Test that invalidate_plot() with nonexistent id does not error."""
        fresh_plot_cache.invalidate_plot("nonexistent")

    def test_invalidate_by_runs_keeps_matching_hash(
        self, fresh_plot_cache, sample_figure
    ):
        """Test that invalidate_by_runs() keeps matching hash, removes others."""
        key_runs1 = CacheKey(
            plot_id="plot1",
            config_hash="hash1",
            runs_hash="runs1",
            theme=PlotTheme.DARK,
        )
        key_runs2 = CacheKey(
            plot_id="plot2",
            config_hash="hash1",
            runs_hash="runs2",
            theme=PlotTheme.DARK,
        )
        key_runs1_light = CacheKey(
            plot_id="plot3",
            config_hash="hash1",
            runs_hash="runs1",
            theme=PlotTheme.LIGHT,
        )

        fresh_plot_cache.set(key_runs1, sample_figure)
        fresh_plot_cache.set(key_runs2, sample_figure)
        fresh_plot_cache.set(key_runs1_light, sample_figure)

        fresh_plot_cache.invalidate_by_runs("runs1")

        assert fresh_plot_cache.get(key_runs1) == sample_figure
        assert fresh_plot_cache.get(key_runs1_light) == sample_figure
        assert fresh_plot_cache.get(key_runs2) is None

    def test_clear_removes_all_entries_and_resets_counters(
        self, fresh_plot_cache, cache_key_dark, cache_key_light, sample_figure
    ):
        """Test that clear() removes all entries and resets counters."""
        fresh_plot_cache.set(cache_key_dark, sample_figure)
        fresh_plot_cache.set(cache_key_light, sample_figure)
        fresh_plot_cache.get(cache_key_dark)
        fresh_plot_cache.get(cache_key_light)

        assert len(fresh_plot_cache._cache) == 2
        assert fresh_plot_cache._hits == 2
        assert fresh_plot_cache._misses == 0

        fresh_plot_cache.clear()

        assert len(fresh_plot_cache._cache) == 0
        assert fresh_plot_cache._hits == 0
        assert fresh_plot_cache._misses == 0


class TestPlotCacheEviction:
    """Tests for PlotCache eviction behavior."""

    def test_eviction_when_exceeding_max_plots(self, sample_figure):
        """Test that eviction occurs when exceeding max_plots."""
        cache = PlotCache(max_plots_per_theme=3)

        for i in range(5):
            key = CacheKey(
                plot_id=f"plot{i}",
                config_hash="hash1",
                runs_hash="runs1",
                theme=PlotTheme.DARK,
            )
            cache.set(key, sample_figure)
            time.sleep(0.01)

        dark_keys = [k for k in cache._cache if k.theme == PlotTheme.DARK]
        assert len(dark_keys) == 3

    def test_eviction_is_per_theme(self, sample_figure):
        """Test that eviction is per-theme (doesn't affect other theme)."""
        cache = PlotCache(max_plots_per_theme=2)

        for i in range(3):
            key_dark = CacheKey(
                plot_id=f"plot{i}",
                config_hash="hash1",
                runs_hash="runs1",
                theme=PlotTheme.DARK,
            )
            cache.set(key_dark, sample_figure)
            time.sleep(0.01)

        key_light = CacheKey(
            plot_id="light_plot",
            config_hash="hash1",
            runs_hash="runs1",
            theme=PlotTheme.LIGHT,
        )
        cache.set(key_light, sample_figure)

        dark_keys = [k for k in cache._cache if k.theme == PlotTheme.DARK]
        light_keys = [k for k in cache._cache if k.theme == PlotTheme.LIGHT]

        assert len(dark_keys) == 2
        assert len(light_keys) == 1

    def test_lru_eviction_order(self, sample_figure):
        """Test that LRU eviction order (recently accessed kept)."""
        cache = PlotCache(max_plots_per_theme=3)

        keys = []
        for i in range(3):
            key = CacheKey(
                plot_id=f"plot{i}",
                config_hash="hash1",
                runs_hash="runs1",
                theme=PlotTheme.DARK,
            )
            keys.append(key)
            cache.set(key, sample_figure)
            time.sleep(0.01)

        time.sleep(0.01)
        cache.get(keys[0])

        time.sleep(0.01)
        new_key = CacheKey(
            plot_id="plot_new",
            config_hash="hash1",
            runs_hash="runs1",
            theme=PlotTheme.DARK,
        )
        cache.set(new_key, sample_figure)

        assert cache.get(keys[0]) == sample_figure
        assert cache.get(keys[1]) is None
        assert cache.get(keys[2]) == sample_figure
        assert cache.get(new_key) == sample_figure


class TestPlotCacheStats:
    """Tests for PlotCache statistics."""

    def test_get_stats_returns_correct_initial_values(self, fresh_plot_cache):
        """Test that get_stats() returns correct initial values."""
        stats = fresh_plot_cache.get_stats()
        assert stats["hit_rate"] == 0.0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["size"] == 0

    def test_get_stats_returns_correct_values_after_operations(
        self, fresh_plot_cache, cache_key_dark, cache_key_light, sample_figure
    ):
        """Test that get_stats() returns correct values after operations."""
        fresh_plot_cache.set(cache_key_dark, sample_figure)
        fresh_plot_cache.set(cache_key_light, sample_figure)

        fresh_plot_cache.get(cache_key_dark)
        fresh_plot_cache.get(cache_key_dark)
        fresh_plot_cache.get(cache_key_light)

        missing_key = CacheKey(
            plot_id="missing",
            config_hash="hash1",
            runs_hash="runs1",
            theme=PlotTheme.DARK,
        )
        fresh_plot_cache.get(missing_key)
        fresh_plot_cache.get(missing_key)

        stats = fresh_plot_cache.get_stats()
        assert stats["hits"] == 3
        assert stats["misses"] == 2
        assert stats["hit_rate"] == 0.6
        assert stats["size"] == 2


class TestGetPlotCacheSingleton:
    """Tests for get_plot_cache() singleton function."""

    def test_returns_plot_cache_instance(self):
        """Test that get_plot_cache returns PlotCache instance."""
        cache = get_plot_cache()
        assert isinstance(cache, PlotCache)

    def test_returns_same_instance_on_multiple_calls(self):
        """Test that get_plot_cache returns same instance on multiple calls."""
        cache1 = get_plot_cache()
        cache2 = get_plot_cache()
        assert cache1 is cache2

    def test_creates_new_instance_when_global_is_none(self):
        """Test that get_plot_cache creates new instance when global is None."""
        cache_module._PLOT_CACHE = None
        cache = get_plot_cache()
        assert cache is not None
        assert isinstance(cache, PlotCache)


class TestComputeConfigHash:
    """Tests for compute_config_hash function."""

    def test_returns_12_character_hash_string(self):
        """Test that compute_config_hash returns 12-character hash string."""
        config = {"x_metric": "ttft", "y_metric": "tpot"}
        hash_str = compute_config_hash(config)
        assert isinstance(hash_str, str)
        assert len(hash_str) == 12

    def test_same_config_produces_same_hash(self):
        """Test that same config produces same hash."""
        config1 = {"x_metric": "ttft", "y_metric": "tpot", "plot_type": "scatter"}
        config2 = {"x_metric": "ttft", "y_metric": "tpot", "plot_type": "scatter"}
        hash1 = compute_config_hash(config1)
        hash2 = compute_config_hash(config2)
        assert hash1 == hash2

    def test_different_config_produces_different_hash(self):
        """Test that different config produces different hash."""
        config1 = {"x_metric": "ttft", "y_metric": "tpot"}
        config2 = {"x_metric": "ttft", "y_metric": "itl"}
        hash1 = compute_config_hash(config1)
        hash2 = compute_config_hash(config2)
        assert hash1 != hash2

    def test_key_order_independence(self):
        """Test that key order does not affect hash."""
        config1 = {"x_metric": "ttft", "y_metric": "tpot", "plot_type": "scatter"}
        config2 = {"plot_type": "scatter", "y_metric": "tpot", "x_metric": "ttft"}
        hash1 = compute_config_hash(config1)
        hash2 = compute_config_hash(config2)
        assert hash1 == hash2

    def test_ignores_irrelevant_keys(self):
        """Test that irrelevant keys are ignored."""
        config1 = {"x_metric": "ttft", "y_metric": "tpot"}
        config2 = {
            "x_metric": "ttft",
            "y_metric": "tpot",
            "irrelevant_key": "value",
        }
        hash1 = compute_config_hash(config1)
        hash2 = compute_config_hash(config2)
        assert hash1 == hash2

    def test_all_relevant_keys_affect_hash(self):
        """Test that all relevant keys affect the hash."""
        base_config = {"x_metric": "ttft"}
        base_hash = compute_config_hash(base_config)

        relevant_keys = [
            "x_metric",
            "x_stat",
            "y_metric",
            "y_stat",
            "group_by",
            "label_by",
            "log_scale",
            "x_axis",
            "stat",
            "source",
            "y_metric_base",
            "plot_type",
            "title",
            "mode",
        ]

        for key in relevant_keys:
            if key == "x_metric":
                continue
            config = base_config.copy()
            config[key] = "test_value"
            hash_with_key = compute_config_hash(config)
            assert hash_with_key != base_hash, f"Key {key} should affect hash"

    def test_handles_empty_config(self):
        """Test that compute_config_hash handles empty config."""
        config = {}
        hash_str = compute_config_hash(config)
        assert isinstance(hash_str, str)
        assert len(hash_str) == 12

    def test_handles_none_values(self):
        """Test that compute_config_hash handles None values."""
        config1 = {"x_metric": None, "y_metric": "tpot"}
        config2 = {"x_metric": None, "y_metric": "tpot"}
        hash1 = compute_config_hash(config1)
        hash2 = compute_config_hash(config2)
        assert hash1 == hash2


class TestComputeRunsHash:
    """Tests for compute_runs_hash function."""

    def test_returns_all_for_none_input(self):
        """Test that compute_runs_hash returns 'all' for None input."""
        hash_str = compute_runs_hash(None)
        assert hash_str == "all"

    def test_returns_8_character_hash_for_list(self):
        """Test that compute_runs_hash returns 8-character hash for list."""
        hash_str = compute_runs_hash([0, 1, 2])
        assert isinstance(hash_str, str)
        assert len(hash_str) == 8

    def test_same_runs_produce_same_hash(self):
        """Test that same runs produce same hash."""
        hash1 = compute_runs_hash([0, 1, 2, 3])
        hash2 = compute_runs_hash([0, 1, 2, 3])
        assert hash1 == hash2

    def test_different_runs_produce_different_hash(self):
        """Test that different runs produce different hash."""
        hash1 = compute_runs_hash([0, 1, 2])
        hash2 = compute_runs_hash([0, 1, 3])
        assert hash1 != hash2

    def test_order_independence(self):
        """Test that order independence (sorted internally)."""
        hash1 = compute_runs_hash([3, 1, 2, 0])
        hash2 = compute_runs_hash([0, 1, 2, 3])
        assert hash1 == hash2

    def test_handles_empty_list(self):
        """Test that compute_runs_hash handles empty list."""
        hash_str = compute_runs_hash([])
        assert isinstance(hash_str, str)
        assert len(hash_str) == 8

    def test_handles_single_run(self):
        """Test that compute_runs_hash handles single run."""
        hash_str = compute_runs_hash([5])
        assert isinstance(hash_str, str)
        assert len(hash_str) == 8
