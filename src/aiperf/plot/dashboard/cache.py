# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Plot caching module for dashboard performance optimization.

This module provides an in-memory cache for Plotly figure objects to avoid
regenerating plots when switching themes, hiding/showing plots, or exporting.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field

import plotly.graph_objects as go

from aiperf.plot.constants import PlotTheme


@dataclass(frozen=True)
class CacheKey:
    """
    Immutable cache key for plot lookup.

    A plot is uniquely identified by:
    - plot_id: The unique identifier for the plot
    - config_hash: Hash of plot configuration (metrics, stats, plot type, etc.)
    - runs_hash: Hash of selected run indices
    - theme: Light or dark theme
    """

    plot_id: str
    config_hash: str
    runs_hash: str
    theme: PlotTheme


@dataclass
class CachedPlot:
    """
    Cached plot with metadata for LRU eviction.

    Attributes:
        figure: The cached Plotly figure object
        created_at: Timestamp when the figure was cached
        last_accessed: Timestamp of last access (for LRU eviction)
    """

    figure: go.Figure
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    def touch(self) -> None:
        """Update last accessed timestamp."""
        self.last_accessed = time.time()


class PlotCache:
    """
    In-memory cache for Plotly figures.

    Stores figures for both themes to enable instant theme switching.
    Uses LRU eviction when cache exceeds max size.

    Attributes:
        max_plots_per_theme: Maximum number of plots to cache per theme
    """

    def __init__(self, max_plots_per_theme: int = 50) -> None:
        """
        Initialize the plot cache.

        Args:
            max_plots_per_theme: Maximum number of plots to cache per theme
        """
        self._cache: dict[CacheKey, CachedPlot] = {}
        self._max_plots_per_theme = max_plots_per_theme
        self._hits = 0
        self._misses = 0

    def get(self, key: CacheKey) -> go.Figure | None:
        """
        Get cached figure or None if not found.

        Args:
            key: Cache key to look up

        Returns:
            Plotly figure object or None if not cached
        """
        cached = self._cache.get(key)
        if cached:
            cached.touch()
            self._hits += 1
            return cached.figure
        self._misses += 1
        return None

    def set(self, key: CacheKey, figure: go.Figure) -> None:
        """
        Store figure in cache.

        Args:
            key: Cache key
            figure: Plotly figure object to cache
        """
        self._cache[key] = CachedPlot(figure=figure)
        self._maybe_evict()

    def invalidate_plot(self, plot_id: str) -> None:
        """
        Invalidate all cached versions of a specific plot.

        Called when plot configuration changes.

        Args:
            plot_id: ID of the plot to invalidate
        """
        keys_to_remove = [k for k in self._cache if k.plot_id == plot_id]
        for key in keys_to_remove:
            del self._cache[key]

    def invalidate_by_runs(self, runs_hash: str) -> None:
        """
        Invalidate all plots with different run selection.

        Called when run selection changes. Keeps only plots matching the new selection.

        Args:
            runs_hash: Hash of the new run selection
        """
        keys_to_remove = [k for k in self._cache if k.runs_hash != runs_hash]
        for key in keys_to_remove:
            del self._cache[key]

    def clear(self) -> None:
        """Clear entire cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def get_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dict with hit_rate, hits, misses, and size
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "hit_rate": hit_rate,
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
        }

    def _maybe_evict(self) -> None:
        """Evict oldest entries (LRU) if cache is too large."""
        for theme in PlotTheme:
            theme_keys = [k for k in self._cache if k.theme == theme]
            if len(theme_keys) > self._max_plots_per_theme:
                sorted_keys = sorted(
                    theme_keys, key=lambda k: self._cache[k].last_accessed
                )
                for key in sorted_keys[: -self._max_plots_per_theme]:
                    del self._cache[key]


_PLOT_CACHE: PlotCache | None = None


def get_plot_cache() -> PlotCache:
    """
    Get or create the global plot cache singleton.

    Returns:
        The global PlotCache instance
    """
    global _PLOT_CACHE
    if _PLOT_CACHE is None:
        _PLOT_CACHE = PlotCache()
    return _PLOT_CACHE


def compute_config_hash(plot_config: dict) -> str:
    """
    Compute stable hash of plot configuration.

    Includes all parameters that affect figure appearance for both modes:
    - Multi-run: x_metric, y_metric, x_stat, y_stat, group_by, label_by
    - Single-run: x_axis, stat, source, y_metric_base
    - Common: plot_type, log_scale, title, mode

    Args:
        plot_config: Plot configuration dictionary

    Returns:
        12-character hash string
    """
    relevant_keys = [
        # Multi-run keys
        "x_metric",
        "x_stat",
        "y_metric",
        "y_stat",
        "group_by",
        "label_by",
        "log_scale",
        # Single-run keys
        "x_axis",
        "stat",
        "source",
        "y_metric_base",
        # Common keys
        "plot_type",
        "title",
        "mode",
    ]

    cache_dict = {k: plot_config.get(k) for k in relevant_keys}
    json_str = json.dumps(cache_dict, sort_keys=True, default=str)
    return hashlib.md5(json_str.encode(), usedforsecurity=False).hexdigest()[:12]


def compute_runs_hash(selected_runs: list[int] | None) -> str:
    """
    Compute hash of selected run indices.

    Args:
        selected_runs: List of run indices or None for all runs

    Returns:
        Hash string representing the run selection
    """
    if selected_runs is None:
        return "all"

    sorted_runs = sorted(selected_runs)
    runs_str = ",".join(str(r) for r in sorted_runs)
    return hashlib.md5(runs_str.encode(), usedforsecurity=False).hexdigest()[:8]
