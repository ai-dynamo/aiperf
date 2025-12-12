# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for plot constants module.

Tests for constants, enums, and configuration values used in plot generation.
"""

import re
from pathlib import Path

import pytest

from aiperf.plot.constants import (
    ALL_STAT_KEYS,
    AVAILABLE_STATS,
    CUMULATIVE_METRIC_PATTERNS,
    DARK_THEME_COLORS,
    DEFAULT_DASHBOARD_PORT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PERCENTILE,
    DEFAULT_PERCENTILES,
    DEFAULT_PLOT_DPI,
    DEFAULT_PLOT_HEIGHT,
    DEFAULT_PLOT_WIDTH,
    DEFAULT_PNG_OUTPUT_DIR,
    DERIVED_METRIC_DIRECTIONS,
    LIGHT_THEME_COLORS,
    NON_METRIC_KEYS,
    NVIDIA_BORDER_DARK,
    NVIDIA_BORDER_LIGHT,
    NVIDIA_CARD_BG,
    NVIDIA_DARK,
    NVIDIA_DARK_BG,
    NVIDIA_GOLD,
    NVIDIA_GRAY,
    NVIDIA_GREEN,
    NVIDIA_TEXT_LIGHT,
    NVIDIA_WHITE,
    OUTLIER_RED,
    PLOT_FONT_FAMILY,
    PLOT_LOG_FILE,
    PROFILE_EXPORT_AIPERF_JSON,
    PROFILE_EXPORT_GPU_TELEMETRY_JSONL,
    PROFILE_EXPORT_JSONL,
    PROFILE_EXPORT_TIMESLICES_CSV,
    STAT_LABELS,
    PlotMode,
    PlotTheme,
)


class TestPlotMode:
    """Tests for PlotMode enum."""

    def test_png_value(self) -> None:
        """Test PlotMode.PNG has correct value."""
        assert PlotMode.PNG == "png"

    def test_dashboard_value(self) -> None:
        """Test PlotMode.DASHBOARD has correct value."""
        assert PlotMode.DASHBOARD == "dashboard"

    def test_case_insensitivity_png(self) -> None:
        """Test PlotMode is case insensitive for PNG."""
        assert PlotMode("PNG") == PlotMode.PNG
        assert PlotMode("png") == PlotMode.PNG
        assert PlotMode("Png") == PlotMode.PNG

    def test_case_insensitivity_dashboard(self) -> None:
        """Test PlotMode is case insensitive for DASHBOARD."""
        assert PlotMode("DASHBOARD") == PlotMode.DASHBOARD
        assert PlotMode("dashboard") == PlotMode.DASHBOARD
        assert PlotMode("Dashboard") == PlotMode.DASHBOARD

    def test_all_values(self) -> None:
        """Test PlotMode has exactly two values."""
        assert len(list(PlotMode)) == 2
        assert set(PlotMode) == {PlotMode.PNG, PlotMode.DASHBOARD}


class TestPlotTheme:
    """Tests for PlotTheme enum."""

    def test_light_value(self) -> None:
        """Test PlotTheme.LIGHT has correct value."""
        assert PlotTheme.LIGHT == "light"

    def test_dark_value(self) -> None:
        """Test PlotTheme.DARK has correct value."""
        assert PlotTheme.DARK == "dark"

    def test_case_insensitivity_light(self) -> None:
        """Test PlotTheme is case insensitive for LIGHT."""
        assert PlotTheme("LIGHT") == PlotTheme.LIGHT
        assert PlotTheme("light") == PlotTheme.LIGHT
        assert PlotTheme("Light") == PlotTheme.LIGHT

    def test_case_insensitivity_dark(self) -> None:
        """Test PlotTheme is case insensitive for DARK."""
        assert PlotTheme("DARK") == PlotTheme.DARK
        assert PlotTheme("dark") == PlotTheme.DARK
        assert PlotTheme("Dark") == PlotTheme.DARK

    def test_all_values(self) -> None:
        """Test PlotTheme has exactly two values."""
        assert len(list(PlotTheme)) == 2
        assert set(PlotTheme) == {PlotTheme.LIGHT, PlotTheme.DARK}


class TestColorConstants:
    """Tests for color constant values."""

    @pytest.mark.parametrize(
        "color_constant,color_value",
        [
            (NVIDIA_GREEN, "#76B900"),
            (NVIDIA_DARK, "#0a0a0a"),
            (NVIDIA_GOLD, "#F4E5C3"),
            (NVIDIA_WHITE, "#FFFFFF"),
            (NVIDIA_DARK_BG, "#1a1a1a"),
            (NVIDIA_GRAY, "#999999"),
            (NVIDIA_BORDER_DARK, "#333333"),
            (NVIDIA_BORDER_LIGHT, "#CCCCCC"),
            (NVIDIA_TEXT_LIGHT, "#E0E0E0"),
            (NVIDIA_CARD_BG, "#252525"),
            (OUTLIER_RED, "#E74C3C"),
        ],
    )
    def test_color_values(self, color_constant: str, color_value: str) -> None:
        """Test color constants have expected values."""
        assert color_constant == color_value

    @pytest.mark.parametrize(
        "color_constant",
        [
            NVIDIA_GREEN,
            NVIDIA_DARK,
            NVIDIA_GOLD,
            NVIDIA_WHITE,
            NVIDIA_DARK_BG,
            NVIDIA_GRAY,
            NVIDIA_BORDER_DARK,
            NVIDIA_BORDER_LIGHT,
            NVIDIA_TEXT_LIGHT,
            NVIDIA_CARD_BG,
            OUTLIER_RED,
        ],
    )
    def test_color_starts_with_hash(self, color_constant: str) -> None:
        """Test all color constants start with '#'."""
        assert color_constant.startswith("#")

    @pytest.mark.parametrize(
        "color_constant",
        [
            NVIDIA_GREEN,
            NVIDIA_DARK,
            NVIDIA_GOLD,
            NVIDIA_WHITE,
            NVIDIA_DARK_BG,
            NVIDIA_GRAY,
            NVIDIA_BORDER_DARK,
            NVIDIA_BORDER_LIGHT,
            NVIDIA_TEXT_LIGHT,
            NVIDIA_CARD_BG,
            OUTLIER_RED,
        ],
    )
    def test_color_length(self, color_constant: str) -> None:
        """Test all color constants are 7 characters (#RRGGBB)."""
        assert len(color_constant) == 7

    @pytest.mark.parametrize(
        "color_constant",
        [
            NVIDIA_GREEN,
            NVIDIA_DARK,
            NVIDIA_GOLD,
            NVIDIA_WHITE,
            NVIDIA_DARK_BG,
            NVIDIA_GRAY,
            NVIDIA_BORDER_DARK,
            NVIDIA_BORDER_LIGHT,
            NVIDIA_TEXT_LIGHT,
            NVIDIA_CARD_BG,
            OUTLIER_RED,
        ],
    )
    def test_color_valid_hex(self, color_constant: str) -> None:
        """Test all color constants are valid hex colors."""
        hex_pattern = re.compile(r"^#[0-9A-Fa-f]{6}$")
        assert hex_pattern.match(color_constant)

    def test_nvidia_green_is_primary(self) -> None:
        """Test NVIDIA_GREEN is the expected primary brand color."""
        assert NVIDIA_GREEN == "#76B900"


class TestThemeColors:
    """Tests for theme color dictionaries."""

    def test_dark_theme_has_required_keys(self) -> None:
        """Test DARK_THEME_COLORS has all required keys."""
        required_keys = {
            "primary",
            "secondary",
            "background",
            "paper",
            "text",
            "grid",
            "border",
        }
        assert set(DARK_THEME_COLORS.keys()) == required_keys

    def test_light_theme_has_required_keys(self) -> None:
        """Test LIGHT_THEME_COLORS has all required keys."""
        required_keys = {
            "primary",
            "secondary",
            "background",
            "paper",
            "text",
            "grid",
            "border",
        }
        assert set(LIGHT_THEME_COLORS.keys()) == required_keys

    def test_dark_theme_has_nvidia_green_primary(self) -> None:
        """Test DARK_THEME_COLORS uses NVIDIA_GREEN as primary."""
        assert DARK_THEME_COLORS["primary"] == NVIDIA_GREEN

    def test_light_theme_has_nvidia_green_primary(self) -> None:
        """Test LIGHT_THEME_COLORS uses NVIDIA_GREEN as primary."""
        assert LIGHT_THEME_COLORS["primary"] == NVIDIA_GREEN

    def test_dark_theme_color_values(self) -> None:
        """Test DARK_THEME_COLORS has expected color values."""
        assert DARK_THEME_COLORS["primary"] == NVIDIA_GREEN
        assert DARK_THEME_COLORS["secondary"] == NVIDIA_GOLD
        assert DARK_THEME_COLORS["background"] == NVIDIA_DARK_BG
        assert DARK_THEME_COLORS["paper"] == NVIDIA_CARD_BG
        assert DARK_THEME_COLORS["text"] == NVIDIA_TEXT_LIGHT
        assert DARK_THEME_COLORS["grid"] == NVIDIA_BORDER_DARK
        assert DARK_THEME_COLORS["border"] == NVIDIA_BORDER_DARK

    def test_light_theme_color_values(self) -> None:
        """Test LIGHT_THEME_COLORS has expected color values."""
        assert LIGHT_THEME_COLORS["primary"] == NVIDIA_GREEN
        assert LIGHT_THEME_COLORS["secondary"] == NVIDIA_GRAY
        assert LIGHT_THEME_COLORS["background"] == NVIDIA_WHITE
        assert LIGHT_THEME_COLORS["paper"] == NVIDIA_WHITE
        assert LIGHT_THEME_COLORS["text"] == NVIDIA_DARK
        assert LIGHT_THEME_COLORS["grid"] == NVIDIA_BORDER_LIGHT
        assert LIGHT_THEME_COLORS["border"] == NVIDIA_BORDER_LIGHT

    def test_themes_have_same_keys(self) -> None:
        """Test both themes have the same structure."""
        assert set(DARK_THEME_COLORS.keys()) == set(LIGHT_THEME_COLORS.keys())

    def test_all_theme_colors_are_valid_hex(self) -> None:
        """Test all colors in theme dictionaries are valid hex colors."""
        hex_pattern = re.compile(r"^#[0-9A-Fa-f]{6}$")
        for color in DARK_THEME_COLORS.values():
            assert hex_pattern.match(color)
        for color in LIGHT_THEME_COLORS.values():
            assert hex_pattern.match(color)


class TestStatConstants:
    """Tests for statistical constant values."""

    def test_all_stat_keys_contains_basic_stats(self) -> None:
        """Test ALL_STAT_KEYS contains avg, min, max, std."""
        assert "avg" in ALL_STAT_KEYS
        assert "min" in ALL_STAT_KEYS
        assert "max" in ALL_STAT_KEYS
        assert "std" in ALL_STAT_KEYS

    def test_all_stat_keys_contains_percentiles(self) -> None:
        """Test ALL_STAT_KEYS contains all percentiles."""
        for p in DEFAULT_PERCENTILES:
            assert f"p{p}" in ALL_STAT_KEYS

    def test_all_stat_keys_contains_p1_through_p99(self) -> None:
        """Test ALL_STAT_KEYS contains p1 through p99."""
        assert "p1" in ALL_STAT_KEYS
        assert "p5" in ALL_STAT_KEYS
        assert "p10" in ALL_STAT_KEYS
        assert "p25" in ALL_STAT_KEYS
        assert "p50" in ALL_STAT_KEYS
        assert "p75" in ALL_STAT_KEYS
        assert "p90" in ALL_STAT_KEYS
        assert "p95" in ALL_STAT_KEYS
        assert "p99" in ALL_STAT_KEYS

    def test_stat_labels_has_entries_for_all_stat_keys(self) -> None:
        """Test STAT_LABELS has entries for all ALL_STAT_KEYS."""
        for stat_key in ALL_STAT_KEYS:
            assert stat_key in STAT_LABELS

    def test_stat_labels_basic_stats(self) -> None:
        """Test STAT_LABELS has correct labels for basic statistics."""
        assert STAT_LABELS["avg"] == "Average"
        assert STAT_LABELS["min"] == "Minimum"
        assert STAT_LABELS["max"] == "Maximum"
        assert STAT_LABELS["std"] == "Std Dev"

    def test_stat_labels_percentiles(self) -> None:
        """Test STAT_LABELS has correct labels for percentiles."""
        assert STAT_LABELS["p1"] == "p1"
        assert STAT_LABELS["p5"] == "p5"
        assert STAT_LABELS["p10"] == "p10"
        assert STAT_LABELS["p25"] == "p25"
        assert STAT_LABELS["p50"] == "p50 (Median)"
        assert STAT_LABELS["p75"] == "p75"
        assert STAT_LABELS["p90"] == "p90"
        assert STAT_LABELS["p95"] == "p95"
        assert STAT_LABELS["p99"] == "p99"

    def test_available_stats_contains_basic_stats(self) -> None:
        """Test AVAILABLE_STATS contains the four basic statistics."""
        assert AVAILABLE_STATS == ["avg", "min", "max", "std"]

    def test_default_percentiles_values(self) -> None:
        """Test DEFAULT_PERCENTILES has expected values."""
        assert DEFAULT_PERCENTILES == [1, 5, 10, 25, 50, 75, 90, 95, 99]

    def test_default_percentile_value(self) -> None:
        """Test DEFAULT_PERCENTILE is p50."""
        assert DEFAULT_PERCENTILE == "p50"

    def test_all_stat_keys_count(self) -> None:
        """Test ALL_STAT_KEYS has correct count."""
        expected_count = len(AVAILABLE_STATS) + len(DEFAULT_PERCENTILES)
        assert len(ALL_STAT_KEYS) == expected_count


class TestFilePatternConstants:
    """Tests for file pattern constants."""

    def test_profile_export_jsonl(self) -> None:
        """Test PROFILE_EXPORT_JSONL has expected value."""
        assert PROFILE_EXPORT_JSONL == "profile_export.jsonl"

    def test_profile_export_aiperf_json(self) -> None:
        """Test PROFILE_EXPORT_AIPERF_JSON has expected value."""
        assert PROFILE_EXPORT_AIPERF_JSON == "profile_export_aiperf.json"

    def test_profile_export_timeslices_csv(self) -> None:
        """Test PROFILE_EXPORT_TIMESLICES_CSV has expected value."""
        assert PROFILE_EXPORT_TIMESLICES_CSV == "profile_export_aiperf_timeslices.csv"

    def test_profile_export_gpu_telemetry_jsonl(self) -> None:
        """Test PROFILE_EXPORT_GPU_TELEMETRY_JSONL has expected value."""
        assert PROFILE_EXPORT_GPU_TELEMETRY_JSONL == "gpu_telemetry_export.jsonl"


class TestPathConstants:
    """Tests for path constants."""

    def test_default_output_dir(self) -> None:
        """Test DEFAULT_OUTPUT_DIR is Path('plots')."""
        assert Path("plots") == DEFAULT_OUTPUT_DIR
        assert isinstance(DEFAULT_OUTPUT_DIR, Path)

    def test_default_png_output_dir(self) -> None:
        """Test DEFAULT_PNG_OUTPUT_DIR is plots/png."""
        assert Path("plots") / "png" == DEFAULT_PNG_OUTPUT_DIR
        assert isinstance(DEFAULT_PNG_OUTPUT_DIR, Path)

    def test_plot_log_file(self) -> None:
        """Test PLOT_LOG_FILE has expected value."""
        assert PLOT_LOG_FILE == "aiperf_plot.log"


class TestPlotSettingConstants:
    """Tests for plot setting constants."""

    def test_default_plot_width(self) -> None:
        """Test DEFAULT_PLOT_WIDTH has expected value."""
        assert DEFAULT_PLOT_WIDTH == 1600

    def test_default_plot_height(self) -> None:
        """Test DEFAULT_PLOT_HEIGHT has expected value."""
        assert DEFAULT_PLOT_HEIGHT == 800

    def test_default_plot_dpi(self) -> None:
        """Test DEFAULT_PLOT_DPI has expected value."""
        assert DEFAULT_PLOT_DPI == 150

    def test_plot_font_family(self) -> None:
        """Test PLOT_FONT_FAMILY contains expected font stack."""
        assert "Roboto" in PLOT_FONT_FAMILY
        assert "Segoe UI" in PLOT_FONT_FAMILY
        assert "sans-serif" in PLOT_FONT_FAMILY

    def test_default_dashboard_port(self) -> None:
        """Test DEFAULT_DASHBOARD_PORT has expected value."""
        assert DEFAULT_DASHBOARD_PORT == 8050


class TestDerivedMetricDirections:
    """Tests for derived metric direction mappings."""

    def test_derived_metric_directions_is_dict(self) -> None:
        """Test DERIVED_METRIC_DIRECTIONS is a dictionary."""
        assert isinstance(DERIVED_METRIC_DIRECTIONS, dict)

    def test_derived_metric_directions_contains_throughput_metrics(self) -> None:
        """Test DERIVED_METRIC_DIRECTIONS contains expected throughput metrics."""
        assert "output_token_throughput_per_gpu" in DERIVED_METRIC_DIRECTIONS
        assert "output_token_throughput_per_user" in DERIVED_METRIC_DIRECTIONS

    def test_throughput_metrics_higher_is_better(self) -> None:
        """Test throughput metrics are marked as higher is better."""
        assert DERIVED_METRIC_DIRECTIONS["output_token_throughput_per_gpu"] is True
        assert DERIVED_METRIC_DIRECTIONS["output_token_throughput_per_user"] is True

    def test_derived_metric_directions_values_are_boolean(self) -> None:
        """Test all values in DERIVED_METRIC_DIRECTIONS are boolean."""
        for value in DERIVED_METRIC_DIRECTIONS.values():
            assert isinstance(value, bool)


class TestCumulativeMetricPatterns:
    """Tests for cumulative metric patterns."""

    def test_cumulative_metric_patterns_is_list(self) -> None:
        """Test CUMULATIVE_METRIC_PATTERNS is a list."""
        assert isinstance(CUMULATIVE_METRIC_PATTERNS, list)

    def test_cumulative_metric_patterns_contains_total(self) -> None:
        """Test CUMULATIVE_METRIC_PATTERNS contains 'total'."""
        assert "total" in CUMULATIVE_METRIC_PATTERNS

    def test_cumulative_metric_patterns_values_are_strings(self) -> None:
        """Test all values in CUMULATIVE_METRIC_PATTERNS are strings."""
        for pattern in CUMULATIVE_METRIC_PATTERNS:
            assert isinstance(pattern, str)


class TestNonMetricKeys:
    """Tests for non-metric keys set."""

    def test_non_metric_keys_is_set(self) -> None:
        """Test NON_METRIC_KEYS is a set."""
        assert isinstance(NON_METRIC_KEYS, set)

    def test_non_metric_keys_contains_expected_values(self) -> None:
        """Test NON_METRIC_KEYS contains expected metadata keys."""
        expected_keys = {
            "input_config",
            "telemetry_data",
            "start_time",
            "end_time",
            "was_cancelled",
            "error_summary",
        }
        assert expected_keys == NON_METRIC_KEYS

    def test_non_metric_keys_values_are_strings(self) -> None:
        """Test all values in NON_METRIC_KEYS are strings."""
        for key in NON_METRIC_KEYS:
            assert isinstance(key, str)
