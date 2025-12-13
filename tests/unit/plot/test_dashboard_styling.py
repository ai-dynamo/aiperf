# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for dashboard styling functions.

Tests for pure functions in aiperf.plot.dashboard.styling module.
"""

import pytest

from aiperf.plot.constants import (
    DARK_THEME_COLORS,
    LIGHT_THEME_COLORS,
    NVIDIA_GREEN,
    PlotTheme,
)
from aiperf.plot.dashboard.styling import (
    get_all_themes_css,
    get_button_style,
    get_dropdown_css,
    get_header_style,
    get_label_style,
    get_main_area_style,
    get_scoped_theme_css,
    get_section_header_style,
    get_sidebar_style,
    get_theme_colors,
)


class TestGetThemeColors:
    """Tests for get_theme_colors function."""

    def test_returns_dark_colors_for_dark_theme(self):
        """Test that dark theme returns dark color palette."""
        result = get_theme_colors(PlotTheme.DARK)
        assert result == DARK_THEME_COLORS

    def test_returns_light_colors_for_light_theme(self):
        """Test that light theme returns light color palette."""
        result = get_theme_colors(PlotTheme.LIGHT)
        assert result == LIGHT_THEME_COLORS

    @pytest.mark.parametrize("theme", [PlotTheme.DARK, PlotTheme.LIGHT])
    def test_returns_required_color_keys(self, theme):
        """Test that all required color keys are present."""
        result = get_theme_colors(theme)
        required_keys = ["background", "text", "paper", "border", "grid"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"


class TestGetDropdownCss:
    """Tests for get_dropdown_css function."""

    @pytest.mark.parametrize("theme", [PlotTheme.DARK, PlotTheme.LIGHT])
    def test_returns_css_string(self, theme):
        """Test that function returns a CSS string."""
        result = get_dropdown_css(theme)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_includes_theme_variables(self):
        """Test that CSS includes theme CSS variables."""
        result = get_dropdown_css(PlotTheme.DARK)
        assert ".theme-dark" in result
        assert ".theme-light" in result
        assert "--bg-color" in result
        assert "--text-color" in result
        assert "--paper-color" in result

    def test_includes_select_control_styling(self):
        """Test that CSS includes Select component styling."""
        result = get_dropdown_css(PlotTheme.DARK)
        assert ".Select-control" in result
        assert ".Select-value" in result
        assert ".Select-menu-outer" in result
        assert ".Select-option" in result

    def test_includes_modal_styling(self):
        """Test that CSS includes modal styling."""
        result = get_dropdown_css(PlotTheme.DARK)
        assert ".modal-content" in result
        assert ".modal-header" in result
        assert ".modal-body" in result


class TestGetScopedThemeCss:
    """Tests for get_scoped_theme_css function."""

    @pytest.mark.parametrize("theme", [PlotTheme.DARK, PlotTheme.LIGHT])
    def test_returns_css_string(self, theme):
        """Test that function returns a CSS string."""
        result = get_scoped_theme_css(theme)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_dark_theme_scoped_to_theme_dark(self):
        """Test that dark theme CSS is scoped to .theme-dark class."""
        result = get_scoped_theme_css(PlotTheme.DARK)
        assert ".theme-dark" in result
        # Should not have standalone .theme-light selectors
        lines_with_theme_light = [
            line
            for line in result.split("\n")
            if ".theme-light" in line and ".theme-dark" not in line
        ]
        assert len(lines_with_theme_light) == 0

    def test_light_theme_scoped_to_theme_light(self):
        """Test that light theme CSS is scoped to .theme-light class."""
        result = get_scoped_theme_css(PlotTheme.LIGHT)
        assert ".theme-light" in result

    def test_includes_plot_container_styling(self):
        """Test that CSS includes plot container styling."""
        result = get_scoped_theme_css(PlotTheme.DARK)
        assert ".plot-container" in result

    def test_includes_button_styling(self):
        """Test that CSS includes button styling."""
        result = get_scoped_theme_css(PlotTheme.DARK)
        assert ".plot-settings-btn" in result
        assert ".plot-hide-btn" in result

    def test_includes_drag_handle_styling(self):
        """Test that CSS includes resize handle styling."""
        result = get_scoped_theme_css(PlotTheme.DARK)
        assert ".resize-handle" in result


class TestGetAllThemesCss:
    """Tests for get_all_themes_css function."""

    def test_returns_css_string(self):
        """Test that function returns a CSS string."""
        result = get_all_themes_css()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_includes_both_theme_scopes(self):
        """Test that CSS includes both theme scopes."""
        result = get_all_themes_css()
        assert ".theme-dark" in result
        assert ".theme-light" in result

    def test_includes_global_styles(self):
        """Test that CSS includes global (theme-independent) styles."""
        result = get_all_themes_css()
        assert "html, body" in result
        assert ".plot-container" in result
        assert ".resize-handle" in result

    def test_includes_nvidia_green_color(self):
        """Test that CSS includes NVIDIA green brand color."""
        result = get_all_themes_css()
        assert NVIDIA_GREEN.lower() in result.lower()


class TestGetHeaderStyle:
    """Tests for get_header_style function."""

    @pytest.mark.parametrize("theme", [PlotTheme.DARK, PlotTheme.LIGHT])
    def test_returns_style_dict(self, theme):
        """Test that function returns a dictionary."""
        result = get_header_style(theme)
        assert isinstance(result, dict)

    @pytest.mark.parametrize("theme", [PlotTheme.DARK, PlotTheme.LIGHT])
    def test_includes_required_style_properties(self, theme):
        """Test that result includes required CSS properties."""
        result = get_header_style(theme)
        assert "display" in result
        assert "padding" in result
        assert "background" in result

    def test_dark_theme_has_different_background(self):
        """Test that dark theme has different background than light theme."""
        dark_result = get_header_style(PlotTheme.DARK)
        light_result = get_header_style(PlotTheme.LIGHT)
        assert dark_result["background"] != light_result["background"]


class TestGetSidebarStyle:
    """Tests for get_sidebar_style function."""

    @pytest.mark.parametrize("theme", [PlotTheme.DARK, PlotTheme.LIGHT])
    def test_returns_style_dict(self, theme):
        """Test that function returns a dictionary."""
        result = get_sidebar_style(theme)
        assert isinstance(result, dict)

    def test_collapsed_style_has_zero_width(self):
        """Test that collapsed sidebar has zero width."""
        result = get_sidebar_style(PlotTheme.DARK, collapsed=True)
        assert result["width"] == "0px"
        assert result["min-width"] == "0px"
        assert "overflow" in result

    def test_expanded_style_has_positive_width(self):
        """Test that expanded sidebar has positive width."""
        result = get_sidebar_style(PlotTheme.DARK, collapsed=False)
        assert result["width"] == "300px"
        assert result["min-width"] == "300px"

    def test_collapsed_has_hidden_overflow(self):
        """Test that collapsed sidebar has hidden overflow."""
        result = get_sidebar_style(PlotTheme.DARK, collapsed=True)
        assert result["overflow"] == "hidden"

    def test_expanded_has_auto_overflow(self):
        """Test that expanded sidebar allows scrolling."""
        result = get_sidebar_style(PlotTheme.DARK, collapsed=False)
        assert result["overflow-y"] == "auto"

    @pytest.mark.parametrize("collapsed", [True, False])
    def test_includes_transition_property(self, collapsed):
        """Test that sidebar style includes transition for animation."""
        result = get_sidebar_style(PlotTheme.DARK, collapsed=collapsed)
        assert "transition" in result


class TestGetMainAreaStyle:
    """Tests for get_main_area_style function."""

    @pytest.mark.parametrize("theme", [PlotTheme.DARK, PlotTheme.LIGHT])
    def test_returns_style_dict(self, theme):
        """Test that function returns a dictionary."""
        result = get_main_area_style(theme)
        assert isinstance(result, dict)

    @pytest.mark.parametrize("theme", [PlotTheme.DARK, PlotTheme.LIGHT])
    def test_includes_flex_property(self, theme):
        """Test that main area uses flex for layout."""
        result = get_main_area_style(theme)
        assert "flex" in result

    @pytest.mark.parametrize("theme", [PlotTheme.DARK, PlotTheme.LIGHT])
    def test_includes_background_property(self, theme):
        """Test that main area has background color."""
        result = get_main_area_style(theme)
        assert "background" in result


class TestGetButtonStyle:
    """Tests for get_button_style function."""

    @pytest.mark.parametrize("theme", [PlotTheme.DARK, PlotTheme.LIGHT])
    def test_returns_style_dict(self, theme):
        """Test that function returns a dictionary."""
        result = get_button_style(theme)
        assert isinstance(result, dict)

    def test_primary_variant_has_nvidia_green_background(self):
        """Test that primary button has NVIDIA green background."""
        result = get_button_style(PlotTheme.DARK, variant="primary")
        assert result["background"] == NVIDIA_GREEN

    def test_secondary_variant_has_paper_background(self):
        """Test that secondary button has paper background."""
        result = get_button_style(PlotTheme.DARK, variant="secondary")
        colors = get_theme_colors(PlotTheme.DARK)
        assert result["background"] == colors["paper"]

    def test_outline_variant_has_transparent_background(self):
        """Test that outline button has transparent background."""
        result = get_button_style(PlotTheme.DARK, variant="outline")
        assert result["background"] == "transparent"

    def test_outline_variant_has_nvidia_green_border(self):
        """Test that outline button has NVIDIA green border."""
        result = get_button_style(PlotTheme.DARK, variant="outline")
        assert NVIDIA_GREEN in result["border"]

    @pytest.mark.parametrize(
        "variant",
        ["primary", "secondary", "outline"],
    )
    def test_all_variants_have_cursor_pointer(self, variant):
        """Test that all button variants have pointer cursor."""
        result = get_button_style(PlotTheme.DARK, variant=variant)
        assert result["cursor"] == "pointer"

    @pytest.mark.parametrize(
        "variant",
        ["primary", "secondary", "outline"],
    )
    def test_all_variants_have_transition(self, variant):
        """Test that all button variants have transition property."""
        result = get_button_style(PlotTheme.DARK, variant=variant)
        assert "transition" in result


class TestGetLabelStyle:
    """Tests for get_label_style function."""

    @pytest.mark.parametrize("theme", [PlotTheme.DARK, PlotTheme.LIGHT])
    def test_returns_style_dict(self, theme):
        """Test that function returns a dictionary."""
        result = get_label_style(theme)
        assert isinstance(result, dict)

    @pytest.mark.parametrize("theme", [PlotTheme.DARK, PlotTheme.LIGHT])
    def test_includes_font_properties(self, theme):
        """Test that label style includes font properties."""
        result = get_label_style(theme)
        assert "font-size" in result
        assert "font-weight" in result
        assert "font-family" in result

    @pytest.mark.parametrize("theme", [PlotTheme.DARK, PlotTheme.LIGHT])
    def test_includes_color_property(self, theme):
        """Test that label style includes color property."""
        result = get_label_style(theme)
        assert "color" in result


class TestGetSectionHeaderStyle:
    """Tests for get_section_header_style function."""

    @pytest.mark.parametrize("theme", [PlotTheme.DARK, PlotTheme.LIGHT])
    def test_returns_style_dict(self, theme):
        """Test that function returns a dictionary."""
        result = get_section_header_style(theme)
        assert isinstance(result, dict)

    @pytest.mark.parametrize("theme", [PlotTheme.DARK, PlotTheme.LIGHT])
    def test_includes_uppercase_text_transform(self, theme):
        """Test that section header has uppercase text transform."""
        result = get_section_header_style(theme)
        assert result.get("text-transform") == "uppercase"

    @pytest.mark.parametrize("theme", [PlotTheme.DARK, PlotTheme.LIGHT])
    def test_includes_nvidia_green_border(self, theme):
        """Test that section header has NVIDIA green bottom border."""
        result = get_section_header_style(theme)
        assert "border-bottom" in result
        assert NVIDIA_GREEN in result["border-bottom"]

    @pytest.mark.parametrize("theme", [PlotTheme.DARK, PlotTheme.LIGHT])
    def test_includes_letter_spacing(self, theme):
        """Test that section header has letter spacing."""
        result = get_section_header_style(theme)
        assert "letter-spacing" in result
