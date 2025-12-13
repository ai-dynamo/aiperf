# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for dashboard component factory functions.

Tests for component creation functions in aiperf.plot.dashboard.components module.
"""

import pytest
from dash import dcc, html

from aiperf.plot.constants import PlotTheme
from aiperf.plot.dashboard.components import (
    create_button,
    create_collapsible_section,
    create_export_controls_card,
    create_global_stat_selector,
    create_label,
    create_log_scale_dropdown,
    create_section_header,
    create_sidebar_toggle_button,
    create_stat_selector_dropdown,
)


class TestCreateSectionHeader:
    """Tests for create_section_header function."""

    @pytest.mark.parametrize("theme", [PlotTheme.DARK, PlotTheme.LIGHT])
    def test_returns_html_div(self, theme):
        """Test that function returns an html.Div component."""
        result = create_section_header("Test Header", theme)
        assert isinstance(result, html.Div)

    @pytest.mark.parametrize("theme", [PlotTheme.DARK, PlotTheme.LIGHT])
    def test_contains_text_content(self, theme):
        """Test that div contains the provided text."""
        result = create_section_header("My Section", theme)
        assert result.children == "My Section"

    @pytest.mark.parametrize("theme", [PlotTheme.DARK, PlotTheme.LIGHT])
    def test_has_style_attribute(self, theme):
        """Test that component has style applied."""
        result = create_section_header("Test", theme)
        assert result.style is not None
        assert isinstance(result.style, dict)


class TestCreateLabel:
    """Tests for create_label function."""

    @pytest.mark.parametrize("theme", [PlotTheme.DARK, PlotTheme.LIGHT])
    def test_returns_html_label(self, theme):
        """Test that function returns an html.Label component."""
        result = create_label("Test Label", theme)
        assert isinstance(result, html.Label)

    @pytest.mark.parametrize("theme", [PlotTheme.DARK, PlotTheme.LIGHT])
    def test_contains_text_content(self, theme):
        """Test that label contains the provided text."""
        result = create_label("Field Name", theme)
        assert result.children == "Field Name"

    @pytest.mark.parametrize("theme", [PlotTheme.DARK, PlotTheme.LIGHT])
    def test_has_style_attribute(self, theme):
        """Test that component has style applied."""
        result = create_label("Test", theme)
        assert result.style is not None


class TestCreateStatSelectorDropdown:
    """Tests for create_stat_selector_dropdown function."""

    def test_returns_html_div(self):
        """Test that function returns an html.Div component."""
        result = create_stat_selector_dropdown("latency")
        assert isinstance(result, html.Div)

    def test_contains_label_and_dropdown(self):
        """Test that div contains both label and dropdown."""
        result = create_stat_selector_dropdown("latency")
        children = result.children
        assert len(children) == 2
        assert isinstance(children[0], html.Label)
        assert isinstance(children[1], dcc.Dropdown)

    def test_dropdown_has_stat_options(self):
        """Test that dropdown has statistical options."""
        result = create_stat_selector_dropdown("latency")
        dropdown = result.children[1]
        values = [opt["value"] for opt in dropdown.options]
        assert "p50" in values
        assert "avg" in values

    def test_dropdown_has_pattern_matching_id(self):
        """Test that dropdown has pattern-matching ID."""
        result = create_stat_selector_dropdown("time_to_first_token")
        dropdown = result.children[1]
        assert dropdown.id == {
            "type": "metric-stat-selector",
            "metric": "time_to_first_token",
        }

    def test_uses_default_stat(self):
        """Test that dropdown uses provided default stat."""
        result = create_stat_selector_dropdown("latency", default_stat="avg")
        dropdown = result.children[1]
        assert dropdown.value == "avg"

    def test_label_formats_metric_name(self):
        """Test that label formats metric name properly."""
        result = create_stat_selector_dropdown("time_to_first_token")
        label = result.children[0]
        assert label.children == "Time To First Token"

    @pytest.mark.parametrize("theme", [PlotTheme.DARK, PlotTheme.LIGHT])
    def test_dropdown_class_varies_by_theme(self, theme):
        """Test that dropdown class varies by theme."""
        result = create_stat_selector_dropdown("latency", theme=theme)
        dropdown = result.children[1]
        if theme == PlotTheme.DARK:
            assert "dark-dropdown" in dropdown.className
        else:
            assert dropdown.className == ""


class TestCreateLogScaleDropdown:
    """Tests for create_log_scale_dropdown function."""

    @pytest.mark.parametrize("theme", [PlotTheme.DARK, PlotTheme.LIGHT])
    def test_returns_html_div(self, theme):
        """Test that function returns an html.Div component."""
        result = create_log_scale_dropdown("pareto", theme)
        assert isinstance(result, html.Div)

    def test_contains_label_and_dropdown(self):
        """Test that div contains both label and dropdown."""
        result = create_log_scale_dropdown("pareto", PlotTheme.DARK)
        children = result.children
        assert len(children) == 2
        assert isinstance(children[0], html.Label)
        assert isinstance(children[1], dcc.Dropdown)

    def test_dropdown_has_scale_options(self):
        """Test that dropdown has log scale options."""
        result = create_log_scale_dropdown("pareto", PlotTheme.DARK)
        dropdown = result.children[1]
        values = [opt["value"] for opt in dropdown.options]
        assert "none" in values
        assert "x" in values
        assert "y" in values
        assert "both" in values

    def test_dropdown_id_uses_plot_id(self):
        """Test that dropdown ID is based on plot ID."""
        result = create_log_scale_dropdown("my-plot", PlotTheme.DARK)
        dropdown = result.children[1]
        assert dropdown.id == "my-plot-log-scale"

    def test_default_value_is_none(self):
        """Test that default value is 'none'."""
        result = create_log_scale_dropdown("pareto", PlotTheme.DARK)
        dropdown = result.children[1]
        assert dropdown.value == "none"


class TestCreateCollapsibleSection:
    """Tests for create_collapsible_section function."""

    @pytest.mark.parametrize("theme", [PlotTheme.DARK, PlotTheme.LIGHT])
    def test_returns_html_div(self, theme):
        """Test that function returns an html.Div component."""
        result = create_collapsible_section(
            section_id="test",
            title="Test Section",
            children=[html.P("Content")],
            theme=theme,
        )
        assert isinstance(result, html.Div)

    def test_contains_header_and_content(self):
        """Test that section has header and content divs."""
        result = create_collapsible_section(
            section_id="test",
            title="Test Section",
            children=[html.P("Content")],
            theme=PlotTheme.DARK,
        )
        children = result.children
        assert len(children) == 2  # Header and content

    def test_header_has_pattern_matching_id(self):
        """Test that header has pattern-matching ID."""
        result = create_collapsible_section(
            section_id="my-section",
            title="Title",
            children=[],
            theme=PlotTheme.DARK,
        )
        header = result.children[0]
        assert header.id == {"type": "section-header", "id": "my-section"}

    def test_content_has_pattern_matching_id(self):
        """Test that content has pattern-matching ID."""
        result = create_collapsible_section(
            section_id="my-section",
            title="Title",
            children=[],
            theme=PlotTheme.DARK,
        )
        content = result.children[1]
        assert content.id == {"type": "section-content", "id": "my-section"}

    def test_content_hidden_when_initially_closed(self):
        """Test that content is hidden when initially_open=False."""
        result = create_collapsible_section(
            section_id="test",
            title="Test",
            children=[html.P("Content")],
            theme=PlotTheme.DARK,
            initially_open=False,
        )
        content = result.children[1]
        assert content.style["display"] == "none"

    def test_content_visible_when_initially_open(self):
        """Test that content is visible when initially_open=True."""
        result = create_collapsible_section(
            section_id="test",
            title="Test",
            children=[html.P("Content")],
            theme=PlotTheme.DARK,
            initially_open=True,
        )
        content = result.children[1]
        assert content.style["display"] == "block"

    def test_header_has_arrow_indicator(self):
        """Test that header has arrow indicator."""
        result = create_collapsible_section(
            section_id="test",
            title="Test",
            children=[],
            theme=PlotTheme.DARK,
            initially_open=False,
        )
        header = result.children[0]
        arrow = header.children[1]
        assert arrow.id == {"type": "section-arrow", "id": "test"}
        assert arrow.children == "▶"  # Closed arrow

    def test_arrow_points_down_when_open(self):
        """Test that arrow points down when initially open."""
        result = create_collapsible_section(
            section_id="test",
            title="Test",
            children=[],
            theme=PlotTheme.DARK,
            initially_open=True,
        )
        header = result.children[0]
        arrow = header.children[1]
        assert arrow.children == "▼"  # Open arrow


class TestCreateButton:
    """Tests for create_button function."""

    @pytest.mark.parametrize("theme", [PlotTheme.DARK, PlotTheme.LIGHT])
    def test_returns_html_button(self, theme):
        """Test that function returns an html.Button component."""
        result = create_button("btn-test", "Click Me", theme)
        assert isinstance(result, html.Button)

    def test_button_has_correct_id(self):
        """Test that button has the specified ID."""
        result = create_button("my-button", "Text", PlotTheme.DARK)
        assert result.id == "my-button"

    def test_button_has_correct_text(self):
        """Test that button displays the specified text."""
        result = create_button("btn", "Submit", PlotTheme.DARK)
        assert result.children == "Submit"

    def test_button_has_initial_click_count(self):
        """Test that button has specified initial click count."""
        result = create_button("btn", "Text", PlotTheme.DARK, n_clicks=5)
        assert result.n_clicks == 5

    def test_default_click_count_is_zero(self):
        """Test that default click count is 0."""
        result = create_button("btn", "Text", PlotTheme.DARK)
        assert result.n_clicks == 0

    @pytest.mark.parametrize(
        "variant",
        ["primary", "secondary", "outline"],
    )
    def test_button_has_style_for_all_variants(self, variant):
        """Test that button has style for all variants."""
        result = create_button("btn", "Text", PlotTheme.DARK, variant=variant)
        assert result.style is not None
        assert isinstance(result.style, dict)


class TestCreateSidebarToggleButton:
    """Tests for create_sidebar_toggle_button function."""

    @pytest.mark.parametrize("theme", [PlotTheme.DARK, PlotTheme.LIGHT])
    def test_returns_html_button(self, theme):
        """Test that function returns an html.Button component."""
        result = create_sidebar_toggle_button(theme)
        assert isinstance(result, html.Button)

    def test_button_has_correct_id(self):
        """Test that button has sidebar-toggle-btn ID."""
        result = create_sidebar_toggle_button(PlotTheme.DARK)
        assert result.id == "sidebar-toggle-btn"

    def test_button_has_hamburger_icon(self):
        """Test that button displays hamburger icon."""
        result = create_sidebar_toggle_button(PlotTheme.DARK)
        assert result.children == "☰"

    def test_button_starts_with_zero_clicks(self):
        """Test that button starts with 0 clicks."""
        result = create_sidebar_toggle_button(PlotTheme.DARK)
        assert result.n_clicks == 0

    def test_button_is_fixed_position(self):
        """Test that button uses fixed positioning."""
        result = create_sidebar_toggle_button(PlotTheme.DARK)
        assert result.style["position"] == "fixed"


class TestCreateGlobalStatSelector:
    """Tests for create_global_stat_selector function."""

    @pytest.mark.parametrize("theme", [PlotTheme.DARK, PlotTheme.LIGHT])
    def test_returns_html_div(self, theme):
        """Test that function returns an html.Div component."""
        result = create_global_stat_selector(theme)
        assert isinstance(result, html.Div)

    def test_contains_label_dropdown_and_button(self):
        """Test that component contains label, dropdown, and button."""
        result = create_global_stat_selector(PlotTheme.DARK)

        # Find components by type
        def find_components(container):
            components = []
            if hasattr(container, "children"):
                children = container.children
                if isinstance(children, list):
                    for child in children:
                        components.extend(find_components(child))
                elif children is not None:
                    components.extend(find_components(children))
            components.append(container)
            return components

        all_components = find_components(result)
        component_types = [type(c) for c in all_components]

        assert html.Label in component_types
        assert dcc.Dropdown in component_types
        assert html.Button in component_types

    def test_dropdown_has_stat_options(self):
        """Test that dropdown has statistical options."""
        result = create_global_stat_selector(PlotTheme.DARK)
        # Find dropdown
        dropdown = None
        for child in result.children:
            if hasattr(child, "children"):
                for subchild in (
                    child.children
                    if isinstance(child.children, list)
                    else [child.children]
                ):
                    if isinstance(subchild, dcc.Dropdown):
                        dropdown = subchild
                        break
        assert dropdown is not None
        values = [opt["value"] for opt in dropdown.options]
        assert "p50" in values
        assert "avg" in values
        assert "p90" in values

    def test_dropdown_has_correct_id(self):
        """Test that dropdown has global-stat-selector ID."""
        result = create_global_stat_selector(PlotTheme.DARK)

        # Find dropdown in the nested structure
        # Structure: Div > [Label, Div > [Dropdown, Button]]
        inner_div = result.children[1]  # Second child is the inner div
        dropdown = inner_div.children[0]  # First child of inner div is dropdown
        assert isinstance(dropdown, dcc.Dropdown)
        assert dropdown.id == "global-stat-selector"


class TestCreateExportControlsCard:
    """Tests for create_export_controls_card function."""

    @pytest.mark.parametrize("theme", [PlotTheme.DARK, PlotTheme.LIGHT])
    def test_returns_html_div(self, theme):
        """Test that function returns an html.Div component."""
        result = create_export_controls_card(theme)
        assert isinstance(result, html.Div)

    def test_contains_download_component(self):
        """Test that card contains dcc.Download component as first child."""
        result = create_export_controls_card(PlotTheme.DARK)
        # Structure: Div > [dcc.Download, collapsible_section]
        download = result.children[0]
        assert isinstance(download, dcc.Download)
        assert download.id == "download-png-bundle"

    def test_contains_collapsible_section(self):
        """Test that card contains collapsible section."""
        result = create_export_controls_card(PlotTheme.DARK)
        # Structure: Div > [dcc.Download, collapsible_section (html.Div)]
        collapsible = result.children[1]
        assert isinstance(collapsible, html.Div)

    def test_has_two_children(self):
        """Test that card has download and collapsible section."""
        result = create_export_controls_card(PlotTheme.DARK)
        assert len(result.children) == 2
        assert isinstance(result.children[0], dcc.Download)
        assert isinstance(result.children[1], html.Div)
