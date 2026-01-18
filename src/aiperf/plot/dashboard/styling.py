# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Styling and theme configuration for the AIPerf dashboard.

This module provides consistent NVIDIA-branded styling for both light and dark themes.
"""

from aiperf.plot.constants import (
    DARK_THEME_COLORS,
    LIGHT_THEME_COLORS,
    NVIDIA_DARK,
    NVIDIA_GRAY,
    NVIDIA_GREEN,
    NVIDIA_WHITE,
    PLOT_FONT_FAMILY,
    PlotTheme,
)


def get_dropdown_css(theme: PlotTheme) -> str:
    """
    Generate CSS specifically for dropdown components.

    This is used for dynamic theme updates via callback.

    Args:
        theme: PlotTheme.LIGHT or PlotTheme.DARK

    Returns:
        CSS string for dropdown styling
    """
    # Get colors for both themes for CSS variables
    dark_colors = get_theme_colors(PlotTheme.DARK)
    light_colors = get_theme_colors(PlotTheme.LIGHT)

    return f"""
        /* CSS Variables for theme switching */
        .theme-dark {{
            --bg-color: {dark_colors["background"]};
            --text-color: {dark_colors["text"]};
            --paper-color: {dark_colors["paper"]};
            --border-color: {dark_colors["border"]};
            --grid-color: {dark_colors["grid"]};
        }}

        .theme-light {{
            --bg-color: {light_colors["background"]};
            --text-color: {light_colors["text"]};
            --paper-color: {light_colors["paper"]};
            --border-color: {light_colors["border"]};
            --grid-color: {light_colors["grid"]};
        }}

        /* Dropdown styling using CSS variables */
        .Select-control {{
            background-color: var(--paper-color) !important;
            border-color: var(--border-color) !important;
        }}

        .Select-value, .Select-value-label {{
            color: var(--text-color) !important;
        }}

        .Select-placeholder {{
            color: {NVIDIA_GRAY} !important;
        }}

        .Select-menu-outer {{
            background-color: var(--paper-color) !important;
            border-color: var(--border-color) !important;
        }}

        .Select-option {{
            background-color: var(--paper-color) !important;
            color: var(--text-color) !important;
        }}

        .Select-option.is-focused {{
            background-color: {NVIDIA_GREEN} !important;
            color: {NVIDIA_WHITE} !important;
        }}

        .Select-option.is-selected {{
            background-color: var(--bg-color) !important;
            color: {NVIDIA_GREEN} !important;
        }}

        .Select-input > input {{
            color: var(--text-color) !important;
        }}

        /* Modal styling using CSS variables (works for ALL modals) */
        .modal-content {{
            background-color: var(--bg-color) !important;
            color: var(--text-color) !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4) !important;
            border: 1px solid var(--border-color) !important;
        }}

        .modal-header {{
            background-color: var(--paper-color) !important;
            color: var(--text-color) !important;
            border-bottom: 1px solid var(--border-color) !important;
        }}

        /* Modal close button (X) */
        .theme-dark .modal .btn-close {{
            filter: invert(1) brightness(1.5);
        }}

        .modal-body {{
            background-color: var(--bg-color) !important;
            color: var(--text-color) !important;
        }}

        .modal-footer {{
            background-color: var(--paper-color) !important;
            color: var(--text-color) !important;
            border-top: 1px solid var(--border-color) !important;
        }}

        .modal-title {{
            color: var(--text-color) !important;
        }}

        /* Modal backdrop - hide it completely when backdrop=False */
        .modal-backdrop {{
            display: none !important;
        }}

        /* Modal container - transparent background, no blocking */
        .modal {{
            background-color: transparent !important;
            pointer-events: none !important;
        }}

        /* Modal dialog - allow interactions */
        .modal-dialog {{
            z-index: 1050 !important;
            pointer-events: auto !important;
        }}

        /* Ensure modal shows above all content */
        .modal.show {{
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            background-color: transparent !important;
        }}

        /* Modal dropdown styling */
        .modal-body .Select-control {{
            background-color: var(--paper-color) !important;
            border-color: var(--border-color) !important;
        }}

        .modal-body .Select-value,
        .modal-body .Select-value-label {{
            color: var(--text-color) !important;
        }}

        .modal-body .Select-menu-outer {{
            background-color: var(--paper-color) !important;
            border-color: var(--border-color) !important;
        }}

        .modal-body .Select-option {{
            background-color: var(--paper-color) !important;
            color: var(--text-color) !important;
        }}

        /* Modal labels */
        .modal-body label {{
            color: var(--text-color) !important;
        }}

        /* Modal summary/details elements */
        .modal-body summary {{
            color: var(--text-color) !important;
        }}

        /* Modal text input fields */
        .modal-body input[type="text"],
        .modal-body .dash-input,
        .modal-body #edit-plot-title,
        .modal-body #custom-plot-title,
        input.dash-input {{
            background-color: var(--paper-color) !important;
            color: var(--text-color) !important;
            border: 1px solid var(--border-color) !important;
            padding: 6px 8px;
            border-radius: 4px;
        }}

        .modal-body input[type="text"]:focus,
        .modal-body .dash-input:focus,
        .modal-body #edit-plot-title:focus,
        .modal-body #custom-plot-title:focus,
        input.dash-input:focus {{
            outline: none;
            border-color: {NVIDIA_GREEN} !important;
        }}
    """


def get_theme_colors(theme: PlotTheme) -> dict[str, str]:
    """
    Get color scheme for the specified theme.

    Args:
        theme: PlotTheme.LIGHT or PlotTheme.DARK

    Returns:
        Dictionary mapping color roles to hex colors
    """
    if theme == PlotTheme.DARK:
        return DARK_THEME_COLORS
    return LIGHT_THEME_COLORS


def get_scoped_theme_css(theme: PlotTheme) -> str:
    """
    Generate theme-scoped CSS for a specific theme.

    All selectors are scoped under .theme-light or .theme-dark class.

    Args:
        theme: PlotTheme.LIGHT or PlotTheme.DARK

    Returns:
        CSS string scoped to theme class
    """
    colors = get_theme_colors(theme)
    theme_class = f".theme-{theme.value}"

    return f"""
        /* Theme-scoped styling for {theme.value} mode */
        {theme_class} {{
            background-color: {colors["background"]};
            color: {colors["text"]};
        }}

        {theme_class} #react-entry-point {{
            background-color: {colors["background"]} !important;
        }}

        {theme_class} ._dash-loading {{
            background-color: {colors["background"]} !important;
        }}

        {theme_class} div {{
            color: {colors["text"]};
        }}

        {theme_class} .tabs {{
            background-color: {colors["background"]} !important;
            border: none !important;
        }}

        /* Dropdown styling for {theme.value} mode */
        {theme_class} .Select-control {{
            background-color: {colors["paper"]} !important;
            border-color: {colors["border"]} !important;
        }}

        {theme_class} .Select-value,
        {theme_class} .Select-value-label {{
            color: {colors["text"]} !important;
        }}

        {theme_class} .Select-placeholder {{
            color: {NVIDIA_GRAY} !important;
        }}

        {theme_class} .Select-menu-outer {{
            background-color: {colors["paper"]} !important;
            border-color: {colors["border"]} !important;
        }}

        {theme_class} .Select-option {{
            background-color: {colors["paper"]} !important;
            color: {colors["text"]} !important;
        }}

        {theme_class} .Select-option.is-focused {{
            background-color: {NVIDIA_GREEN} !important;
            color: {NVIDIA_WHITE} !important;
        }}

        {theme_class} .Select-option.is-selected {{
            background-color: {colors["background"]} !important;
            color: {NVIDIA_GREEN} !important;
        }}

        {theme_class} .Select-input > input {{
            color: {colors["text"]} !important;
        }}

        /* Bootstrap switch labels */
        {theme_class} .form-check-label {{
            color: {colors["text"]} !important;
        }}

        {theme_class} .custom-control-label {{
            color: {colors["text"]} !important;
        }}

        /* Run selector container */
        {theme_class} .run-selector-container {{
            background: {colors["paper"]} !important;
        }}

        /* Custom scrollbar for run selector (webkit browsers) */
        {theme_class} .run-selector-container::-webkit-scrollbar {{
            width: 8px;
        }}

        {theme_class} .run-selector-container::-webkit-scrollbar-track {{
            background: {colors["paper"]} !important;
            border-radius: 4px;
        }}

        {theme_class} .run-selector-container::-webkit-scrollbar-thumb {{
            background: {colors["border"]} !important;
            border-radius: 4px;
        }}

        {theme_class} .run-selector-container::-webkit-scrollbar-thumb:hover {{
            background: {NVIDIA_GRAY} !important;
        }}

        /* General labels */
        {theme_class} label {{
            color: {colors["text"]} !important;
        }}

        /* Summary/details elements */
        {theme_class} summary {{
            color: {colors["text"]} !important;
        }}

        /* Checklist labels */
        {theme_class} .run-selector-container label {{
            color: {colors["text"]} !important;
        }}

        {theme_class} .run-selector-container input[type="checkbox"] + label {{
            color: {colors["text"]} !important;
        }}

        /* Plot container with hover effects */
        .plot-container {{
            position: relative;
            min-height: 400px;
            width: 100%;
            max-width: 100%;
            box-sizing: border-box;
            overflow: visible;
        }}

        /* Flexible grid sizing - plots can span 1 or 2 columns */
        .plot-container.size-half {{
            grid-column: span 1;  /* 1 of 2 = 50% */
        }}

        .plot-container.size-full {{
            grid-column: span 2;  /* 2 of 2 = 100% */
        }}

        /* Settings button (⚙) - green by default in light theme, shown on hover */
        .plot-settings-btn {{
            position: absolute;
            top: 40px;
            right: 8px;
            width: 28px;
            height: 28px;
            background: rgba(118, 185, 0, 0.2);
            color: {NVIDIA_GREEN};
            border: 1px solid rgba(118, 185, 0, 0.4);
            border-radius: 50%;
            cursor: pointer;
            opacity: 0;
            transition: opacity 0.2s, transform 0.2s, background 0.2s;
            z-index: 1000;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            font-weight: normal;
            line-height: 1;
            padding: 0;
            backdrop-filter: blur(4px);
        }}

        .plot-container:hover .plot-settings-btn {{
            opacity: 1;
        }}

        .plot-settings-btn:hover {{
            transform: scale(1.1) rotate(45deg);
            background: rgba(118, 185, 0, 0.4);
            border-color: rgba(118, 185, 0, 0.6);
        }}

        /* Hide plot button (eye icon) - positioned below settings button */
        .plot-hide-btn {{
            position: absolute;
            top: 75px;
            right: 8px;
            width: 28px;
            height: 28px;
            background: rgba(118, 185, 0, 0.2);
            color: {NVIDIA_GREEN};
            border: 1px solid rgba(118, 185, 0, 0.4);
            border-radius: 50%;
            cursor: pointer;
            opacity: 0;
            transition: opacity 0.2s, transform 0.2s, background 0.2s;
            z-index: 1000;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            font-weight: normal;
            line-height: 1;
            padding: 0;
            backdrop-filter: blur(4px);
        }}

        .plot-container:hover .plot-hide-btn {{
            opacity: 1;
        }}

        .plot-hide-btn:hover {{
            transform: scale(1.1);
            background: rgba(255, 0, 0, 0.3);
            border-color: rgba(255, 0, 0, 0.5);
            color: #ff5555;
        }}

        /* Resize handle for adjusting plot size */
        .resize-handle {{
            position: absolute;
            bottom: 4px;
            right: 4px;
            width: 20px;
            height: 20px;
            cursor: pointer;
            opacity: 0;
            z-index: 1001;
            transition: opacity 0.2s;
            user-select: none;
            color: {NVIDIA_GRAY};
            font-size: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        .plot-container:hover .resize-handle {{
            opacity: 0.6;
        }}

        .resize-handle:hover {{
            opacity: 1 !important;
            color: {NVIDIA_GREEN};
        }}


        /* Add plot button slot */
        {theme_class} .plot-add-slot {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background: {colors["paper"]};
            border: 2px dashed {colors["border"]};
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
            min-height: 300px;
            width: 100%;
            max-width: 100%;
            box-sizing: border-box;
        }}

        {theme_class} .plot-add-slot:hover {{
            border-color: {NVIDIA_GREEN};
            background: {colors["background"]};
        }}

        /* Context menu styles */
        .context-menu-item {{
            padding: 8px 16px;
            font-size: 13px;
            font-family: {PLOT_FONT_FAMILY};
            color: {colors["text"]};
            cursor: pointer;
            transition: background 0.2s;
            user-select: none;
        }}

        .context-menu-item:hover {{
            background: {colors["background"]};
        }}

        .context-menu-item-danger {{
            color: #e74c3c;
        }}

        .context-menu-item-danger:hover {{
            background: rgba(231, 76, 60, 0.1);
        }}

        {theme_class} .plot-add-icon {{
            font-size: 48px;
            color: {NVIDIA_GREEN};
            margin-bottom: 8px;
        }}

        {theme_class} .plot-add-text {{
            color: {colors["text"]};
            font-size: 14px;
            font-family: {PLOT_FONT_FAMILY};
        }}

        /* Radio button styling */
        {theme_class} input[type="radio"] {{
            accent-color: {NVIDIA_GREEN};
            cursor: pointer;
        }}

        {theme_class} .dash-radioitems label {{
            color: {colors["text"]} !important;
            cursor: pointer;
            padding-left: 4px;
        }}

        {theme_class} .dash-radioitems label:hover {{
            color: {NVIDIA_GREEN} !important;
        }}

        /* Checkbox styling */
        input[type="checkbox"] {{
            accent-color: {NVIDIA_GREEN};
        }}

        /* Cursor pointer for clickable plot points */
        {theme_class} .plot-container .js-plotly-plot .plotly .scatterlayer .trace .points path {{
            cursor: pointer;
        }}
    """


def get_all_themes_css() -> str:
    """
    Generate CSS for both light and dark themes.

    This function combines scoped CSS for both themes plus global styles
    that don't change with theme. Use this for injecting all CSS at startup.

    Returns:
        Complete CSS string with both theme variants
    """
    global_css = f"""
        /* Global styling (theme-independent) */
        html, body {{
            margin: 0 !important;
            padding: 0 !important;
            font-family: {PLOT_FONT_FAMILY};
            height: 100%;
        }}

        /* Plot container with hover effects */
        .plot-container {{
            position: relative;
            min-height: 400px;
            width: 100%;
            max-width: 100%;
            box-sizing: border-box;
            overflow: visible;
        }}

        /* Flexible grid sizing - plots can span 1 or 2 columns */
        .plot-container.size-half {{
            grid-column: span 1;  /* 1 of 2 = 50% */
        }}

        .plot-container.size-full {{
            grid-column: span 2;  /* 2 of 2 = 100% */
        }}

        /* Settings button (⚙) - white in dark theme, shown on hover */
        .plot-settings-btn {{
            position: absolute;
            top: 40px;
            right: 8px;
            width: 28px;
            height: 28px;
            background: rgba(255, 255, 255, 0.15);
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            cursor: pointer;
            opacity: 0;
            transition: opacity 0.2s, transform 0.2s, background 0.2s;
            z-index: 1000;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            font-weight: normal;
            line-height: 1;
            padding: 0;
            backdrop-filter: blur(4px);
        }}

        .plot-container:hover .plot-settings-btn {{
            opacity: 1;
        }}

        .plot-settings-btn:hover {{
            transform: scale(1.1) rotate(45deg);
            background: rgba(118, 185, 0, 0.3);
            border-color: rgba(118, 185, 0, 0.5);
        }}

        /* Hide plot button (eye icon) - positioned below settings button */
        .plot-hide-btn {{
            position: absolute;
            top: 75px;
            right: 8px;
            width: 28px;
            height: 28px;
            background: rgba(255, 255, 255, 0.15);
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            cursor: pointer;
            opacity: 0;
            transition: opacity 0.2s, transform 0.2s, background 0.2s;
            z-index: 1000;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            font-weight: normal;
            line-height: 1;
            padding: 0;
            backdrop-filter: blur(4px);
        }}

        .plot-container:hover .plot-hide-btn {{
            opacity: 1;
        }}

        .plot-hide-btn:hover {{
            transform: scale(1.1);
            background: rgba(255, 0, 0, 0.3);
            border-color: rgba(255, 0, 0, 0.5);
            color: #ff8888;
        }}

        /* Resize handle for adjusting plot size */
        .resize-handle {{
            position: absolute;
            bottom: 4px;
            right: 4px;
            width: 20px;
            height: 20px;
            cursor: pointer;
            opacity: 0;
            z-index: 1001;
            transition: opacity 0.2s;
            user-select: none;
            color: {NVIDIA_GRAY};
            font-size: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        .plot-container:hover .resize-handle {{
            opacity: 0.6;
        }}

        .resize-handle:hover {{
            opacity: 1 !important;
            color: {NVIDIA_GREEN};
        }}


        /* Checkbox styling */
        input[type="checkbox"] {{
            accent-color: {NVIDIA_GREEN};
        }}
    """

    light_css = get_scoped_theme_css(PlotTheme.LIGHT)
    dark_css = get_scoped_theme_css(PlotTheme.DARK)

    return global_css + "\n\n" + light_css + "\n\n" + dark_css


def get_header_style(theme: PlotTheme) -> dict:
    """Get header container style."""
    colors = get_theme_colors(theme)
    return {
        "display": "flex",
        "align-items": "center",
        "padding": "16px 24px",
        "background": NVIDIA_DARK if theme == PlotTheme.DARK else colors["background"],
        "box-shadow": "0 2px 8px rgba(0,0,0,0.1)",
        "border-bottom": f"1px solid {colors['border']}",
    }


def get_sidebar_style(theme: PlotTheme, collapsed: bool = False) -> dict:
    """
    Get sidebar style with collapse support.

    Args:
        theme: Plot theme
        collapsed: Whether sidebar is collapsed

    Returns:
        Dictionary of CSS properties
    """
    colors = get_theme_colors(theme)

    if collapsed:
        return {
            "width": "0px",
            "min-width": "0px",
            "padding": "0",
            "overflow": "hidden",
            "transition": "width 0.3s ease-in-out",
            "background": colors["background"],
            "border-right": f"1px solid {colors['border']}",
            "height": "calc(100vh - 70px)",
            "flex-shrink": "0",
            "pointer-events": "none",
        }

    return {
        "width": "300px",
        "min-width": "300px",
        "background": colors["background"],
        "height": "calc(100vh - 70px)",
        "overflow-y": "auto",
        "overflow-x": "hidden",
        "border-right": f"1px solid {colors['border']}",
        "padding": "20px",
        "font-family": PLOT_FONT_FAMILY,
        "flex-shrink": "0",
        "transition": "width 0.3s ease-in-out",
        "display": "flex",
        "flex-direction": "column",
        "align-items": "center",
        "pointer-events": "auto",
    }


def get_main_area_style(theme: PlotTheme) -> dict:
    """Get main area style."""
    colors = get_theme_colors(theme)
    return {
        "flex": "1",
        "background": colors["background"],
        "font-family": PLOT_FONT_FAMILY,
    }


def get_button_style(theme: PlotTheme, variant: str = "primary") -> dict:
    """
    Get button style.

    Args:
        theme: Plot theme
        variant: Button variant ('primary', 'secondary', 'outline')
    """
    colors = get_theme_colors(theme)

    if variant == "primary":
        return {
            "width": "100%",
            "padding": "8px 12px",
            "background": NVIDIA_GREEN,
            "color": NVIDIA_WHITE,
            "border": "none",
            "border-radius": "6px",
            "cursor": "pointer",
            "font-weight": "600",
            "font-size": "12px",
            "font-family": PLOT_FONT_FAMILY,
            "margin-bottom": "8px",
            "transition": "all 0.2s",
        }
    elif variant == "secondary":
        return {
            "width": "100%",
            "padding": "8px 12px",
            "background": colors["paper"],
            "color": colors["text"],
            "border": f"1px solid {colors['border']}",
            "border-radius": "6px",
            "cursor": "pointer",
            "font-weight": "500",
            "font-size": "12px",
            "font-family": PLOT_FONT_FAMILY,
            "margin-bottom": "8px",
            "transition": "all 0.2s",
        }
    else:  # outline
        return {
            "width": "100%",
            "padding": "6px 12px",
            "background": "transparent",
            "color": NVIDIA_GREEN,
            "border": f"2px solid {NVIDIA_GREEN}",
            "border-radius": "6px",
            "cursor": "pointer",
            "font-weight": "600",
            "font-size": "11px",
            "font-family": PLOT_FONT_FAMILY,
            "margin-bottom": "6px",
            "transition": "all 0.2s",
        }


def get_label_style(theme: PlotTheme) -> dict:
    """Get form label style."""
    colors = get_theme_colors(theme)
    return {
        "font-size": "11px",
        "font-weight": "600",
        "color": colors["text"],
        "display": "block",
        "font-family": PLOT_FONT_FAMILY,
        "margin-bottom": "4px",
    }


def get_section_header_style(theme: PlotTheme) -> dict:
    """Get section header style."""
    return {
        "font-size": "11px",
        "font-weight": "700",
        "text-transform": "uppercase",
        "letter-spacing": "1px",
        "color": NVIDIA_GRAY,
        "margin-bottom": "16px",
        "font-family": PLOT_FONT_FAMILY,
        "padding-bottom": "8px",
        "border-bottom": f"2px solid {NVIDIA_GREEN}",
    }
