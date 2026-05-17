# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Matplotlib-based PNG export for code-gen reporting.

The Matplotlib renderer is a standalone function called directly by code-gen
reporting, not through the Plotly handler path. This module provides the
integration entry point that constructs the data contract and delegates to
the renderer.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from aiperf.plot.constants import DEFAULT_PLOT_DPI, PlotTheme
from aiperf.plot.models.uncertainty import LatencyThroughputUncertaintyData
from aiperf.plot.renderers import render_matplotlib_uncertainty


def export_uncertainty_matplotlib(
    data: LatencyThroughputUncertaintyData,
    output_path: Path,
    theme: PlotTheme = PlotTheme.LIGHT,
    dpi: int = DEFAULT_PLOT_DPI,
) -> Path:
    """Export uncertainty plot as PNG using the Matplotlib renderer.

    This is the code-gen reporting entry point for uncertainty plots.
    It renders the figure via Matplotlib (not Plotly/Kaleido) and writes
    directly to disk.

    Args:
        data: Validated data contract with benchmark points and metadata.
        output_path: Destination file path for the PNG.
        theme: NVIDIA brand theme (light or dark).
        dpi: Output resolution in dots per inch.

    Returns:
        The output_path that was written.
    """
    fig = render_matplotlib_uncertainty(data, theme)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)
    return output_path
