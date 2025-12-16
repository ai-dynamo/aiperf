# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PNG-specific base exporter functionality.

Provides PNG export methods using kaleido for static image generation.
"""

from pathlib import Path

import plotly.graph_objects as go

from aiperf.plot.constants import (
    DEFAULT_PLOT_DPI,
    DEFAULT_PLOT_HEIGHT,
    DEFAULT_PLOT_WIDTH,
)
from aiperf.plot.exporters.base import BaseExporter


class BasePNGExporter(BaseExporter):
    """
    Base class for PNG export functionality.

    Provides PNG-specific methods like figure export using kaleido
    and summary file generation.
    """

    def _export_figure(self, fig: go.Figure, path: Path) -> None:
        """
        Export a Plotly figure to PNG file using kaleido.

        Args:
            fig: Plotly Figure object
            path: Output file path
        """
        fig.write_image(
            str(path),
            width=DEFAULT_PLOT_WIDTH,
            height=DEFAULT_PLOT_HEIGHT,
            scale=DEFAULT_PLOT_DPI / 100,
        )

    def _create_summary_file(self, generated_files: list[Path]) -> None:
        """
        Create a summary text file listing all generated plots.

        Args:
            generated_files: List of generated file paths
        """
        summary_path = self.output_dir / "summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("AIPerf Plot Export Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated {len(generated_files)} plots:\n\n")
            for path in generated_files:
                f.write(f"  - {path.name}\n")
            f.write(f"\nOutput directory: {self.output_dir}\n")
        self.debug(f"Summary saved to {summary_path}")
