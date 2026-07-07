# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI command for analyzing mooncake traces."""

from __future__ import annotations

from pathlib import Path

from cyclopts import App

app = App(name="analyze-trace")


@app.default
def analyze_trace(
    input_file: Path,
    block_size: int = 512,
    output_file: Path | None = None,
) -> None:
    """Analyze a mooncake trace file for ISL/OSL distributions and cache hit rates.

    Args:
        input_file: Path to input mooncake trace JSONL file
        block_size: KV cache block size for analysis (default: 512)
        output_file: Optional output path for analysis report (JSON)
    """
    from aiperf.cli_utils import exit_on_error
    from aiperf.common.exceptions import DatasetError
    from aiperf.dataset.synthesis.cli import analyze_trace as _analyze_trace

    with exit_on_error(
        title="Error Analyzing Trace",
        show_traceback=False,
        quiet_for=(DatasetError,),
    ):
        _analyze_trace(input_file, block_size=block_size, output_file=output_file)
