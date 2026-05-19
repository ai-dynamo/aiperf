# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end: envelope -> callback -> run_plot_controller -> PNGs on disk.

Drives the full ``build_auto_plot_callback`` pipeline against a real
fixture artifact directory (``tests/unit/plot/fixtures/qwen_concurrency1``)
and asserts that the per-run plots that land in ``<run>/plots/`` match
exactly the envelope's ``single_run_defaults`` -- i.e. the envelope's
listed plot is produced AND plots present in the shipped default but
absent from the envelope are NOT produced.

The materialization contract (envelope -> YAML on disk, round-trip) is
fully covered by ``tests/unit/plot/test_auto_plot_materialization.py``;
this test is the integration-level check that the envelope actually
drives the plot-selection downstream.

If the fixture artifact directory cannot be located (e.g. layout drift),
the test skips with a clear reason rather than producing a misleading
PASS or a confusing FAIL.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aiperf.config.plot import PlotEnvelopeConfig
from aiperf.plot.auto_plot import build_auto_plot_callback


@pytest.fixture
def artifact_dir(tmp_path: Path) -> Path:
    """Copy a real single-run fixture into ``tmp_path`` and return its path.

    Uses the same fixture data the rest of ``tests/unit/plot/`` relies on
    so the JSONL/JSON shapes are guaranteed to be parseable by the plot
    controller. Skips with a clear reason if the source directory is not
    where we expect.
    """
    src = Path(__file__).parent / "fixtures" / "qwen_concurrency1"
    if not src.exists():
        pytest.skip(f"fixture artifact dir not found at {src}")
    required = ["profile_export.jsonl", "profile_export_aiperf.json"]
    missing = [name for name in required if not (src / name).exists()]
    if missing:
        pytest.skip(f"fixture missing required files: {missing}")
    dst = tmp_path / "run"
    shutil.copytree(src, dst)
    return dst


def test_envelope_drives_only_listed_plots(artifact_dir: Path) -> None:
    """Per-run plots written to ``<run>/plots/`` match the envelope.

    The envelope declares one single-run plot (``ttft_over_time``); the
    shipped default config also lists ``ttft_timeline`` and others under
    ``single_run_defaults``. After running the callback, only
    ``ttft_over_time.png`` should appear -- the shipped extras must NOT
    leak through because the materialized envelope replaces the default.
    """
    envelope = PlotEnvelopeConfig.model_validate(
        {
            "visualization": {
                "single_run_defaults": ["ttft_over_time"],
                "single_run_plots": {
                    "ttft_over_time": {
                        "type": "scatter",
                        "x": "request_number",
                        "y": "time_to_first_token",
                        "title": "TTFT Per Request",
                    },
                },
            },
        }
    )
    cb = build_auto_plot_callback(plot_required=True, plot_envelope=envelope)
    run = MagicMock()
    run.artifact_dir = artifact_dir

    cb(run)

    plots_dir = artifact_dir / "plots"
    assert plots_dir.exists(), f"expected plots dir at {plots_dir}"
    pngs = sorted(p.name for p in plots_dir.glob("*.png"))
    assert "ttft_over_time.png" in pngs, (
        f"envelope-listed plot missing from {plots_dir}: got {pngs}"
    )
    # Plots in the shipped default's single_run_defaults but absent from the
    # envelope must not appear -- the materialized envelope replaces the
    # shipped default entirely, it does not merge.
    assert "ttft_timeline.png" not in pngs, (
        f"non-envelope plot leaked into {plots_dir}: got {pngs}"
    )


def test_replay_via_aiperf_plot_picks_up_materialized_yaml(
    artifact_dir: Path,
) -> None:
    """Reproducibility: callback writes the materialized YAML, then a fresh
    ``run_plot_controller`` (without ``config=``) auto-detects the receipt
    via PlotConfig's Priority 1.5 and renders the same envelope-driven plot
    set, not the shipped default's.
    """
    from aiperf.plot.cli_runner import run_plot_controller

    envelope = PlotEnvelopeConfig.model_validate(
        {
            "visualization": {
                "single_run_defaults": ["ttft_over_time"],
                "single_run_plots": {
                    "ttft_over_time": {
                        "type": "scatter",
                        "x": "request_number",
                        "y": "time_to_first_token",
                        "title": "TTFT Replay Test",
                    },
                },
            },
        }
    )
    cb = build_auto_plot_callback(plot_required=True, plot_envelope=envelope)
    run = MagicMock()
    run.artifact_dir = artifact_dir
    cb(run)

    # The materialized receipt must exist after the callback.
    assert (artifact_dir / ".aiperf-plot-config.yaml").exists()

    # Wipe the prior plots so we can verify the replay produces them fresh
    # via Priority 1.5 auto-detect, not stale files.
    plots_dir = artifact_dir / "plots"
    if plots_dir.exists():
        for png in plots_dir.glob("*.png"):
            png.unlink()

    # Replay: aiperf plot <run> with NO --config flag -- relies entirely on
    # PlotConfig auto-detecting the materialized envelope from the run dir.
    run_plot_controller(paths=[str(artifact_dir)])

    pngs = sorted(p.name for p in plots_dir.glob("*.png"))
    assert "ttft_over_time.png" in pngs, (
        f"envelope-listed plot missing on replay: got {pngs}"
    )
    assert "ttft_timeline.png" not in pngs, (
        f"shipped-default plot leaked on replay (auto-detect failed): got {pngs}"
    )
