# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic models for the top-level ``plot:`` section of ``AIPerfConfig``.

Mirrors the structure of ``src/aiperf/plot/default_plot_config.yaml`` 1:1 but
keeps preset bodies as ``dict[str, dict[str, Any]]`` because the existing
``PlotConfig._preset_to_plot_spec`` converter already validates them. Reuses
``ExperimentClassificationConfig`` from ``aiperf.plot.core.plot_specs``.
"""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import ConfigDict, Field, model_validator

from aiperf.config.base import BaseConfig
from aiperf.plot.core.plot_specs import ExperimentClassificationConfig

__all__ = [
    "PlotEnvelopeConfig",
    "PlotSettings",
    "PlotVisualization",
    "ServerMetricsDownsampling",
    "load_plot_envelope_from_path",
]


class ServerMetricsDownsampling(BaseConfig):
    """Server-metrics downsampling controls. Mirrors the YAML
    ``settings.server_metrics_downsampling`` block."""

    model_config = ConfigDict(extra="forbid")

    enabled: Annotated[
        bool,
        Field(
            default=True,
            description="Enable downsampling. Disable for raw Parquet rendering "
            "(slower, more detail).",
        ),
    ]

    window_size_seconds: Annotated[
        float,
        Field(
            default=5.0,
            gt=0,
            description="Window size in seconds for aggregating points. Larger "
            "values produce fewer points and faster rendering.",
        ),
    ]

    aggregation_method: Annotated[
        Literal["mean", "max", "min", "median"],
        Field(
            default="mean",
            description="How to combine points within each window. ``mean`` for "
            "smooth trends; ``max`` to preserve peaks; ``min`` for floors; "
            "``median`` for outlier-robust summaries.",
        ),
    ]


class PlotSettings(BaseConfig):
    """Top-level ``plot.settings`` block."""

    model_config = ConfigDict(extra="forbid")

    server_metrics_downsampling: Annotated[
        ServerMetricsDownsampling,
        Field(
            default_factory=ServerMetricsDownsampling,
            description="Server-metrics downsampling controls.",
        ),
    ]


class PlotVisualization(BaseConfig):
    """Top-level ``plot.visualization`` block. Holds the preset definitions and
    the per-mode default-name lists. Preset bodies are intentionally typed
    ``dict[str, Any]``: validation happens downstream in
    ``PlotConfig._preset_to_plot_spec``."""

    model_config = ConfigDict(extra="forbid")

    multi_run_defaults: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="Names of cross-variation comparison plots to render by "
            "default. Each name must be a key in ``multi_run_plots``.",
        ),
    ]

    single_run_defaults: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="Names of single-run time-series plots to render by "
            "default. Each name must be a key in ``single_run_plots``.",
        ),
    ]

    multi_run_plots: Annotated[
        dict[str, dict[str, Any]],
        Field(
            default_factory=dict,
            description="Cross-variation preset definitions. Bodies are validated "
            "downstream by ``PlotConfig._preset_to_plot_spec``.",
        ),
    ]

    single_run_plots: Annotated[
        dict[str, dict[str, Any]],
        Field(
            default_factory=dict,
            description="Single-run preset definitions. Bodies are validated "
            "downstream by ``PlotConfig._preset_to_plot_spec``.",
        ),
    ]

    @model_validator(mode="after")
    def _validate_default_names_exist(self) -> Self:
        # Why: catch typo'd default names at envelope-load time. _preset_to_plot_spec
        # would catch this too, but only at render time after the run completes.
        errors: list[str] = []
        for default_name in self.multi_run_defaults:
            if default_name not in self.multi_run_plots:
                hint = self._suggest(default_name, list(self.multi_run_plots))
                errors.append(
                    f"plot {default_name!r} listed in multi_run_defaults but not "
                    f"defined in multi_run_plots. Defined: "
                    f"{sorted(self.multi_run_plots)}.{hint}"
                )
        for default_name in self.single_run_defaults:
            if default_name not in self.single_run_plots:
                hint = self._suggest(default_name, list(self.single_run_plots))
                errors.append(
                    f"plot {default_name!r} listed in single_run_defaults but not "
                    f"defined in single_run_plots. Defined: "
                    f"{sorted(self.single_run_plots)}.{hint}"
                )
        if errors:
            raise ValueError("; ".join(errors))
        return self

    @staticmethod
    def _suggest(name: str, defined: list[str]) -> str:
        close = difflib.get_close_matches(name, defined, n=1, cutoff=0.6)
        return f" Did you mean {close[0]!r}?" if close else ""


class PlotEnvelopeConfig(BaseConfig):
    """Top-level ``plot:`` section of ``AIPerfConfig``. Mirrors
    ``default_plot_config.yaml`` exactly so the auto-plot callback can
    materialize this back to disk and feed it to the existing ``PlotConfig``
    file-path loader."""

    model_config = ConfigDict(extra="forbid")

    visualization: Annotated[
        PlotVisualization,
        Field(
            description="Preset definitions and per-mode default-name lists. "
            "Required field — ``plot: {}`` is rejected.",
        ),
    ]

    settings: Annotated[
        PlotSettings,
        Field(
            default_factory=PlotSettings,
            description="Global plot settings (downsampling, etc.).",
        ),
    ]

    experiment_classification: Annotated[
        ExperimentClassificationConfig | None,
        Field(
            default=None,
            description="Optional baseline-vs-treatment classification. When set, "
            "all multi-run plots auto-group by ``experiment_group`` regardless "
            "of preset-level ``groups:``.",
        ),
    ]


def load_plot_envelope_from_path(
    path: str | Path,
    *,
    source_dir: Path | None,
) -> PlotEnvelopeConfig:
    """Resolve a Form-A bare-string plot path and validate it.

    Args:
        path: The string from ``AIPerfConfig.plot`` (Form A).
        source_dir: Directory of the AIPerf YAML, used to resolve relative
            paths. Pass ``None`` when the config has no source file (stdin /
            dict loaders / K8s CR); relative paths are rejected in that case.

    Raises:
        ConfigurationError: If the path is relative without a ``source_dir``,
            does not exist, or fails to parse.
    """
    from ruamel.yaml import YAML
    from ruamel.yaml.error import YAMLError

    from aiperf.config.loader.errors import ConfigurationError

    raw = str(path)
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        if source_dir is None:
            raise ConfigurationError(
                f"plot path {raw!r} is relative but config has no source file; "
                "use an absolute path or inline the plot section",
            )
        candidate = (source_dir / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if not candidate.exists():
        raise ConfigurationError(
            f"plot path not found: {candidate} (from envelope plot: {raw!r}; "
            f"resolved relative to {source_dir})",
            file_path=candidate,
        )

    try:
        yaml = YAML(typ="safe")
        with candidate.open(encoding="utf-8") as f:
            data = yaml.load(f)
    except (YAMLError, OSError) as e:
        raise ConfigurationError(
            f"failed to parse plot YAML at {candidate}: {e}",
            file_path=candidate,
        ) from e

    if not isinstance(data, dict):
        raise ConfigurationError(
            f"plot YAML at {candidate} must contain a mapping at the top level, "
            f"got {type(data).__name__}",
            file_path=candidate,
        )

    return PlotEnvelopeConfig.model_validate(data)
