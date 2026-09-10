# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helper methods and convenience properties for BenchmarkConfig.

Split out of `config.py` to keep the model file under the ergonomics
line limit. This mixin relies on fields defined on `BenchmarkConfig`
(models, datasets, phases, runtime, logging, gpu_telemetry, artifacts,
server_metrics) and exists only to host accessor logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.config.comm.build import build_comm_config

if TYPE_CHECKING:
    from aiperf.common.enums import (
        AIPerfLogLevel,
        CacheBustTarget,
        GPUTelemetryMode,
        PromptCorpus,
        ServerMetricsFormat,
    )
    from aiperf.config.comm import BaseZMQCommunicationConfig
    from aiperf.config.dataset import DatasetConfig
    from aiperf.config.phases import PhaseConfig
    from aiperf.plugin.enums import UIType


class BenchmarkHelpersMixin:
    """Helper methods + convenience properties for BenchmarkConfig."""

    # ==========================================================================
    # HELPER METHODS
    # ==========================================================================

    def get_model_names(self) -> list[str]:
        """Get list of all model names from the configuration."""
        return [item.name for item in self.models.items]  # type: ignore[attr-defined]

    def get_dataset(self, name: str) -> DatasetConfig:
        """Get a dataset by name.

        Raises:
            KeyError: If dataset not found.
        """
        for d in self.datasets:  # type: ignore[attr-defined]
            if d.name == name:
                return d
        available = [d.name for d in self.datasets]  # type: ignore[attr-defined]
        raise KeyError(f"Dataset '{name}' not found. Available: {sorted(available)}")

    def get_default_dataset_name(self) -> str:
        """Returns the name of the first dataset in the list (the default)."""
        return self.datasets[0].name  # type: ignore[attr-defined]

    def get_default_dataset(self) -> DatasetConfig:
        """Get the default dataset (first dataset in the list)."""
        return self.datasets[0]  # type: ignore[attr-defined]

    def get_profiling_phases(self) -> list[PhaseConfig]:
        """Get profiling-kind phase configs."""
        return [
            phase
            for phase in self.phases  # type: ignore[attr-defined]
            if phase.kind == "profiling"
        ]

    def get_warmup_phases(self) -> list[PhaseConfig]:
        """Get warmup-kind phase configs."""
        return [
            phase
            for phase in self.phases  # type: ignore[attr-defined]
            if phase.kind == "warmup"
        ]

    def get_cache_bust_target(self) -> CacheBustTarget:
        """Resolve the active dataset's cache-bust target (single engine read).

        The DAG/agentic engine threads this one value into the
        ``BranchOrchestrator`` / conversation source. Home differs by dataset
        kind:

        - synthetic: ``default_dataset.prompts.cache_bust.target`` (guard a
          ``None`` ``prompts`` -> NONE);
        - file: ``default_dataset.cache_bust.target``;
        - otherwise NONE.
        """
        from aiperf.common.enums import CacheBustTarget

        dataset = self.get_default_dataset()
        prompts = getattr(dataset, "prompts", None)
        if prompts is not None:
            cache_bust = getattr(prompts, "cache_bust", None)
            if cache_bust is not None:
                return cache_bust.target
        cache_bust = getattr(dataset, "cache_bust", None)
        if cache_bust is not None:
            return cache_bust.target
        return CacheBustTarget.NONE

    def get_prompt_corpus(self) -> PromptCorpus | None:
        """Resolve the active dataset's authored ``prompts.corpus``."""
        dataset = self.get_default_dataset()
        prompts = getattr(dataset, "prompts", None)
        if prompts is None:
            return None
        return getattr(prompts, "corpus", None)

    def get_system_prompt(self) -> str | None:
        """Resolve the active dataset's verbatim system prompt text, if any.

        Reads ``SystemPromptMixin.resolved_system_prompt``, which every dataset
        variant carries and which already collapsed the
        ``system_prompt`` / ``system_prompt_file`` distinction during
        validation. Returns ``None`` when the user configured neither.
        """
        return getattr(self.get_default_dataset(), "resolved_system_prompt", None)

    # ==========================================================================
    # CONVENIENCE PROPERTIES
    # ==========================================================================

    @property
    def comm_config(self) -> BaseZMQCommunicationConfig:
        """Get the ZMQ communication configuration.

        Cached so all callers get the same IPC paths. Without caching,
        each access creates a new ZMQIPCConfig with a fresh temp directory.
        """
        if not hasattr(self, "_comm_config_cache"):
            object.__setattr__(self, "_comm_config_cache", build_comm_config(self))  # type: ignore[arg-type]
        return self._comm_config_cache

    @property
    def ui_type(self) -> UIType:
        """Get the UI type (shortcut for runtime.ui)."""
        return self.runtime.ui  # type: ignore[attr-defined]

    @property
    def workers_max(self) -> int | None:
        """Maximum number of workers, or None for auto-detect."""
        return self.runtime.workers  # type: ignore[attr-defined]

    @property
    def record_processor_service_count(self) -> int | None:
        """Number of record processors, or None for auto-detect."""
        return self.runtime.record_processors  # type: ignore[attr-defined]

    @property
    def log_level(self) -> AIPerfLogLevel:
        """Get the logging level (shortcut for logging.level)."""
        return self.logging.level  # type: ignore[attr-defined]

    @property
    def verbose(self) -> bool:
        """True if logging level is DEBUG or more verbose."""
        from aiperf.common.enums import AIPerfLogLevel

        return self.logging.level in (AIPerfLogLevel.DEBUG, AIPerfLogLevel.TRACE)  # type: ignore[attr-defined]

    @property
    def extra_verbose(self) -> bool:
        """True if logging level is TRACE."""
        from aiperf.common.enums import AIPerfLogLevel

        return self.logging.level == AIPerfLogLevel.TRACE  # type: ignore[attr-defined]

    @property
    def gpu_telemetry_disabled(self) -> bool:
        """True if GPU telemetry collection is disabled."""
        return not self.gpu_telemetry.enabled  # type: ignore[attr-defined]

    @property
    def gpu_telemetry_mode(self) -> GPUTelemetryMode:
        """GPU telemetry display mode."""
        return self.gpu_telemetry.mode  # type: ignore[attr-defined]

    @gpu_telemetry_mode.setter
    def gpu_telemetry_mode(self, value: GPUTelemetryMode) -> None:
        self.gpu_telemetry.mode = value  # type: ignore[attr-defined]

    @property
    def server_metrics_disabled(self) -> bool:
        """True if server metrics collection is disabled."""
        return not self.server_metrics.enabled  # type: ignore[attr-defined]

    @property
    def server_metrics_formats(self) -> list[ServerMetricsFormat]:
        """Server metrics export formats."""
        return self.server_metrics.formats  # type: ignore[attr-defined]
