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

from aiperf.config._comm import build_comm_config
from aiperf.plugin.enums import ServiceRunType

if TYPE_CHECKING:
    from aiperf.common.enums import AIPerfLogLevel, GPUTelemetryMode
    from aiperf.config.artifacts import ArtifactsConfig
    from aiperf.config.dataset import DatasetConfig
    from aiperf.config.phases import PhaseConfig
    from aiperf.config.zmq import BaseZMQCommunicationConfig
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
        if name not in self.datasets:  # type: ignore[attr-defined]
            raise KeyError(
                f"Dataset '{name}' not found. Available: {sorted(self.datasets.keys())}"  # type: ignore[attr-defined]
            )
        return self.datasets[name]  # type: ignore[attr-defined]

    def get_default_dataset_name(self) -> str:
        """Get the default dataset name (first dataset in the datasets dict)."""
        return next(iter(self.datasets.keys()))  # type: ignore[attr-defined]

    def get_default_dataset(self) -> DatasetConfig:
        """Get the default dataset (first dataset in the datasets dict)."""
        return next(iter(self.datasets.values()))  # type: ignore[attr-defined]

    def get_phase_dataset(self, phase: PhaseConfig) -> DatasetConfig:
        """Get the dataset for a specific phase.

        If the phase doesn't specify a dataset, uses the first dataset
        (the default).
        """
        dataset_name = phase.dataset or self.get_default_dataset_name()
        return self.get_dataset(dataset_name)

    def get_profiling_phases(self) -> dict[str, PhaseConfig]:
        """Get phase configs with exclude_from_results=False."""
        return {
            name: phase
            for name, phase in self.phases.items()  # type: ignore[attr-defined]
            if not phase.exclude_from_results
        }

    def get_warmup_phases(self) -> dict[str, PhaseConfig]:
        """Get warmup phase configs (excluded from results)."""
        return {
            name: phase
            for name, phase in self.phases.items()  # type: ignore[attr-defined]
            if phase.exclude_from_results
        }

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
    def worker_group_service_count(self) -> int:
        """Number of WorkerGroupManager services required by this run."""
        if self.runtime.service_run_type == ServiceRunType.KUBERNETES:  # type: ignore[attr-defined]
            import math

            from aiperf.common.environment import Environment

            workers_per_group = (
                self.runtime.workers_per_pod  # type: ignore[attr-defined]
                or Environment.WORKER.DEFAULT_WORKERS_PER_POD
            )
            requested_workers = self.runtime.workers or workers_per_group  # type: ignore[attr-defined]
            return max(1, math.ceil(requested_workers / workers_per_group))
        return 1

    @property
    def worker_group_declared_worker_capacity(self) -> int:
        """Worker capacity declared by the active group-manager adapter."""
        if self.runtime.service_run_type == ServiceRunType.KUBERNETES:  # type: ignore[attr-defined]
            from aiperf.common.environment import Environment

            return (
                self.runtime.workers_per_pod  # type: ignore[attr-defined]
                or Environment.WORKER.DEFAULT_WORKERS_PER_POD
            )
        from aiperf.workers.scaling import calculate_worker_count

        return calculate_worker_count(self)  # type: ignore[arg-type]

    @property
    def worker_group_declared_record_processor_capacity(self) -> int:
        """Record-processor capacity declared by the active group adapter."""
        if self.runtime.service_run_type == ServiceRunType.KUBERNETES:  # type: ignore[attr-defined]
            from aiperf.common.environment import Environment

            if self.runtime.record_processors_per_pod is not None:  # type: ignore[attr-defined]
                return self.runtime.record_processors_per_pod  # type: ignore[attr-defined]
            worker_capacity = self.worker_group_declared_worker_capacity
            return max(1, worker_capacity // Environment.RECORD.PROCESSOR_SCALE_FACTOR)
        if self.runtime.record_processors is not None:  # type: ignore[attr-defined]
            return self.runtime.record_processors  # type: ignore[attr-defined]
        from aiperf.workers.scaling import calculate_record_processor_count

        return calculate_record_processor_count(
            self.worker_group_declared_worker_capacity
        )

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
    def output(self) -> ArtifactsConfig:
        """Alias for artifacts config (convenience access via config.output.*)."""
        return self.artifacts  # type: ignore[attr-defined]

    @property
    def server_metrics_disabled(self) -> bool:
        """True if server metrics collection is disabled."""
        return not self.server_metrics.enabled  # type: ignore[attr-defined]

    @property
    def server_metrics_formats(self) -> list:
        """Server metrics export formats."""
        return self.server_metrics.formats  # type: ignore[attr-defined]

    @property
    def benchmark_id(self) -> str:
        """Benchmark ID from artifacts config."""
        return self.artifacts.benchmark_id  # type: ignore[attr-defined]
