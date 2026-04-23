# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-bootstrap configuration resolvers.

Each resolver reads ``run.cfg`` and populates ``run.resolved``.
The chain is sync (no event loop at call site) and order-explicit.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from aiperf.config._dataset_resolver import DatasetResolver

if TYPE_CHECKING:
    from aiperf.config.benchmark import BenchmarkRun

__all__ = [
    "ArtifactDirResolver",
    "CommConfigResolver",
    "ConfigResolver",
    "ConfigResolverChain",
    "DatasetResolver",
    "GpuMetricsResolver",
    "TimingResolver",
    "TokenizerResolver",
    "build_default_resolver_chain",
]

logger = logging.getLogger(__name__)


@runtime_checkable
class ConfigResolver(Protocol):
    """Reads run.cfg, populates run.resolved."""

    def resolve(self, run: BenchmarkRun) -> None: ...


class ConfigResolverChain:
    """Iterate over resolvers in order, calling each one."""

    def __init__(self, resolvers: list[ConfigResolver]) -> None:
        self._resolvers = resolvers

    def resolve_all(self, run: BenchmarkRun) -> None:
        """Run every resolver in sequence."""
        for resolver in self._resolvers:
            resolver.resolve(run)


class ArtifactDirResolver:
    """Resolve artifact_dir to absolute path and create the directory tree.

    When the user hasn't explicitly set a custom artifact directory, appends
    an auto-generated subdirectory name based on the model, endpoint type,
    and stimulus (e.g. ``artifacts/llama-3-8b-openai-chat-concurrency_10/``).
    This matches origin/main's ``UserConfig._compute_artifact_directory()``.
    """

    def resolve(self, run: BenchmarkRun) -> None:
        cfg = run.cfg
        artifact_dir = run.artifact_dir.resolve()

        # Auto-generate descriptive subdirectory if the user didn't set a custom dir.
        # We detect "not custom" by checking if it's the Pydantic default (./artifacts).
        if "dir" not in cfg.artifacts.model_fields_set:
            subdir_name = self._compute_artifact_name(cfg)
            if subdir_name:
                artifact_dir = artifact_dir / subdir_name

        run.artifact_dir = artifact_dir
        run.cfg.artifacts.dir = artifact_dir
        artifact_dir.mkdir(parents=True, exist_ok=True)
        run.resolved.artifact_dir_created = True
        logger.debug("Artifact directory created: %s", artifact_dir)

    @staticmethod
    def _compute_artifact_name(cfg: object) -> str:
        """Build a descriptive directory name from model, service kind, and stimulus.

        Produces names like ``llama-3-8b-openai-chat-concurrency_10``.
        """
        from aiperf.config.config import BenchmarkConfig

        assert isinstance(cfg, BenchmarkConfig)

        parts: list[str] = []

        # 1. Model name
        model_names = cfg.get_model_names()
        if model_names:
            model_name = model_names[0]
            if len(model_names) > 1:
                model_name = f"{model_name}_multi"
            if "/" in model_name:
                model_name = "_".join(model_name.split("/"))
            parts.append(model_name)

        # 2. Service kind + endpoint type
        try:
            from aiperf.plugin import plugins

            metadata = plugins.get_endpoint_metadata(cfg.endpoint.type)
            parts.append(f"{metadata.service_kind}-{cfg.endpoint.type}")
        except Exception:  # noqa: BLE001 - missing/partial plugin registry must not fail artifact-dir naming; falls back to str(endpoint.type)
            parts.append(str(cfg.endpoint.type))

        # 3. Stimulus from the first non-warmup phase
        stimulus = _get_stimulus(cfg)
        if stimulus:
            parts.append(stimulus)

        return "-".join(parts)


def _get_stimulus(cfg: object) -> str:
    """Extract stimulus description from the first non-warmup phase."""
    for phase in cfg.phases.values():  # type: ignore[union-attr]
        if phase.exclude_from_results:
            continue
        return _describe_phase(phase)
    return ""


def _describe_phase(phase: object) -> str:
    """Render a single phase's stimulus description."""
    from aiperf.config.phases import (
        ConcurrencyPhase,
        FixedSchedulePhase,
        UserCentricPhase,
    )

    if isinstance(phase, ConcurrencyPhase):
        return f"concurrency{phase.concurrency}"
    if isinstance(phase, UserCentricPhase):
        return _describe_user_centric(phase)
    if isinstance(phase, FixedSchedulePhase):
        return "fixed_schedule"
    return _describe_rate_phase(phase)


def _describe_user_centric(phase: object) -> str:
    parts = ["user_centric"]
    num_users = phase.num_users  # type: ignore[attr-defined]
    if num_users is not None:
        parts.append(f"users{num_users}")
    request_rate = phase.request_rate  # type: ignore[attr-defined]
    if request_rate is not None:
        parts.append(f"qps{request_rate}")
    return "-".join(parts)


def _describe_rate_phase(phase: object) -> str:
    """Rate phases (poisson, gamma, constant) - render by attribute presence."""
    rate = getattr(phase, "request_rate", None)
    concurrency = getattr(phase, "concurrency", None)
    parts: list[str] = []
    if concurrency is not None:
        parts.append(f"concurrency{concurrency}")
    if rate is not None:
        parts.append(f"request_rate{rate}")
    return "-".join(parts)


class TokenizerResolver:
    """Validate tokenizer early (before spawning services) to fail fast."""

    def resolve(self, run: BenchmarkRun) -> None:
        config = run.cfg
        if not config.tokenizer:
            return

        from aiperf.common.tokenizer_validator import validate_tokenizer_early

        aiperf_logger = _get_aiperf_logger()
        run.resolved.tokenizer_names = validate_tokenizer_early(config, aiperf_logger)


class GpuMetricsResolver:
    """Validate and cache custom GPU metrics CSV if configured."""

    def resolve(self, run: BenchmarkRun) -> None:
        csv_path = run.cfg.gpu_telemetry.metrics_file
        if csv_path is None:
            return

        if not csv_path.exists():
            raise FileNotFoundError(f"Custom GPU metrics file not found: {csv_path}")

        from aiperf.gpu_telemetry.metrics_config import MetricsConfigLoader

        logger.info("Custom GPU metrics file configured: %s", csv_path)
        loader = MetricsConfigLoader()
        custom_metrics, dcgm_mappings = loader.build_custom_metrics_from_csv(csv_path)
        logger.info(
            "Validated %d custom metrics from %s", len(custom_metrics), csv_path
        )
        run.resolved.gpu_custom_metrics = custom_metrics
        run.resolved.gpu_dcgm_mappings = dcgm_mappings


class CommConfigResolver:
    """Resolve the ZMQ communication config from runtime.communication.

    Maps user-facing communication config (IPC/TCP/DUAL) to the internal
    ZMQ config classes that services actually consume. This is the single
    place where communication topology decisions are made.
    """

    def resolve(self, run: BenchmarkRun) -> None:
        from aiperf.common.enums import CommunicationType
        from aiperf.config.zmq import ZMQDualBindConfig, ZMQIPCConfig, ZMQTCPConfig

        comm = run.cfg.runtime.communication
        if comm is None:
            run.resolved.comm_config = ZMQIPCConfig()
            return

        if comm.type == CommunicationType.IPC:
            run.resolved.comm_config = ZMQIPCConfig(
                path=getattr(comm, "path", None),
            )
        elif comm.type == CommunicationType.TCP:
            run.resolved.comm_config = ZMQTCPConfig(
                host=comm.host,
                records_push_pull_port=comm.records_port,
                credit_router_port=comm.credit_router_port,
            )
        elif comm.type == CommunicationType.DUAL:
            controller_host = comm.controller_host
            if controller_host is None:
                from aiperf.kubernetes.environment import K8sEnvironment

                controller_host = K8sEnvironment.ZMQ.CONTROLLER_HOST
            run.resolved.comm_config = ZMQDualBindConfig(
                ipc_path=comm.ipc_path,
                tcp_host=comm.tcp_host,
                controller_host=controller_host,
                records_push_pull_tcp_port=comm.records_port,
                credit_router_tcp_port=comm.credit_router_port,
            )
        else:
            run.resolved.comm_config = ZMQIPCConfig()

        logger.debug(
            "Resolved comm config: %s", type(run.resolved.comm_config).__name__
        )


class TimingResolver:
    """Sum phase durations, validate fixed_schedule timing data requirements."""

    def resolve(self, run: BenchmarkRun) -> None:
        from aiperf.plugin.enums import PhaseType

        total = 0.0
        for phase_name, phase in run.cfg.phases.items():
            if phase.duration is None:
                run.resolved.total_expected_duration = None
                return
            total += phase.duration
            if phase.grace_period is not None:
                total += phase.grace_period

            # Validate fixed_schedule phases have timing data in their dataset
            if str(phase.type) == str(PhaseType.FIXED_SCHEDULE):
                self._validate_fixed_schedule_timing(run, phase_name, phase)

        run.resolved.total_expected_duration = total

    @staticmethod
    def _validate_fixed_schedule_timing(
        run: BenchmarkRun, phase_name: str, phase: object
    ) -> None:
        timing_map = run.resolved.dataset_has_timing_data
        if timing_map is None:
            return
        dataset_name = (
            getattr(phase, "dataset", None) or run.cfg.get_default_dataset_name()
        )
        has_timing = timing_map.get(dataset_name)
        if has_timing is False:
            raise ValueError(
                f"Phase '{phase_name}' uses fixed_schedule which requires "
                f"timestamp or delay fields in the dataset, but dataset "
                f"'{dataset_name}' has no timing data in its first record"
            )


def build_default_resolver_chain() -> ConfigResolverChain:
    """Build the default resolver chain for pre-bootstrap resolution."""
    return ConfigResolverChain(
        [
            ArtifactDirResolver(),
            TokenizerResolver(),
            GpuMetricsResolver(),
            CommConfigResolver(),
            DatasetResolver(),
            TimingResolver(),
        ]
    )


def _get_aiperf_logger() -> object:
    """Lazy import to avoid circular dependency."""
    from aiperf.common.aiperf_logger import AIPerfLogger

    return AIPerfLogger(__name__)
