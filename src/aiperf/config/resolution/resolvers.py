# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-bootstrap configuration resolvers.

Each resolver reads ``run.cfg`` and populates ``run.resolved``.
The chain is sync (no event loop at call site) and order-explicit.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.config.dataset.resolver import DatasetResolver

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun
    from aiperf.config.user_files import RunMeta

# 9-11 digits covers epoch-seconds from 1973 (10^9) through 5138 (10^11),
# which comfortably brackets any realistic AIPerfJob creation timestamp.
# Inlined from aiperf.operator.results_layout to keep the config package
# free of operator/kubernetes imports.
_EPOCH_RE = re.compile(r"^\d{9,11}$")

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

_logger = AIPerfLogger(__name__)


@runtime_checkable
class ConfigResolver(Protocol):
    """Reads run.cfg, populates run.resolved."""

    def resolve(self, run: BenchmarkRun) -> None:
        """Populate ``run.resolved`` from ``run.cfg``.

        Implementations must be synchronous and should mutate only the supplied
        ``BenchmarkRun``. Raise focused exceptions for missing or invalid inputs so
        config validation fails before services start.
        """
        ...


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
    and stimulus (e.g. ``artifacts/llama-3-8b-openai-chat-concurrency10/``).
    """

    def resolve(self, run: BenchmarkRun, *, for_probe: bool = False) -> None:
        """Resolve artifact_dir and (optionally) materialize user_files.

        Args:
            run: The BenchmarkRun whose ``cfg.artifacts.dir`` to normalize.
            for_probe: When True, skip user_files materialization. The probe
                run in ``cli_runner._estimate_and_log_duration`` clones
                ``first_config`` only to estimate duration; per-variation runs
                materialize user_files into their own dirs, so writing them
                here would produce a stray artifact tree and bake in template
                values (e.g. ``{{ epoch }}``) that don't match the real run.
        """
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
        _logger.debug(f"Artifact directory created: {artifact_dir}")

        if run.cfg.artifacts.user_files and not for_probe:
            from aiperf.config.user_files import (
                build_user_file_context,
                materialize_user_files,
            )

            run_meta = _derive_run_meta(artifact_dir)
            context = build_user_file_context(
                run.cfg,
                run_meta,
                run_dir=artifact_dir,
                variables=run.variables,
            )
            materialize_user_files(
                run.cfg.artifacts.user_files,
                run_dir=artifact_dir,
                context=context,
            )

    @staticmethod
    def _compute_artifact_name(cfg: object) -> str:
        """Build a descriptive directory name from model, service kind, and stimulus.

        Produces names like ``llama-3-8b-openai-chat-concurrency10``.
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
        except Exception:  # missing/partial plugin registry must not fail artifact-dir naming; falls back to str(endpoint.type)
            parts.append(str(cfg.endpoint.type))

        # 3. Stimulus from the first non-warmup phase
        stimulus = _get_stimulus(cfg)
        if stimulus:
            parts.append(stimulus)

        return "-".join(parts)


def _get_stimulus(cfg: object) -> str:
    """Extract stimulus description from the first non-warmup phase."""
    for phase in cfg.phases:  # type: ignore[union-attr]
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
    num_users = phase.users  # type: ignore[attr-defined]
    if num_users is not None:
        parts.append(f"users{num_users}")
    request_rate = phase.rate  # type: ignore[attr-defined]
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


def _derive_run_meta(artifact_dir: Path) -> RunMeta:
    """Derive RunMeta (epoch, job_name, namespace) from the resolved artifact_dir.

    Operator-managed runs use the ``<base>/<ns>/<name>/<epoch>`` layout (see
    ``aiperf.operator.results_layout.run_dir``). When the leaf matches
    ``_EPOCH_RE`` we treat the parent as the AIPerfJob name and the leaf as
    the run epoch. Otherwise (local-CLI runs, custom paths) the leaf IS the
    run identifier and we substitute wall-clock seconds for the epoch.

    Using ``_EPOCH_RE`` (not ``str.isdigit``) shrinks the false-positive
    surface — e.g. ``/tmp/bench/42`` is correctly treated as a local layout
    rather than a one-day-old operator run.

    Namespace is sourced from ``AIPERF_NAMESPACE`` (injected by the operator
    via the downward API). Empty string for local runs — the ``{{ namespace }}``
    template var resolves to ``""`` outside Kubernetes.
    """
    # Lazy import to avoid cycles via aiperf.config.resolution.plan.
    from aiperf.config.user_files import RunMeta

    leaf = artifact_dir.name
    namespace = os.environ.get("AIPERF_NAMESPACE", "")
    # ``legacy`` is the sentinel run-dir name used by ``aiperf kube results``
    # / ``results_operator.py`` for runs that predate the operator's
    # epoch-stamped layout. Treat it the same as a numeric epoch so the run
    # metadata reflects the historical sentinel, not wall-clock time.
    if _EPOCH_RE.match(leaf) or leaf == "legacy":
        return RunMeta(
            epoch=leaf,
            job_name=artifact_dir.parent.name,
            namespace=namespace,
        )
    return RunMeta(
        epoch=str(int(time.time())),
        job_name=leaf,
        namespace=namespace,
    )


class TokenizerResolver:
    """Validate tokenizer early (before spawning services) to fail fast."""

    def resolve(self, run: BenchmarkRun) -> None:
        from aiperf.common.tokenizer_validator import validate_tokenizer_early

        run.resolved.tokenizer_names = validate_tokenizer_early(
            run.cfg, _get_aiperf_logger()
        )


class GpuMetricsResolver:
    """Validate and cache custom GPU metrics CSV if configured."""

    def resolve(self, run: BenchmarkRun) -> None:
        csv_path = run.cfg.gpu_telemetry.metrics_file
        if csv_path is None:
            return

        if not csv_path.exists():
            raise FileNotFoundError(f"Custom GPU metrics file not found: {csv_path}")

        from aiperf.gpu_telemetry.metrics_config import MetricsConfigLoader

        _logger.info(f"Custom GPU metrics file configured: {csv_path}")
        loader = MetricsConfigLoader()
        custom_metrics, dcgm_mappings = loader.build_custom_metrics_from_csv(csv_path)
        _logger.info(f"Validated {len(custom_metrics)} custom metrics from {csv_path}")
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
        from aiperf.config.comm import ZMQDualBindConfig, ZMQIPCConfig, ZMQTCPConfig

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
                controller_host = os.environ.get("AIPERF_K8S_ZMQ_CONTROLLER_HOST")
            run.resolved.comm_config = ZMQDualBindConfig(
                ipc_path=comm.ipc_path,
                tcp_host=comm.tcp_host,
                controller_host=controller_host,
                records_push_pull_tcp_port=comm.records_port,
                credit_router_tcp_port=comm.credit_router_port,
            )
        else:
            run.resolved.comm_config = ZMQIPCConfig()

        _logger.debug(
            f"Resolved comm config: {type(run.resolved.comm_config).__name__}"
        )


class TimingResolver:
    """Sum phase durations (plus grace_period), validate fixed_schedule timing data requirements."""

    def resolve(self, run: BenchmarkRun) -> None:
        from aiperf.plugin.enums import PhaseType

        total = 0.0
        duration_unknown = False
        for phase in run.cfg.phases:
            if str(phase.type) == str(PhaseType.FIXED_SCHEDULE):
                self._validate_fixed_schedule_timing(run, phase.name, phase)

            if phase.duration is None:
                duration_unknown = True
                continue
            total += phase.duration
            if phase.grace_period is not None:
                total += phase.grace_period

        run.resolved.total_expected_duration = None if duration_unknown else total

    @staticmethod
    def _validate_fixed_schedule_timing(
        run: BenchmarkRun, phase_name: str, phase: object
    ) -> None:
        timing_map = run.resolved.dataset_has_timing_data
        dataset_name = (
            getattr(phase, "dataset", None) or run.cfg.get_default_dataset_name()
        )
        if timing_map is None:
            raise ValueError(
                f"Phase '{phase_name}' uses fixed_schedule, but could not verify "
                f"timing data for dataset '{dataset_name}'"
            )
        has_timing = timing_map.get(dataset_name)
        if has_timing is not True:
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


def _get_aiperf_logger() -> AIPerfLogger:
    return _logger
