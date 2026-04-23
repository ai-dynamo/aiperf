# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-bootstrap configuration resolvers.

Each resolver reads ``run.cfg`` and populates ``run.resolved``.
The chain is sync (no event loop at call site) and order-explicit.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

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
    from aiperf.config.phases import (
        ConcurrencyPhase,
        FixedSchedulePhase,
        UserCentricPhase,
    )

    for phase in cfg.phases.values():  # type: ignore[union-attr]
        if phase.exclude_from_results:
            continue

        if isinstance(phase, ConcurrencyPhase):
            return f"concurrency{phase.concurrency}"
        if isinstance(phase, UserCentricPhase):
            parts = ["user_centric"]
            if phase.num_users is not None:
                parts.append(f"users{phase.num_users}")
            if phase.request_rate is not None:
                parts.append(f"qps{phase.request_rate}")
            return "-".join(parts)
        if isinstance(phase, FixedSchedulePhase):
            return "fixed_schedule"

        # Rate phases (poisson, gamma, constant)
        rate = getattr(phase, "request_rate", None)
        concurrency = getattr(phase, "concurrency", None)
        parts = []
        if concurrency is not None:
            parts.append(f"concurrency{concurrency}")
        if rate is not None:
            parts.append(f"request_rate{rate}")
        if parts:
            return "-".join(parts)

    return ""


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


class DatasetResolver:
    """Resolve file-based dataset paths, detect types, timing, and sampling."""

    def resolve(self, run: BenchmarkRun) -> None:
        from aiperf.config.dataset import FileDataset
        from aiperf.plugin.enums import CustomDatasetType, DatasetSamplingStrategy

        paths: dict[str, object] = {}
        types: dict[str, CustomDatasetType] = {}
        sampling: dict[str, DatasetSamplingStrategy] = {}
        has_timing: dict[str, bool] = {}
        total_records: dict[str, int] = {}
        session_counts: dict[str, int] = {}
        format_map = self._build_format_map()

        for name, ds in run.cfg.datasets.items():
            if not isinstance(ds, FileDataset):
                continue

            # 1. Resolve and validate path
            resolved = ds.path.resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"Dataset '{name}' file not found: {resolved}")
            paths[name] = resolved

            # 2. Detect dataset type from explicit format or via can_load
            first_record = None
            dataset_type = format_map.get(str(ds.format)) if ds.format else None
            if dataset_type is None:
                dataset_type, first_record = self._detect_type(str(resolved))

            if dataset_type is not None:
                types[name] = dataset_type
                # 3. Resolve sampling strategy
                # Only use loader's recommended strategy if user hasn't explicitly set one
                loader_sampling = self._get_preferred_sampling(dataset_type)
                if (
                    ds.sampling == DatasetSamplingStrategy.SEQUENTIAL
                    and loader_sampling != DatasetSamplingStrategy.SEQUENTIAL
                ):
                    sampling[name] = loader_sampling
                else:
                    sampling[name] = ds.sampling
                # 4. Detect timing data from actual first record
                has_timing[name] = self._check_timing_data(str(resolved), first_record)

            # 5. Count records and sessions (for validation and fixed_schedule)
            if not resolved.is_dir():
                records, sessions = self._count_records_and_sessions(
                    str(resolved), dataset_type
                )
                total_records[name] = records
                session_counts[name] = sessions

        if paths:
            run.resolved.dataset_file_paths = paths  # type: ignore[assignment]
        if types:
            run.resolved.dataset_types = types
            run.resolved.dataset_sampling_strategies = sampling
            run.resolved.dataset_has_timing_data = has_timing
        if total_records:
            run.resolved.dataset_total_records = total_records
            run.resolved.dataset_session_count = session_counts
        if paths or types:
            logger.debug("Resolved %d dataset paths, %d types", len(paths), len(types))

    @staticmethod
    def _build_format_map() -> dict[str, object]:
        from aiperf.common.enums import DatasetFormat
        from aiperf.plugin.enums import CustomDatasetType

        return {
            str(DatasetFormat.SINGLE_TURN): CustomDatasetType.SINGLE_TURN,
            str(DatasetFormat.MULTI_TURN): CustomDatasetType.MULTI_TURN,
            str(DatasetFormat.MOONCAKE_TRACE): CustomDatasetType.MOONCAKE_TRACE,
            str(DatasetFormat.RANDOM_POOL): CustomDatasetType.RANDOM_POOL,
        }

    @staticmethod
    def _detect_type(
        file_path: str,
    ) -> tuple[object | None, dict | None]:
        """Auto-detect dataset type by querying registered loaders.

        Returns (detected_type, first_record) so the caller can reuse
        the already-parsed first line for timing data detection.
        """
        from pathlib import Path

        from aiperf.common.utils import load_json_str
        from aiperf.plugin import plugins
        from aiperf.plugin.enums import CustomDatasetType, PluginType

        path = Path(file_path)
        if path.is_dir():
            data = None
        else:
            try:
                with open(file_path) as f:
                    for line in f:
                        if line := line.strip():
                            data = load_json_str(line)
                            break
                    else:
                        return None, None
            except (OSError, ValueError):
                return None, None

        # Check explicit type field in data
        if data is not None and data.get("type") in CustomDatasetType:
            explicit_type = CustomDatasetType(data["type"])
            LoaderClass = plugins.get_class(
                PluginType.CUSTOM_DATASET_LOADER, explicit_type
            )
            if LoaderClass.can_load(data, file_path):
                return explicit_type, data

        # Structural detection
        detected = None
        for entry, LoaderClass in plugins.iter_all(PluginType.CUSTOM_DATASET_LOADER):
            if LoaderClass.can_load(data, file_path):
                if detected is not None:
                    logger.warning(
                        "Multiple loaders match dataset '%s', skipping auto-detection",
                        file_path,
                    )
                    return None, data
                detected = CustomDatasetType(entry.name)
        return detected, data

    @staticmethod
    def _check_timing_data(file_path: str, first_record: dict | None) -> bool:
        """Check whether the first record has timestamp or delay fields.

        Inspects the actual data rather than assuming from dataset type,
        because trace formats like mooncake may omit timing fields.
        """
        record = first_record
        if record is None:
            from pathlib import Path

            from aiperf.common.utils import load_json_str

            if Path(file_path).is_dir():
                return False
            try:
                with open(file_path) as f:
                    for line in f:
                        if line := line.strip():
                            record = load_json_str(line)
                            break
            except (OSError, ValueError):
                return False

        if record is None:
            return False
        return record.get("timestamp") is not None or record.get("delay") is not None

    @staticmethod
    def _count_records_and_sessions(
        file_path: str, dataset_type: object | None
    ) -> tuple[int, int]:
        """Count total non-empty records and unique sessions in a JSONL file.

        For multi-turn datasets, sessions are identified by session_id or
        chat_id fields. For single-turn, each record is its own session.
        """
        from aiperf.plugin.enums import CustomDatasetType

        is_multi_turn = dataset_type in (
            CustomDatasetType.MULTI_TURN,
            CustomDatasetType.BAILIAN_TRACE,
        )
        record_count = 0
        session_ids: set[str] = set()

        try:
            with open(file_path) as f:
                for line in f:
                    if not (line := line.strip()):
                        continue
                    record_count += 1
                    if is_multi_turn:
                        _add_session_id(line, session_ids)
        except OSError:
            return 0, 0

        if is_multi_turn and session_ids:
            return record_count, len(session_ids)
        return record_count, record_count

    @staticmethod
    def _get_preferred_sampling(dataset_type: object) -> object:
        """Get the loader's preferred sampling strategy."""
        from aiperf.plugin import plugins
        from aiperf.plugin.enums import DatasetSamplingStrategy, PluginType

        try:
            LoaderClass = plugins.get_class(
                PluginType.CUSTOM_DATASET_LOADER, dataset_type
            )
            if hasattr(LoaderClass, "get_preferred_sampling_strategy"):
                return LoaderClass.get_preferred_sampling_strategy()
        except (KeyError, ValueError):
            pass
        return DatasetSamplingStrategy.SEQUENTIAL


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


def _add_session_id(line: str, session_ids: set[str]) -> None:
    """Parse a JSONL line and add its session_id/chat_id to the set."""
    from aiperf.common.utils import load_json_str

    try:
        data = load_json_str(line)
    except (ValueError, TypeError):
        return
    sid = data.get("session_id") or data.get("chat_id")
    if sid is not None:
        session_ids.add(str(sid))
