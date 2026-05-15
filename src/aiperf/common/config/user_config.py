# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

if TYPE_CHECKING:
    from aiperf.plugin.schema.schemas import EndpointMetadata

from orjson import JSONDecodeError
from pydantic import BeforeValidator, Field, model_validator
from typing_extensions import Self

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.config.accuracy_config import AccuracyConfig
from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.cli_parameter import CLIParameter, DisableCLI
from aiperf.common.config.config_defaults import (
    LoadGeneratorDefaults,
    MLflowDefaults,
    ServerMetricsDefaults,
)
from aiperf.common.config.config_validators import (
    coerce_value,
    parse_str_or_dict_as_tuple_list,
    parse_str_or_list,
)
from aiperf.common.config.endpoint_config import EndpointConfig
from aiperf.common.config.groups import Groups
from aiperf.common.config.input_config import InputConfig
from aiperf.common.config.loadgen_config import LoadGeneratorConfig
from aiperf.common.config.output_config import OutputConfig
from aiperf.common.config.tokenizer_config import TokenizerConfig
from aiperf.common.enums import GPUTelemetryMode, ServerMetricsFormat
from aiperf.common.utils import load_json_str
from aiperf.plugin import plugins
from aiperf.plugin.enums import (
    ArrivalPattern,
    EndpointType,
    GPUTelemetryCollectorType,
    TimingMode,
)

_logger = AIPerfLogger(__name__)


def _is_localhost_url(url: str) -> bool:
    """Check if a URL points to localhost."""
    from urllib.parse import urlparse

    # Handle IPv6 localhost without brackets (e.g. "::1:8000" or "http://::1:8000").
    # `EndpointConfig` now prepends `http://` to scheme-less URLs, so we accept
    # both the pre-normalization and post-normalization forms here.
    url_without_scheme = url.removeprefix("http://").removeprefix("https://")
    if url_without_scheme.startswith("::1:") or url_without_scheme.startswith("[::1]"):
        return True

    # Add scheme if missing for proper parsing
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"

    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    return hostname.lower() in ("localhost", "127.0.0.1", "::1")


def _normalize_otel_metrics_url(url: str) -> str:
    """Normalize OTel collector URL to an OTLP metrics endpoint."""
    from urllib.parse import urlparse, urlunparse

    normalized_url = url.strip()
    if not normalized_url:
        raise ValueError("--otel-url cannot be empty.")

    # Only auto-prefix host[:port] style values. Explicit schemes must be validated as-is.
    if "://" not in normalized_url:
        normalized_url = f"http://{normalized_url}"

    parsed = urlparse(normalized_url)
    # `urlparse("http://:4318")` yields netloc=":4318" but hostname=None —
    # netloc truthiness alone is not enough. Require a non-empty hostname so
    # bare-port values don't slip through and produce a malformed OTLP endpoint.
    if not parsed.scheme or not parsed.netloc or not parsed.hostname:
        raise ValueError(
            f"Invalid --otel-url value: {url!r}. Expected host[:port] or a full URL."
        )
    if parsed.scheme.lower() not in ("http", "https"):
        raise ValueError(
            f"Invalid --otel-url value: {url!r}. "
            f"Only http and https schemes are supported (got {parsed.scheme!r}). "
            "OTLP/gRPC is not supported; use the OTLP/HTTP exporter endpoint."
        )

    path = parsed.path.rstrip("/")
    if path.endswith("/v1/metrics"):
        normalized_path = path
    elif not path:
        normalized_path = "/v1/metrics"
    else:
        normalized_path = f"{path}/v1/metrics"

    return urlunparse(parsed._replace(path=normalized_path))


def _should_quote_arg(x: Any) -> bool:
    """Determine if the value should be quoted in the CLI command."""
    return isinstance(x, str) and not x.startswith("-") and x not in ("profile")


def _collect_pre_session_refs(data: dict, into: set[str]) -> None:
    """Add ``pre_session_spawns`` child ids (bare strings only) into ``into``."""
    for child in data.get("pre_session_spawns", []) or []:
        if isinstance(child, str):
            into.add(child)


def _collect_turn_refs(turn: dict, into: set[str]) -> None:
    """Add child ids referenced from one turn's ``forks`` and ``spawns`` into ``into``.

    ``forks`` entries are a bare ``"<sid>"`` or a ``{"child": "<sid>", ...}``
    object. ``spawns`` entries are a bare ``"<sid>"`` or a
    ``{"children": [...], ...}`` object (DagSpawn form).
    """
    for fork_entry in turn.get("forks", []) or []:
        if isinstance(fork_entry, str):
            into.add(fork_entry)
        elif isinstance(fork_entry, dict):
            child = fork_entry.get("child")
            if isinstance(child, str):
                into.add(child)
    for spawn_entry in turn.get("spawns", []) or []:
        if isinstance(spawn_entry, str):
            into.add(spawn_entry)
        elif isinstance(spawn_entry, dict):
            for child in spawn_entry.get("children", []) or []:
                if isinstance(child, str):
                    into.add(child)


# CLI keyword -> collector type for local-only GPU telemetry collectors.
_LOCAL_COLLECTOR_KEYWORDS: dict[str, GPUTelemetryCollectorType] = {
    "pynvml": GPUTelemetryCollectorType.PYNVML,
    "amdsmi": GPUTelemetryCollectorType.AMDSMI,
}

# Collector type -> human-readable name for warning/error messages.
_LOCAL_ONLY_COLLECTORS: dict[GPUTelemetryCollectorType, str] = {
    GPUTelemetryCollectorType.PYNVML: "pynvml",
    GPUTelemetryCollectorType.AMDSMI: "amdsmi",
}

# Install hint surfaced when a local collector's Python bindings are missing.
_LOCAL_COLLECTOR_INSTALL_HINTS: dict[GPUTelemetryCollectorType, str] = {
    GPUTelemetryCollectorType.PYNVML: (
        "pynvml package not installed. Install with: pip install nvidia-ml-py"
    ),
    GPUTelemetryCollectorType.AMDSMI: (
        "amdsmi package not installed. The amdsmi Python bindings ship with "
        "ROCm; install from /opt/rocm/share/amd_smi/amdsmi-*.whl or your "
        "distro's amd-smi-lib package."
    ),
}


def _ensure_local_collector_importable(
    collector_type: GPUTelemetryCollectorType,
) -> None:
    """Verify that the Python bindings for a local collector are importable.

    Catches broader than just ``ImportError``: amdsmi (and to a lesser extent
    pynvml) can also raise ``OSError`` when the wheel is installed but the
    underlying native library (libamd_smi, libnvidia-ml) is missing or fails
    to load. Surface those failures with the same friendly install hint
    instead of leaking an internal traceback.
    """
    module_name = _LOCAL_ONLY_COLLECTORS[collector_type]
    try:
        import_module(module_name)
    except (ImportError, OSError) as e:
        raise ValueError(_LOCAL_COLLECTOR_INSTALL_HINTS[collector_type]) from e


class UserConfig(BaseConfig):
    """
    A configuration class for defining top-level user settings.
    """

    _timing_mode: TimingMode = TimingMode.REQUEST_RATE

    def _endpoint_metadata(self) -> "EndpointMetadata":
        """Get the endpoint metadata for the current endpoint type."""
        try:
            return self._cached_endpoint_metadata
        except AttributeError:
            from aiperf.plugin import plugins

            meta = plugins.get_endpoint_metadata(self.endpoint.type)
            self._cached_endpoint_metadata = meta
            return meta

    @model_validator(mode="after")
    def validate_cli_args(self) -> Self:
        """Set the CLI command based on the command line arguments, if it has not already been set."""
        if not self.cli_command:
            from aiperf.common.redact import redact_cli_command

            args = [coerce_value(x) for x in sys.argv[1:]]
            # Note: Use single quotes to avoid conflicts with double quotes in arguments.
            args = [f"'{x}'" if _should_quote_arg(x) else str(x) for x in args]
            cmd = " ".join(["aiperf", *args])
            # redact_cli_command handles --api-key, sensitive headers, and URL-typed
            # flags (--url, --otel-url, --mlflow-tracking-uri) in one pass.
            self.cli_command = redact_cli_command(cmd)
        return self

    @model_validator(mode="after")
    def generate_benchmark_id(self) -> Self:
        """Generate a unique benchmark ID if not already set.

        This ID is shared across all export formats (JSON, CSV, Parquet, etc.)
        to enable correlation of data from the same benchmark run.
        """
        if not self.benchmark_id:
            import uuid

            self.benchmark_id = str(uuid.uuid4())
        return self

    # TODO: Dataset validator class for these

    @model_validator(mode="after")
    def validate_timing_mode(self) -> Self:
        """Set the timing mode based on the user config. Will be called after all user config is set."""
        if self.input.fixed_schedule:
            self._timing_mode = TimingMode.FIXED_SCHEDULE
            if (
                self.loadgen.request_count is None
                and self.input.conversation.num is None
            ):
                self.loadgen.request_count = self._count_dataset_entries()
                _logger.info(
                    f"No request count value provided for fixed schedule mode, setting to dataset entry count: {self.loadgen.request_count}"
                )
        elif self._should_use_fixed_schedule_for_trace_dataset():
            self._timing_mode = TimingMode.FIXED_SCHEDULE
            _logger.info(
                f"Automatically enabling fixed schedule mode for {self.input.custom_dataset_type} dataset with timestamps"
            )
            if (
                self.loadgen.request_count is None
                and self.input.conversation.num is None
            ):
                self.loadgen.request_count = self._count_dataset_entries()
                _logger.info(
                    f"No request count value provided for trace dataset, setting to dataset entry count: {self.loadgen.request_count}"
                )
        elif self.loadgen.user_centric_rate is not None:
            # User-centric rate mode: per-user rate limiting (LMBenchmark parity)
            # --user-centric-rate takes the QPS value directly
            self._timing_mode = TimingMode.USER_CENTRIC_RATE
            if self.loadgen.num_users is None:
                raise ValueError("--user-centric-rate requires --num-users to be set")
            # TODO: Design a better way to create mutually exclusive options.
            if (
                "request_rate" in self.loadgen.model_fields_set
                or "arrival_pattern" in self.loadgen.model_fields_set
            ):
                raise ValueError(
                    "--user-centric-rate cannot be used together with --request-rate or --arrival-pattern"
                )

            if (
                self.loadgen.benchmark_duration is not None
                and "benchmark_grace_period" not in self.loadgen.model_fields_set
            ):
                # By default, lmbench waits indefinitely for all responses.
                self.loadgen.benchmark_grace_period = float("inf")

            # User-centric mode only makes sense for multi-turn conversations.
            # With single-turn, it degenerates to request-rate mode with extra overhead.
            if self.input.conversation.turn.mean < 2:
                raise ValueError(
                    "--user-centric-rate requires multi-turn conversations (--session-turns-mean >= 2). "
                    "For single-turn workloads, use --request-rate instead."
                )
        elif self.loadgen.request_rate is not None:
            # Request rate is checked first, as if user has provided request rate and concurrency,
            # we will still use the request rate strategy.
            self._timing_mode = TimingMode.REQUEST_RATE
            if self.loadgen.arrival_pattern == ArrivalPattern.CONCURRENCY_BURST:
                raise ValueError(
                    f"Request rate mode cannot be {ArrivalPattern.CONCURRENCY_BURST!r} when a request rate is specified."
                )
            if (
                self.loadgen.request_count is None
                and self.input.conversation.num is None
                and self.loadgen.benchmark_duration is None
            ):
                self._autodefault_request_count_with_forking_check(
                    LoadGeneratorDefaults.MIN_REQUEST_COUNT
                )
        else:
            # Default to concurrency burst mode if no request rate or schedule is provided.
            # CONCURRENCY_BURST works with either session concurrency OR prefill concurrency.
            if (
                self.loadgen.concurrency is None
                and self.loadgen.prefill_concurrency is None
            ):
                # Only set default session concurrency if neither concurrency type is specified
                _logger.warning("No concurrency value provided, setting to 1")
                self.loadgen.concurrency = 1

            if (
                self.loadgen.request_count is None
                and self.input.conversation.num is None
                and self.loadgen.benchmark_duration is None
            ):
                # Use whichever concurrency is set for calculating default request count
                effective_concurrency = (
                    self.loadgen.concurrency or self.loadgen.prefill_concurrency
                )
                fallback = max(
                    LoadGeneratorDefaults.MIN_REQUEST_COUNT,
                    effective_concurrency
                    * LoadGeneratorDefaults.REQUEST_COUNT_MULTIPLIER,
                )
                self._autodefault_request_count_with_forking_check(fallback)
            self._timing_mode = TimingMode.REQUEST_RATE
            self.loadgen.arrival_pattern = ArrivalPattern.CONCURRENCY_BURST

        if (
            "arrival_pattern" not in self.loadgen.model_fields_set
            and self.loadgen.arrival_smoothness is not None
        ):
            self.loadgen.arrival_pattern = ArrivalPattern.GAMMA
            _logger.info(
                "Arrival smoothness specified, but arrival pattern is not. Setting arrival pattern to gamma by default."
            )
        elif (
            self.loadgen.arrival_pattern != ArrivalPattern.GAMMA
            and self.loadgen.arrival_smoothness is not None
        ):
            raise ValueError(
                "--arrival-smoothness can only be used with --arrival-pattern gamma. "
                "Please specify --arrival-pattern gamma to use --arrival-smoothness."
            )

        return self

    @model_validator(mode="after")
    def validate_num_users_requirements(self) -> Self:
        """Validate that num_users requirements are met when set.

        When --num-users is set along with --num-sessions or --request-count,
        both --num-sessions and --request-count (if specified) must be >= --num-users
        to ensure there are enough sessions and requests for all users.
        """
        if self.loadgen.num_users is None:
            return self

        # Check if either num_sessions or request_count is set
        has_num_sessions = self.input.conversation.num is not None
        has_request_count = self.loadgen.request_count is not None

        if not (has_num_sessions or has_request_count):
            return self

        num_users = self.loadgen.num_users

        # Validate num_sessions if set
        if has_num_sessions and self.input.conversation.num < num_users:
            raise ValueError(
                f"--num-sessions ({self.input.conversation.num}) cannot be less than "
                f"--num-users ({num_users}). Each user needs at least one session."
            )

        # Validate request_count if set
        if has_request_count and self.loadgen.request_count < num_users:
            raise ValueError(
                f"--request-count ({self.loadgen.request_count}) cannot be less than "
                f"--num-users ({num_users}). There must be at least one request per user."
            )

        return self

    @model_validator(mode="after")
    def validate_benchmark_mode(self) -> Self:
        """Validate benchmarking associated args are correctly set."""
        if (
            "benchmark_grace_period" in self.loadgen.model_fields_set
            and self.loadgen.benchmark_duration is None
        ):
            raise ValueError(
                "--benchmark-grace-period can only be used with "
                "duration-based benchmarking (--benchmark-duration)."
            )

        return self

    @model_validator(mode="after")
    def validate_warmup_grace_period(self) -> Self:
        """Validate warmup grace period is only used when --warmup-duration is set."""
        if (
            "warmup_grace_period" in self.loadgen.model_fields_set
            and self.loadgen.warmup_duration is None
        ):
            raise ValueError(
                "--warmup-grace-period can only be used when --warmup-duration is set. "
                "Set --warmup-duration."
            )

        return self

    @model_validator(mode="after")
    def validate_unused_options(self) -> Self:
        """Validate that options are not set without their required companion options.

        These options are only meaningful with specific configurations.
        Rather than silently ignoring them, we raise an error.
        """
        # --num-users without --user-centric-rate
        if (
            "num_users" in self.loadgen.model_fields_set
            and self.loadgen.user_centric_rate is None
        ):
            raise ValueError(
                "--num-users can only be used with --user-centric-rate. "
                "Either add --user-centric-rate or remove --num-users."
            )

        # --request-cancellation-delay without --request-cancellation-rate
        if (
            "request_cancellation_delay" in self.loadgen.model_fields_set
            and self.loadgen.request_cancellation_rate is None
        ):
            raise ValueError(
                "--request-cancellation-delay can only be used with --request-cancellation-rate. "
                "Either add --request-cancellation-rate or remove --request-cancellation-delay."
            )

        # --fixed-schedule-* options without --fixed-schedule
        fixed_schedule_enabled = self.input.fixed_schedule
        fixed_schedule_options_set = []

        if "fixed_schedule_auto_offset" in self.input.model_fields_set:
            fixed_schedule_options_set.append("--fixed-schedule-auto-offset")
        if "fixed_schedule_start_offset" in self.input.model_fields_set:
            fixed_schedule_options_set.append("--fixed-schedule-start-offset")
        if "fixed_schedule_end_offset" in self.input.model_fields_set:
            fixed_schedule_options_set.append("--fixed-schedule-end-offset")

        if fixed_schedule_options_set and not fixed_schedule_enabled:
            options_str = ", ".join(fixed_schedule_options_set)
            raise ValueError(
                f"{options_str} can only be used with --fixed-schedule. "
                "Either add --fixed-schedule or remove these options."
            )

        # --request-rate-ramp-duration without --request-rate
        # Rate ramping only works with rate-based scheduling (not user-centric or fixed-schedule)
        if (
            "request_rate_ramp_duration" in self.loadgen.model_fields_set
            and self.timing_mode != TimingMode.REQUEST_RATE
        ):
            raise ValueError(
                "--request-rate-ramp-duration can only be used with --request-rate scheduling."
            )

        return self

    @model_validator(mode="after")
    def validate_sweep_incompatibilities(self) -> Self:
        """Validate that parameter sweeps are not combined with incompatible modes.

        Raises:
            ValueError: If parameter sweep is combined with fixed schedule mode.
        """
        # Use parameter-agnostic sweep detection
        sweep_param = self.loadgen.get_sweep_parameter()
        is_sweep = sweep_param is not None

        if is_sweep:
            # Check for fixed schedule mode incompatibility
            # Fixed schedule mode is incompatible because it replays exact timing patterns
            # from a trace file, which doesn't make sense when varying concurrency
            if self.input.fixed_schedule:
                param_name, param_values = sweep_param
                raise ValueError(
                    f"Parameter sweeps (e.g., --{param_name} {','.join(map(str, param_values))}) cannot be used with --fixed-schedule mode. "
                    "Fixed schedule replays exact timing patterns from trace files, which is incompatible with "
                    "varying parameter values. Use a single parameter value or remove --fixed-schedule."
                )

            # Also check if trace dataset will auto-enable fixed schedule
            if self._should_use_fixed_schedule_for_trace_dataset():
                param_name, param_values = sweep_param
                raise ValueError(
                    f"Parameter sweeps (e.g., --{param_name} {','.join(map(str, param_values))}) cannot be used with mooncake_trace datasets "
                    "that have timestamps (which auto-enable fixed schedule mode). "
                    "Fixed schedule replays exact timing patterns from trace files, which is incompatible with "
                    "varying parameter values. Use a single parameter value or use a dataset without timestamps."
                )

        return self

    def _should_use_fixed_schedule_for_trace_dataset(self) -> bool:
        """Check if a trace dataset has timestamps and should use fixed schedule.

        Returns:
            True if fixed schedule should be enabled for this trace dataset.
        """
        if self.input.disable_auto_fixed_schedule:
            return False

        if self.input.custom_dataset_type is None or not plugins.is_trace_dataset(
            self.input.custom_dataset_type
        ):
            return False

        if not self.input.file:
            return False

        try:
            with open(self.input.file) as f:
                for line in f:
                    if not (line := line.strip()):
                        continue
                    try:
                        data = load_json_str(line)
                        return "timestamp" in data and data["timestamp"] is not None
                    except (JSONDecodeError, KeyError):
                        continue
        except (OSError, FileNotFoundError):
            _logger.warning(
                f"Could not read dataset file {self.input.file} to check for timestamps"
            )

        return False

    def _is_forking_dataset(self) -> bool:
        """True if the configured custom dataset can fork (DAG branches).

        Today only ``dag_jsonl`` carries fork semantics; other custom
        datasets (single_turn, raw_payload, inputs_json, mooncake_trace)
        are linear. Used by the auto-default logic to size
        ``--num-conversations`` rather than ``--request-count`` for
        forking datasets, since the former lets the DAG run to
        completion and the latter would truncate mid-tree.
        """
        return self.input.custom_dataset_type == "dag_jsonl"

    def _maybe_autodefault_num_conversations_for_forking_dataset(self) -> None:
        """If forking and both num/request-count unset, seed num-conversations.

        Used by ``validate_timing_mode`` to default ``--num-conversations``
        to the *root* count (sessions not referenced by any fork list)
        rather than auto-defaulting ``--request-count``, which (since it
        counts fork-spawned children) would silently truncate the DAG
        mid-tree.
        """
        if not self._is_forking_dataset():
            return
        roots = self._count_dag_root_entries()
        if roots > 0:
            self.input.conversation.num = roots
            _logger.info(
                f"No request count or conversation count provided for forking dataset; "
                f"defaulting --num-conversations to {roots} (run each root in the file once). "
                f"Use --request-count for a literal wire-request cap instead."
            )

    def _autodefault_request_count_with_forking_check(
        self, fallback_request_count: int
    ) -> None:
        """Apply the load-cap autodefault when both request-count and num
        conversations are unset. Forking datasets seed num-conversations
        to root count instead so the DAG isn't truncated mid-tree.
        """
        self._maybe_autodefault_num_conversations_for_forking_dataset()
        if self.loadgen.request_count is None and self.input.conversation.num is None:
            _logger.warning(
                f"No request count value provided, setting to {fallback_request_count}"
            )
            self.loadgen.request_count = fallback_request_count

    @staticmethod
    def _collect_dag_session_and_fork_ids(file_path: str) -> tuple[set[str], set[str]]:
        """Walk a dag_jsonl file once, returning (all_session_ids, referenced_ids).

        ``referenced_ids`` covers every id that the orchestrator dispatches as
        a child of another conversation: bare-string and object-form
        ``forks`` entries, bare-string and ``DagSpawn``-object ``spawns``
        entries, and top-level ``pre_session_spawns``. Anything in
        ``referenced_ids`` is NOT a root and must not be sampled standalone.
        """
        all_ids: set[str] = set()
        referenced_ids: set[str] = set()
        with open(file_path) as f:
            for raw in f:
                if not (line := raw.strip()):
                    continue
                try:
                    data = load_json_str(line)
                except JSONDecodeError:
                    continue
                sid = data.get("session_id")
                if isinstance(sid, str):
                    all_ids.add(sid)
                _collect_pre_session_refs(data, referenced_ids)
                for turn in data.get("turns", []) or []:
                    if isinstance(turn, dict):
                        _collect_turn_refs(turn, referenced_ids)
        return all_ids, referenced_ids

    def _count_dag_root_entries(self) -> int:
        """Count root conversations (no incoming forks) in a dag_jsonl file.

        Roots are conversations not referenced by any other conversation's
        ``forks`` list. The loader only ever samples roots — non-root
        entries (children) are seeded into the orchestrator from their
        parent's worker context, so sizing ``--num-conversations`` by
        total entry count would over-run a file with deep fanout (e.g.
        a file with 1 root + 2 children = 3 entries should default to
        1 conversation, not 3).
        """
        if not self.input.file:
            return 0
        try:
            all_ids, referenced_ids = self._collect_dag_session_and_fork_ids(
                self.input.file
            )
        except (OSError, FileNotFoundError) as e:
            _logger.error(
                f"Cannot read dag_jsonl file {self.input.file} for root counting: {e}"
            )
            return 0
        return len(all_ids - referenced_ids)

    def _count_dataset_entries(self) -> int:
        """Count the number of valid entries in a custom dataset file or directory.

        For directories, recursively counts non-empty lines across all .jsonl files.

        Returns:
            int: Number of non-empty lines
        """
        if not self.input.file:
            return 0

        path = self.input.file
        try:
            if path.is_dir():
                count = 0
                for jsonl_file in path.rglob("*.jsonl"):
                    with open(jsonl_file) as f:
                        count += sum(1 for line in f if line.strip())
                return count
            with open(path) as f:
                return sum(1 for line in f if line.strip())
        except (OSError, FileNotFoundError) as e:
            _logger.error(f"Cannot read dataset file {path}: {e}")
            return 0

    endpoint: Annotated[
        EndpointConfig,
        Field(
            description="Endpoint configuration",
        ),
    ]

    input: Annotated[
        InputConfig,
        Field(
            description="Input configuration",
        ),
    ] = InputConfig()

    output: Annotated[
        OutputConfig,
        Field(
            description="Output configuration",
        ),
    ] = OutputConfig()

    tokenizer: Annotated[
        TokenizerConfig,
        Field(
            description="Tokenizer configuration",
        ),
    ] = TokenizerConfig()

    loadgen: Annotated[
        LoadGeneratorConfig,
        Field(
            description="Load Generator configuration",
        ),
    ] = LoadGeneratorConfig()

    accuracy: Annotated[
        AccuracyConfig,
        Field(
            description="Accuracy benchmarking configuration",
        ),
    ] = AccuracyConfig()

    cli_command: Annotated[
        str | None,
        Field(
            default=None,
            description="The CLI command for the user config.",
        ),
        DisableCLI(reason="This is automatically set by the CLI"),
    ] = None

    benchmark_id: Annotated[
        str | None,
        Field(
            default=None,
            description="Unique identifier for this benchmark run (UUID). Generated automatically and shared across all export formats for correlation.",
        ),
        DisableCLI(reason="This is automatically generated at runtime"),
    ] = None

    mlflow_tracking_uri: Annotated[
        str | None,
        Field(
            default=MLflowDefaults.TRACKING_URI,
            description=(
                "MLflow Tracking Server URI used for post-run uploads "
                "(e.g., http://localhost:5000). "
                "When set, AIPerf uploads params, metrics, tags, and artifacts "
                "(including plots) to MLflow after profiling completes."
            ),
        ),
        CLIParameter(
            name=("--mlflow-tracking-uri",),
            group=Groups.OUTPUT,
        ),
    ] = MLflowDefaults.TRACKING_URI

    mlflow_experiment: Annotated[
        str,
        Field(
            default=MLflowDefaults.EXPERIMENT,
            description=(
                "MLflow experiment name for post-run uploads. "
                "Requires --mlflow-tracking-uri to be set."
            ),
        ),
        CLIParameter(
            name=("--mlflow-experiment",),
            group=Groups.OUTPUT,
        ),
    ] = MLflowDefaults.EXPERIMENT

    mlflow_run_name: Annotated[
        str | None,
        Field(
            default=MLflowDefaults.RUN_NAME,
            description=(
                "Optional MLflow run name for post-run uploads. "
                "If omitted, AIPerf derives a name from benchmark metadata."
            ),
        ),
        CLIParameter(
            name=("--mlflow-run-name",),
            group=Groups.OUTPUT,
        ),
    ] = MLflowDefaults.RUN_NAME

    mlflow_tags: Annotated[
        list[tuple[str, str]] | None,
        Field(
            default=MLflowDefaults.TAGS,
            description=(
                "Additional MLflow run tags to attach on upload. "
                "Specify as key:value pairs (e.g., --mlflow-tag team:perf) "
                "or as JSON string."
            ),
        ),
        BeforeValidator(parse_str_or_dict_as_tuple_list),
        CLIParameter(
            name=("--mlflow-tag",),
            consume_multiple=True,
            group=Groups.OUTPUT,
        ),
    ] = MLflowDefaults.TAGS

    mlflow_artifact_globs: Annotated[
        list[str] | None,
        Field(
            default=MLflowDefaults.ARTIFACT_GLOBS,
            description=(
                "Optional artifact glob patterns for MLflow upload, relative to "
                "--output-artifact-dir. Can be specified multiple times. "
                "If not set, sensible defaults include exports and plot files."
            ),
        ),
        BeforeValidator(parse_str_or_list),
        CLIParameter(
            name=("--mlflow-artifact-glob",),
            consume_multiple=True,
            group=Groups.OUTPUT,
        ),
    ] = MLflowDefaults.ARTIFACT_GLOBS

    mlflow_parent_run_id: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Optional MLflow run id to attach this run to as a child "
                "(passed through to mlflow.start_run(parent_run_id=...)). "
                "Applied only when a new MLflow run is created; ignored on "
                "live-run reuse because MLflow pins the parent at creation time."
            ),
        ),
        CLIParameter(
            name=("--mlflow-parent-run-id",),
            group=Groups.OUTPUT,
        ),
    ] = None

    gpu_telemetry: Annotated[
        list[str] | None,
        Field(
            description=(
                "Enable GPU telemetry console display and optionally specify: "
                "(1) 'pynvml' to use local pynvml library instead of DCGM HTTP endpoints, "
                "(2) 'amdsmi' to use local amdsmi library for AMD ROCm GPUs, "
                "(3) 'dashboard' for realtime dashboard mode, "
                "(4) custom DCGM exporter URLs (e.g., http://node1:9401/metrics), "
                "(5) custom metrics CSV file (e.g., custom_gpu_metrics.csv). "
                "Default: DCGM mode with localhost:9400 and localhost:9401 endpoints. "
                "Examples: --gpu-telemetry pynvml | --gpu-telemetry amdsmi | --gpu-telemetry dashboard node1:9400"
            ),
        ),
        BeforeValidator(parse_str_or_list),
        CLIParameter(
            name=("--gpu-telemetry",),
            consume_multiple=True,
            group=Groups.TELEMETRY,
        ),
    ] = None

    no_gpu_telemetry: Annotated[
        bool,
        Field(
            description="Disable GPU telemetry collection entirely.",
        ),
        CLIParameter(
            name=("--no-gpu-telemetry",),
            group=Groups.TELEMETRY,
        ),
    ] = False

    otel_url: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Enable real-time metric streaming to an OpenTelemetry collector via OTLP. "
                "Requires the AIPerf otel extra (`aiperf[otel]`). "
                "Accepts one collector URL. "
                "The value can be a collector base URL or full OTLP metrics endpoint. "
                "If no path is specified, '/v1/metrics' is appended automatically. "
                "Examples: --otel-url localhost:4318 | --otel-url http://collector:4318 "
            ),
        ),
        CLIParameter(
            name=("--otel-url",),
            group=Groups.TELEMETRY,
        ),
    ] = None

    stream: Annotated[
        list[str] | None,
        Field(
            default=None,
            description=(
                "Select which AIPerf telemetry domains to stream over OTel. "
                "Valid values: 'metrics', 'timing', or 'default'. "
                "'default' streams both metrics and timing domains. "
                "If omitted and --otel-url is set, default behavior is used. "
                "Examples: --stream metrics | --stream timing | --stream default "
                "| --stream metrics timing"
            ),
        ),
        BeforeValidator(parse_str_or_list),
        CLIParameter(
            name=("--stream",),
            consume_multiple=True,
            group=Groups.TELEMETRY,
        ),
    ] = None

    otel_resource_attributes: Annotated[
        list[str] | None,
        Field(
            default=None,
            description=(
                "Custom OTel resource attributes as key=value pairs. "
                "Merged into the default resource attributes on every exported metric. "
                "Examples: --otel-resource-attributes team=inference "
                "| --otel-resource-attributes env=prod,region=us-west-2"
            ),
        ),
        BeforeValidator(parse_str_or_list),
        CLIParameter(
            name=("--otel-resource-attributes",),
            consume_multiple=True,
            group=Groups.TELEMETRY,
        ),
    ] = None

    gen_ai_provider: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Explicit value for the gen_ai.provider.name OTel attribute. "
                "When unset, AIPerf auto-infers from the endpoint URL host; "
                "falls back to '_OTHER' if no match. "
                "Example: --gen-ai-provider openai"
            ),
        ),
        CLIParameter(
            name=("--gen-ai-provider",),
            group=Groups.TELEMETRY,
        ),
    ] = None

    _gpu_telemetry_mode: GPUTelemetryMode = GPUTelemetryMode.SUMMARY
    _gpu_telemetry_collector_type: GPUTelemetryCollectorType = (
        GPUTelemetryCollectorType.DCGM
    )
    _gpu_telemetry_urls: list[str] = []
    _gpu_telemetry_metrics_file: Path | None = None
    _otel_metrics_url: str | None = None
    _otel_stream_metrics_enabled: bool = True
    _otel_stream_timing_enabled: bool = True

    @model_validator(mode="after")
    def _parse_gpu_telemetry_config(self) -> Self:
        """Parse gpu_telemetry list into mode, collector type, URLs, and metrics file."""
        if (
            "no_gpu_telemetry" in self.model_fields_set
            and "gpu_telemetry" in self.model_fields_set
        ):
            raise ValueError(
                "Cannot use both --no-gpu-telemetry and --gpu-telemetry together. "
                "Use only one or the other."
            )

        if not self.gpu_telemetry:
            return self

        mode, collector_type, urls, metrics_file = self._classify_gpu_telemetry_items(
            self.gpu_telemetry
        )

        if collector_type in _LOCAL_ONLY_COLLECTORS and urls:
            name = _LOCAL_ONLY_COLLECTORS[collector_type]
            raise ValueError(
                f"Cannot use {name} with DCGM URLs. Use either '{name}' for local "
                "GPU monitoring or URLs for DCGM endpoints, not both."
            )

        self._gpu_telemetry_mode = mode
        self._gpu_telemetry_collector_type = collector_type
        self._gpu_telemetry_urls = urls
        self._gpu_telemetry_metrics_file = metrics_file

        self._warn_if_local_collector_with_remote_urls(collector_type)
        return self

    @staticmethod
    def _classify_gpu_telemetry_items(
        items: list[str],
    ) -> tuple[GPUTelemetryMode, GPUTelemetryCollectorType, list[str], Path | None]:
        """Walk the ``--gpu-telemetry`` items and classify each one."""
        mode = GPUTelemetryMode.SUMMARY
        collector_type = GPUTelemetryCollectorType.DCGM
        urls: list[str] = []
        metrics_file: Path | None = None

        for item in items:
            lowered = item.lower()
            if item.endswith(".csv"):
                metrics_file = Path(item)
                if not metrics_file.exists():
                    raise ValueError(f"GPU metrics file not found: {item}")
            elif lowered in _LOCAL_COLLECTOR_KEYWORDS:
                selected = _LOCAL_COLLECTOR_KEYWORDS[lowered]
                if (
                    collector_type in _LOCAL_ONLY_COLLECTORS
                    and collector_type != selected
                ):
                    prior = _LOCAL_ONLY_COLLECTORS[collector_type]
                    chosen = _LOCAL_ONLY_COLLECTORS[selected]
                    raise ValueError(
                        f"Conflicting local GPU telemetry collectors: "
                        f"'{prior}' and '{chosen}'. Choose exactly one."
                    )
                collector_type = selected
                _ensure_local_collector_importable(collector_type)
            elif item == "dashboard":
                mode = GPUTelemetryMode.REALTIME_DASHBOARD
            elif item.startswith("http") or ":" in item:
                normalized_url = item if item.startswith("http") else f"http://{item}"
                urls.append(normalized_url)
            else:
                raise ValueError(
                    f"Invalid GPU telemetry item: {item}. Valid options are: "
                    "'pynvml', 'amdsmi', 'dashboard', '.csv' file, and URLs."
                )
        return mode, collector_type, urls, metrics_file

    def _warn_if_local_collector_with_remote_urls(
        self, collector_type: GPUTelemetryCollectorType
    ) -> None:
        """Warn when a local-only collector is paired with non-localhost servers."""
        if collector_type not in _LOCAL_ONLY_COLLECTORS:
            return
        name = _LOCAL_ONLY_COLLECTORS[collector_type]
        non_local_urls = [
            url for url in self.endpoint.urls if not _is_localhost_url(url)
        ]
        if non_local_urls:
            _logger.warning(
                f"Using {name} for GPU telemetry with non-localhost server URL(s): {non_local_urls}. "
                f"{name} collects GPU metrics from the local machine only. "
                "If the inference server is running remotely, the GPU telemetry will not reflect "
                "the server's GPU usage. Consider using DCGM mode with the server's metrics endpoint instead."
            )

    @property
    def gpu_telemetry_mode(self) -> GPUTelemetryMode:
        """Get the GPU telemetry display mode (parsed from gpu_telemetry list)."""
        return self._gpu_telemetry_mode

    @gpu_telemetry_mode.setter
    def gpu_telemetry_mode(self, value: GPUTelemetryMode) -> None:
        """Set the GPU telemetry display mode."""
        self._gpu_telemetry_mode = value

    @property
    def gpu_telemetry_collector_type(self) -> GPUTelemetryCollectorType:
        """Get the GPU telemetry collector type (DCGM or PYNVML)."""
        return self._gpu_telemetry_collector_type

    @property
    def gpu_telemetry_urls(self) -> list[str]:
        """Get the parsed GPU telemetry DCGM endpoint URLs."""
        return self._gpu_telemetry_urls

    @property
    def gpu_telemetry_metrics_file(self) -> Path | None:
        """Get the path to custom GPU metrics CSV file."""
        return self._gpu_telemetry_metrics_file

    @property
    def gpu_telemetry_disabled(self) -> bool:
        """Check if GPU telemetry collection is disabled."""
        return self.no_gpu_telemetry

    @model_validator(mode="after")
    def _parse_otel_config(self) -> Self:
        """Parse and normalize OTel collector URL configuration."""
        valid_telemetry_values = {"metrics", "timing", "default"}
        selected_values = [value.lower() for value in (self.stream or [])]

        if not selected_values:
            # Default behavior is to stream both domains when selection is omitted.
            self._otel_stream_metrics_enabled = True
            self._otel_stream_timing_enabled = True
        else:
            invalid_values = [
                value
                for value in selected_values
                if value not in valid_telemetry_values
            ]
            if invalid_values:
                raise ValueError(
                    "Invalid --stream value(s): "
                    + ", ".join(sorted(set(invalid_values)))
                    + ". "
                    "Valid options are: metrics, timing, default."
                )
            if "default" in selected_values:
                # Default mode means stream both domains, even if combined with others.
                self._otel_stream_metrics_enabled = True
                self._otel_stream_timing_enabled = True
            else:
                self._otel_stream_metrics_enabled = "metrics" in selected_values
                self._otel_stream_timing_enabled = "timing" in selected_values

        if self.otel_url is None:
            # Warn/reject if OTel secondary options set without --otel-url.
            # gen_ai_provider is included because it is consumed only by the OTel
            # GenAI-semconv strategy (see infer_provider_name in genai_semconv.py),
            # so passing it without --otel-url is a silent no-op for the user.
            has_otel_secondary = bool(
                self.stream or self.otel_resource_attributes or self.gen_ai_provider
            )
            if has_otel_secondary:
                raise ValueError(
                    "--stream, --otel-resource-attributes, and --gen-ai-provider "
                    "require --otel-url to be set."
                )
            self._otel_metrics_url = None
            return self

        self._otel_metrics_url = _normalize_otel_metrics_url(self.otel_url)
        return self

    @model_validator(mode="after")
    def _validate_otel_resource_attributes(self) -> Self:
        """Reject malformed --otel-resource-attributes entries.

        Each entry must be ``key=value``. Missing ``=``, empty key, and empty
        value are rejected — silently dropping them or emitting empty keys/
        values produces OTLP resource attributes that tools like the
        Collector and MLflow reject or display as ``""``.
        """
        if not self.otel_resource_attributes:
            return self
        for item in self.otel_resource_attributes:
            for pair in item.split(","):
                pair = pair.strip()
                if not pair:
                    continue
                if "=" not in pair:
                    raise ValueError(
                        f"Invalid --otel-resource-attributes entry {pair!r}: "
                        "expected key=value."
                    )
                key, _, value = pair.partition("=")
                key = key.strip()
                value = value.strip()
                if not key:
                    raise ValueError(
                        f"Invalid --otel-resource-attributes entry {pair!r}: "
                        "key cannot be empty."
                    )
                if not value:
                    raise ValueError(
                        f"Invalid --otel-resource-attributes entry {pair!r}: "
                        "value cannot be empty."
                    )
        return self

    @model_validator(mode="after")
    def _validate_mlflow_config(self) -> Self:
        """Validate and normalize MLflow post-run upload configuration."""
        if self.mlflow_artifact_globs is not None:
            normalized_globs: list[str] = []
            for glob in self.mlflow_artifact_globs:
                normalized_glob = glob.strip()
                if not normalized_glob:
                    raise ValueError("--mlflow-artifact-glob entries cannot be empty.")
                normalized_globs.append(normalized_glob)
            self.mlflow_artifact_globs = normalized_globs

        if self.mlflow_tracking_uri is None:
            has_secondary = (
                self.mlflow_experiment != MLflowDefaults.EXPERIMENT
                or self.mlflow_run_name is not None
                or self.mlflow_tags is not None
                or self.mlflow_artifact_globs is not None
                or self.mlflow_parent_run_id is not None
            )
            if has_secondary:
                raise ValueError(
                    "--mlflow-experiment, --mlflow-run-name, --mlflow-tag, "
                    "--mlflow-artifact-glob, and --mlflow-parent-run-id "
                    "require --mlflow-tracking-uri to be set."
                )
            return self

        tracking_uri = self.mlflow_tracking_uri.strip()
        if not tracking_uri:
            raise ValueError("--mlflow-tracking-uri cannot be empty.")
        self.mlflow_tracking_uri = tracking_uri

        if not self.mlflow_experiment.strip():
            raise ValueError(
                "--mlflow-experiment cannot be empty when --mlflow-tracking-uri is set."
            )
        self.mlflow_experiment = self.mlflow_experiment.strip()

        if self.mlflow_run_name is not None:
            run_name = self.mlflow_run_name.strip()
            self.mlflow_run_name = run_name or None

        return self

    @property
    def otel_metrics_url(self) -> str | None:
        """Get the normalized OTLP/HTTP metrics endpoint URL."""
        return self._otel_metrics_url

    @property
    def otel_collector_enabled(self) -> bool:
        """Check if an OpenTelemetry collector sink is configured."""
        return bool(self._otel_metrics_url)

    @property
    def otel_custom_resource_attributes(self) -> dict[str, str]:
        """Parse --otel-resource-attributes into a dict of key=value pairs.

        Pre-validated by ``_validate_otel_resource_attributes`` — malformed
        entries raise there, so this accessor can assume well-formed input.
        """
        if not self.otel_resource_attributes:
            return {}
        attrs: dict[str, str] = {}
        for item in self.otel_resource_attributes:
            for pair in item.split(","):
                pair = pair.strip()
                if not pair:
                    continue
                key, _, value = pair.partition("=")
                attrs[key.strip()] = value.strip()
        return attrs

    @property
    def otel_stream_metrics_enabled(self) -> bool:
        """Check if request-level metric telemetry is enabled for OTel streaming."""
        return self._otel_stream_metrics_enabled

    @property
    def otel_stream_timing_enabled(self) -> bool:
        """Check if phase-level timing telemetry is enabled for OTel streaming."""
        return self._otel_stream_timing_enabled

    @property
    def gen_ai_provider_name(self) -> str:
        """Resolved gen_ai.provider.name attribute value.

        (a) explicit --gen-ai-provider override,
        (b) auto-infer from endpoint URL host,
        (c) literal '_OTHER'.

        Plain ``@property`` (not cached) because ``UserConfig`` is a Pydantic
        model without ``frozen=True``; caching the first access would go stale
        if a test or caller later mutates ``gen_ai_provider`` or
        ``endpoint.urls``. The inference runs a handful of small regexes and is
        called once per OTel record attribute build — not worth the caching risk.
        """
        from aiperf.post_processors.strategies.genai_semconv import infer_provider_name

        return infer_provider_name(self)

    @property
    def mlflow_enabled(self) -> bool:
        """Check if MLflow post-run upload is enabled."""
        return self.mlflow_tracking_uri is not None

    @property
    def mlflow_tags_dict(self) -> dict[str, str]:
        """Get MLflow tags as a normalized dict[str, str]."""
        tags: dict[str, str] = {}
        for key, value in self.mlflow_tags or []:
            key_str = str(key).strip()
            if not key_str:
                continue
            tags[key_str] = str(value)
        return tags

    @property
    def mlflow_resolved_artifact_globs(self) -> list[str] | tuple[str, ...]:
        """Get explicit or default artifact glob patterns for MLflow upload."""
        if self.mlflow_artifact_globs:
            return self.mlflow_artifact_globs
        return MLflowDefaults.DEFAULT_ARTIFACT_GLOBS

    server_metrics: Annotated[
        list[str] | None,
        Field(
            description=(
                "Server metrics collection (ENABLED BY DEFAULT). "
                "Automatically collects from inference endpoint base_url + `/metrics`. "
                "Optionally specify additional custom Prometheus-compatible endpoint URLs "
                "(e.g., http://node1:8081/metrics, http://node2:9090/metrics). "
                "Use `--no-server-metrics` to disable collection. "
                "Example: `--server-metrics node1:8081 node2:9090/metrics` for additional endpoints"
            ),
        ),
        BeforeValidator(parse_str_or_list),
        CLIParameter(
            name=("--server-metrics",),
            consume_multiple=True,
            group=Groups.SERVER_METRICS,
        ),
    ] = None

    no_server_metrics: Annotated[
        bool,
        Field(
            description="Disable server metrics collection entirely.",
        ),
        CLIParameter(
            name=("--no-server-metrics",),
            group=Groups.SERVER_METRICS,
        ),
    ] = False

    server_metrics_formats: Annotated[
        list[ServerMetricsFormat],
        Field(
            description=(
                "Specify which output formats to generate for server metrics. "
                "Multiple formats can be specified (e.g., `--server-metrics-formats json csv parquet`)."
            ),
        ),
        BeforeValidator(parse_str_or_list),
        CLIParameter(
            name=("--server-metrics-formats",),
            consume_multiple=True,
            group=Groups.SERVER_METRICS,
        ),
    ] = ServerMetricsDefaults.DEFAULT_FORMATS

    _server_metrics_urls: list[str] = []

    @model_validator(mode="after")
    def _parse_server_metrics_config(self) -> Self:
        """Parse server_metrics list into URLs.

        Empty list [] means enabled with automatic discovery only.
        Non-empty list means enabled with custom URLs.
        Use --no-server-metrics to disable collection.
        """
        from aiperf.common.metric_utils import normalize_metrics_endpoint_url

        if (
            "no_server_metrics" in self.model_fields_set
            and "server_metrics" in self.model_fields_set
        ):
            raise ValueError(
                "Cannot use both --no-server-metrics and --server-metrics together. "
                "Use only one or the other."
            )

        urls: list[str] = []

        for item in self.server_metrics or []:
            # Check for URLs (anything with : or starting with http)
            if item.startswith("http") or ":" in item:
                normalized_url = item if item.startswith("http") else f"http://{item}"
                normalized_url = normalize_metrics_endpoint_url(normalized_url)
                urls.append(normalized_url)

        self._server_metrics_urls = urls
        return self

    @property
    def server_metrics_disabled(self) -> bool:
        """Check if server metrics collection is disabled."""
        return self.no_server_metrics

    @property
    def server_metrics_urls(self) -> list[str]:
        """Get the parsed server metrics Prometheus endpoint URLs."""
        return self._server_metrics_urls

    @model_validator(mode="after")
    def _compute_config(self) -> Self:
        """Compute additional configuration.

        This method is automatically called after the model is validated to compute additional configuration.
        """

        if "artifact_directory" not in self.output.model_fields_set:
            self.output.artifact_directory = self._compute_artifact_directory()

        return self

    def _compute_artifact_directory(self) -> Path:
        """Compute the artifact directory based on the user selected options."""
        names: list[str] = [
            self._get_artifact_model_name(),
            self._get_artifact_service_kind(),
            self._get_artifact_stimulus(),
        ]
        return self.output.artifact_directory / "-".join(names)

    def _get_artifact_model_name(self) -> str:
        """Get the artifact model name based on the user selected options."""
        model_name: str = self.endpoint.model_names[0]
        if len(self.endpoint.model_names) > 1:
            model_name = f"{model_name}_multi"

        # Preprocess Huggingface model names that include '/' in their model name.
        if "/" in model_name:
            filtered_name = "_".join(model_name.split("/"))

            _logger.info(
                f"Model name '{model_name}' cannot be used to create artifact "
                f"directory. Instead, '{filtered_name}' will be used."
            )
            model_name = filtered_name
        return model_name

    def _get_artifact_service_kind(self) -> str:
        """Get the service kind name based on the endpoint config."""
        metadata = self._endpoint_metadata()
        return f"{metadata.service_kind}-{self.endpoint.type}"

    def _get_artifact_stimulus(self) -> str:
        """Get the stimulus name based on the timing mode."""
        match self._timing_mode:
            case TimingMode.REQUEST_RATE:
                stimulus = []
                if self.loadgen.concurrency is not None:
                    if isinstance(self.loadgen.concurrency, list):
                        stimulus.append(
                            f"concurrency_sweep_{'_'.join(map(str, self.loadgen.concurrency))}"
                        )
                    else:
                        stimulus.append(f"concurrency{self.loadgen.concurrency}")
                if self.loadgen.request_rate is not None:
                    stimulus.append(f"request_rate{self.loadgen.request_rate}")
                return "-".join(stimulus)
            case TimingMode.FIXED_SCHEDULE:
                return "fixed_schedule"
            case TimingMode.USER_CENTRIC_RATE:
                stimulus = ["user_centric"]
                if self.loadgen.num_users is not None:
                    stimulus.append(f"users{self.loadgen.num_users}")
                if self.loadgen.user_centric_rate is not None:
                    stimulus.append(f"qps{self.loadgen.user_centric_rate}")
                return "-".join(stimulus)
            case _:
                raise ValueError(f"Unknown timing mode '{self._timing_mode}'.")

    @property
    def timing_mode(self) -> TimingMode:
        """Get the timing mode based on the user config."""
        return self._timing_mode

    @model_validator(mode="after")
    def validate_multi_turn_options(self) -> Self:
        """Validate multi-turn options."""
        # Multi-turn validation: only one of request_count or num_sessions should be set
        if (
            self.loadgen.request_count is not None
            and self.input.conversation.num is not None
        ):
            raise ValueError(
                "Both a request-count and number of conversations are set. This can result in confusing output. "
                "Use either --request-count or --conversation-num but not both."
            )

        # Same validation for warmup options
        if (
            self.loadgen.warmup_request_count is not None
            and self.loadgen.warmup_num_sessions is not None
        ):
            raise ValueError(
                "Both --warmup-request-count and --num-warmup-sessions are set. "
                "Use either --warmup-request-count or --num-warmup-sessions but not both."
            )

        return self

    @model_validator(mode="after")
    def validate_concurrency_limits(self) -> Self:
        """Validate that concurrency does not exceed the appropriate limit."""
        if self.loadgen.concurrency is None:
            return self

        # Get concurrency values to check (handle both int and list)
        concurrency_values = (
            [self.loadgen.concurrency]
            if isinstance(self.loadgen.concurrency, int)
            else self.loadgen.concurrency
        )

        # For multi-turn scenarios, check against conversation_num
        if self.input.conversation.num is not None:
            for concurrency in concurrency_values:
                if concurrency > self.input.conversation.num:
                    raise ValueError(
                        f"Concurrency ({concurrency}) cannot be greater than "
                        f"the number of conversations ({self.input.conversation.num}). "
                        "Either reduce --concurrency or increase --conversation-num."
                    )
        # For single-turn scenarios, check against request_count if it is set
        elif self.loadgen.request_count is not None:
            for concurrency in concurrency_values:
                if concurrency > self.loadgen.request_count:
                    raise ValueError(
                        f"Concurrency ({concurrency}) cannot be greater than "
                        f"the request count ({self.loadgen.request_count}). Either reduce "
                        "--concurrency or increase --request-count."
                    )

        return self

    @model_validator(mode="after")
    def validate_prefill_concurrency(self) -> Self:
        """Validate prefill_concurrency configuration.

        Prefill concurrency requires:
        1. Streaming to be enabled (FirstToken event is only available with streaming)
        2. prefill_concurrency <= concurrency (cannot have more prefill slots than total slots)
        """
        prefill_concurrency = self.loadgen.prefill_concurrency
        warmup_prefill_concurrency = self.loadgen.warmup_prefill_concurrency

        # Check if any prefill concurrency is set
        if prefill_concurrency is None and warmup_prefill_concurrency is None:
            return self

        # Validate streaming requirement
        if not self.endpoint.streaming:
            raise ValueError(
                "--prefill-concurrency requires --streaming to be enabled. "
                "Prefill concurrency relies on FirstToken events which are only "
                "available with streaming responses."
            )

        # Validate prefill_concurrency <= concurrency
        # For sweep mode, check against all concurrency values
        if prefill_concurrency is not None and self.loadgen.concurrency is not None:
            concurrency_values = (
                [self.loadgen.concurrency]
                if isinstance(self.loadgen.concurrency, int)
                else self.loadgen.concurrency
            )
            for concurrency in concurrency_values:
                if prefill_concurrency > concurrency:
                    raise ValueError(
                        f"--prefill-concurrency ({prefill_concurrency}) cannot be greater than "
                        f"--concurrency ({concurrency}). "
                        "Prefill concurrency limits how many requests can be in the prefill stage, "
                        "which cannot exceed the total concurrent requests."
                    )

        # Validate warmup_prefill_concurrency <= warmup_concurrency (or concurrency)
        if warmup_prefill_concurrency is not None:
            effective_warmup_concurrency = (
                self.loadgen.warmup_concurrency or self.loadgen.concurrency
            )
            if effective_warmup_concurrency is not None:
                # Handle list concurrency for warmup
                warmup_concurrency_values = (
                    [effective_warmup_concurrency]
                    if isinstance(effective_warmup_concurrency, int)
                    else effective_warmup_concurrency
                )
                for warmup_concurrency in warmup_concurrency_values:
                    if warmup_prefill_concurrency > warmup_concurrency:
                        raise ValueError(
                            f"--warmup-prefill-concurrency ({warmup_prefill_concurrency}) cannot be "
                            f"greater than warmup concurrency ({warmup_concurrency}). "
                            "Prefill concurrency limits how many requests can be in the prefill stage, "
                            "which cannot exceed the total concurrent requests."
                        )

        return self

    @model_validator(mode="after")
    def validate_dataset_sampling_strategy(self) -> Self:
        """Validate that the dataset sampling strategy is compatible with the timing mode."""
        if (
            self.timing_mode == TimingMode.FIXED_SCHEDULE
            and self.input.dataset_sampling_strategy is not None
        ):
            raise ValueError(
                "Dataset sampling strategy is not compatible with fixed schedule mode. "
                "Please remove the --dataset-sampling-strategy option."
            )
        return self

    @model_validator(mode="after")
    def validate_user_context_requires_dataset_entries(self) -> Self:
        """Validate that user context prompt requires num-dataset-entries to be specified."""
        if (
            self.input.prompt.prefix_prompt.user_context_prompt_length is not None
            and "num_dataset_entries" not in self.input.conversation.model_fields_set
        ):
            raise ValueError(
                "--user-context-prompt-length requires --num-dataset-entries to be specified. "
                "Each dataset entry needs a unique user context prompt, so the number of dataset entries must be defined."
            )
        return self

    @model_validator(mode="after")
    def validate_mutually_exclusive_prompt_options(self) -> Self:
        """Ensure shared system/user context options don't conflict with legacy prefix options."""
        has_context_prompts = (
            self.input.prompt.prefix_prompt.shared_system_prompt_length is not None
            or self.input.prompt.prefix_prompt.user_context_prompt_length is not None
        )
        has_legacy_prefix = (
            self.input.prompt.prefix_prompt.length > 0
            or self.input.prompt.prefix_prompt.pool_size > 0
        )

        if has_context_prompts and has_legacy_prefix:
            raise ValueError(
                "Cannot use both `--shared-system-prompt-length`/`--user-context-prompt-length` "
                "and `--prefix-prompt-length`/`--prefix-prompt-pool-size`. "
                "These are mutually exclusive prompt configuration modes."
            )
        return self

    @model_validator(mode="after")
    def validate_rankings_token_options(self) -> Self:
        """Validate rankings token options usage."""

        # Check if prompt input tokens have been changed from defaults
        prompt_tokens_modified = any(
            field in self.input.prompt.input_tokens.model_fields_set
            for field in ["mean", "stddev"]
        )

        # Check if any rankings-specific token options have been changed from defaults
        rankings_tokens_modified = any(
            field in self.input.rankings.passages.model_fields_set
            for field in ["prompt_token_mean", "prompt_token_stddev"]
        ) or any(
            field in self.input.rankings.query.model_fields_set
            for field in ["prompt_token_mean", "prompt_token_stddev"]
        )

        # Check if any rankings-specific passage options have been changed from defaults
        rankings_passages_modified = any(
            field in self.input.rankings.passages.model_fields_set
            for field in ["mean", "stddev"]
        )

        rankings_options_modified = (
            rankings_tokens_modified or rankings_passages_modified
        )

        endpoint_type_is_rankings = "rankings" in self.endpoint.type.lower()

        # Validate that rankings options are only used with rankings endpoints
        rankings_endpoints = [
            endpoint_type
            for endpoint_type in EndpointType
            if "rankings" in endpoint_type.lower()
        ]
        if rankings_options_modified and not endpoint_type_is_rankings:
            raise ValueError(
                f"Rankings-specific options (`--rankings-passages-mean`, `--rankings-passages-stddev`, "
                "`--rankings-passages-prompt-token-mean`, `--rankings-passages-prompt-token-stddev`, "
                "`--rankings-query-prompt-token-mean`, `--rankings-query-prompt-token-stddev`) "
                "can only be used with rankings endpoint types "
                f"Rankings endpoints: ({', '.join(rankings_endpoints)})."
            )

        # Validate that prompt tokens and rankings tokens are not both set
        if prompt_tokens_modified and (
            rankings_tokens_modified or endpoint_type_is_rankings
        ):
            raise ValueError(
                "The `--prompt-input-tokens-mean`/`--prompt-input-tokens-stddev` options "
                "cannot be used together with rankings-specific token options or the rankings endpoints"
                "Ranking options: (`--rankings-passages-prompt-token-mean`, `--rankings-passages-prompt-token-stddev`, "
                "`--rankings-query-prompt-token-mean`, `--rankings-query-prompt-token-stddev`). "
                f"Rankings endpoints: ({', '.join(rankings_endpoints)})."
                "Please use only one set of options."
            )
        return self

    @model_validator(mode="after")
    def default_no_text_for_non_tokenizing_endpoints(self) -> Self:
        """Reject explicit text options and zero out text defaults for non-tokenizing
        endpoints (e.g., image_retrieval)."""
        metadata = self._endpoint_metadata()
        if metadata.tokenizes_input:
            return self

        def err(option: str) -> ValueError:
            return ValueError(
                f"{option} cannot be used with "
                f"--endpoint-type {self.endpoint.type} because it does not "
                "support text input."
            )

        if (
            "mean" in self.input.prompt.input_tokens.model_fields_set
            and self.input.prompt.input_tokens.mean > 0
        ):
            raise err("Synthetic input token mean (--synthetic-input-tokens-mean)")
        else:
            self.input.prompt.input_tokens.mean = 0

        if (
            "stddev" in self.input.prompt.input_tokens.model_fields_set
            and self.input.prompt.input_tokens.stddev > 0
        ):
            raise err("Synthetic input token stddev (--synthetic-input-tokens-stddev)")
        else:
            self.input.prompt.input_tokens.stddev = 0

        if (
            "batch_size" in self.input.prompt.model_fields_set
            and self.input.prompt.batch_size > 0
        ):
            raise err("Text batch size (--batch-size-text)")
        else:
            self.input.prompt.batch_size = 0

        if self.input.prompt.sequence_distribution is not None:
            raise err("Sequence distribution (--sequence-distribution)")

        if self.input.prompt.prefix_prompt.model_fields_set:
            raise err("Prefix prompt options")

        return self

    @model_validator(mode="after")
    def reject_tokenizer_for_non_token_endpoints(self) -> Self:
        """Reject --tokenizer* flags when the endpoint neither tokenizes input nor
        produces tokens."""
        metadata = self._endpoint_metadata()
        if metadata.tokenizes_input or metadata.produces_tokens:
            return self

        user_set = self.tokenizer.model_fields_set - {"resolved_names"}
        if user_set:
            raise ValueError(
                "Tokenizer options cannot be used with "
                f"--endpoint-type {self.endpoint.type} because it does not "
                "tokenize input or produce tokens."
            )

        return self

    @model_validator(mode="after")
    def validate_must_have_stop_condition(self) -> Self:
        """Validate that at least one stop condition is set (requests, sessions, or duration)"""
        if (
            self.loadgen.request_count is None
            and self.input.conversation.num is None
            and self.loadgen.benchmark_duration is None
        ):
            raise ValueError(
                "At least one stop condition must be set (--request-count, --num-sessions, or --benchmark-duration)"
            )
        return self

    @model_validator(mode="after")
    def validate_accuracy_config(self) -> Self:
        """Validate accuracy benchmarking configuration."""
        # Stub: validation logic will be added when accuracy mode is implemented
        return self
