# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path
from typing import Annotated, Any

from orjson import JSONDecodeError
from pydantic import BeforeValidator, Field, model_validator
from typing_extensions import Self

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.cli_parameter import CLIParameter, DisableCLI
from aiperf.common.config.config_defaults import (
    LoadGeneratorDefaults,
    ServerMetricsDefaults,
)
from aiperf.common.config.config_validators import coerce_value, parse_str_or_list
from aiperf.common.config.endpoint_config import EndpointConfig
from aiperf.common.config.groups import Groups
from aiperf.common.config.input_config import InputConfig
from aiperf.common.config.loadgen_config import LoadGeneratorConfig
from aiperf.common.config.output_config import OutputConfig
from aiperf.common.config.tokenizer_config import TokenizerConfig
from aiperf.common.enums import (
    GPUTelemetryMode,
    ServerMetricsFormat,
)
from aiperf.common.utils import load_json_str
from aiperf.plugin.enums import (
    ArrivalPattern,
    CustomDatasetType,
    EndpointType,
    TimingMode,
)

_logger = AIPerfLogger(__name__)


def _should_quote_arg(x: Any) -> bool:
    """Determine if the value should be quoted in the CLI command."""
    return isinstance(x, str) and not x.startswith("-") and x not in ("profile")


class UserConfig(BaseConfig):
    """
    A configuration class for defining top-level user settings.
    """

    _timing_mode: TimingMode = TimingMode.REQUEST_RATE

    @model_validator(mode="after")
    def validate_cli_args(self) -> Self:
        """Set the CLI command based on the command line arguments, if it has not already been set."""
        if not self.cli_command:
            args = [coerce_value(x) for x in sys.argv[1:]]
            # Note: Use single quotes to avoid conflicts with double quotes in arguments.
            args = [f"'{x}'" if _should_quote_arg(x) else str(x) for x in args]
            self.cli_command = " ".join(["aiperf", *args])
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
        elif self._should_use_fixed_schedule_for_mooncake_trace():
            self._timing_mode = TimingMode.FIXED_SCHEDULE
            _logger.info(
                "Automatically enabling fixed schedule mode for mooncake_trace dataset with timestamps"
            )
            if (
                self.loadgen.request_count is None
                and self.input.conversation.num is None
            ):
                self.loadgen.request_count = self._count_dataset_entries()
                _logger.info(
                    f"No request count value provided for mooncake trace dataset, setting to dataset entry count: {self.loadgen.request_count}"
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
                _logger.warning(
                    f"No request count value provided, setting to {LoadGeneratorDefaults.MIN_REQUEST_COUNT}"
                )
                self.loadgen.request_count = LoadGeneratorDefaults.MIN_REQUEST_COUNT
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
                self.loadgen.request_count = max(
                    LoadGeneratorDefaults.MIN_REQUEST_COUNT,
                    effective_concurrency
                    * LoadGeneratorDefaults.REQUEST_COUNT_MULTIPLIER,
                )
                _logger.warning(
                    f"No request count value provided, setting to {self.loadgen.request_count}"
                )
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

    def _should_use_fixed_schedule_for_mooncake_trace(self) -> bool:
        """Check if mooncake_trace dataset has timestamps and should use fixed schedule.

        Returns:
            bool: True if fixed schedule should be enabled for this mooncake trace
        """
        if self.input.custom_dataset_type != CustomDatasetType.MOONCAKE_TRACE:
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

    def _count_dataset_entries(self) -> int:
        """Count the number of valid entries in a custom dataset file.

        Returns:
            int: Number of non-empty lines in the file
        """
        if not self.input.file:
            return 0

        try:
            with open(self.input.file) as f:
                return sum(1 for line in f if line.strip())
        except (OSError, FileNotFoundError) as e:
            _logger.error(f"Cannot read dataset file {self.input.file}: {e}")
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

    gpu_telemetry: Annotated[
        list[str] | None,
        Field(
            description=(
                "Enable GPU telemetry console display and optionally specify: "
                "(1) 'dashboard' for realtime dashboard mode, "
                "(2) custom DCGM exporter URLs (e.g., http://node1:9401/metrics), "
                "(3) custom metrics CSV file (e.g., custom_gpu_metrics.csv). "
                "Default endpoints localhost:9400 and localhost:9401 are always attempted. "
                "Example: --gpu-telemetry dashboard node1:9400 custom.csv"
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

    _gpu_telemetry_mode: GPUTelemetryMode = GPUTelemetryMode.SUMMARY
    _gpu_telemetry_urls: list[str] = []
    _gpu_telemetry_metrics_file: Path | None = None

    @model_validator(mode="after")
    def _parse_gpu_telemetry_config(self) -> Self:
        """Parse gpu_telemetry list into mode, URLs, and metrics file."""
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

        mode = GPUTelemetryMode.SUMMARY
        urls = []
        metrics_file = None

        for item in self.gpu_telemetry:
            # Check for CSV file (file extension heuristic)
            if item.endswith(".csv"):
                metrics_file = Path(item)
                if not metrics_file.exists():
                    raise ValueError(f"GPU metrics file not found: {item}")
                continue

            # Check for dashboard mode
            if item in ["dashboard"]:
                mode = GPUTelemetryMode.REALTIME_DASHBOARD
            # Check for URLs
            elif item.startswith("http") or ":" in item:
                normalized_url = item if item.startswith("http") else f"http://{item}"
                urls.append(normalized_url)

        self._gpu_telemetry_mode = mode
        self._gpu_telemetry_urls = urls
        self._gpu_telemetry_metrics_file = metrics_file
        return self

    @property
    def gpu_telemetry_mode(self) -> GPUTelemetryMode:
        """Get the GPU telemetry display mode (parsed from gpu_telemetry list)."""
        return self._gpu_telemetry_mode

    @gpu_telemetry_mode.setter
    def gpu_telemetry_mode(self, value: GPUTelemetryMode) -> None:
        """Set the GPU telemetry display mode."""
        self._gpu_telemetry_mode = value

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
        # Lazy import to avoid circular dependency
        from aiperf.plugin import plugins

        metadata = plugins.get_endpoint_metadata(self.endpoint.type)
        return f"{metadata.service_kind}-{self.endpoint.type}"

    def _get_artifact_stimulus(self) -> str:
        """Get the stimulus name based on the timing mode."""
        match self._timing_mode:
            case TimingMode.REQUEST_RATE:
                stimulus = []
                if self.loadgen.concurrency is not None:
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

        # For multi-turn scenarios, check against conversation_num
        if (
            self.input.conversation.num is not None
            and self.loadgen.concurrency > self.input.conversation.num
        ):
            raise ValueError(
                f"Concurrency ({self.loadgen.concurrency}) cannot be greater than "
                f"the number of conversations ({self.input.conversation.num}). "
                "Either reduce --concurrency or increase --conversation-num."
            )
        # For single-turn scenarios, check against request_count if it is set
        elif (
            self.loadgen.request_count is not None
            and self.loadgen.concurrency > self.loadgen.request_count
        ):
            raise ValueError(
                f"Concurrency ({self.loadgen.concurrency}) cannot be greater than "
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
        if (
            prefill_concurrency is not None
            and self.loadgen.concurrency is not None
            and prefill_concurrency > self.loadgen.concurrency
        ):
            raise ValueError(
                f"--prefill-concurrency ({prefill_concurrency}) cannot be greater than "
                f"--concurrency ({self.loadgen.concurrency}). "
                "Prefill concurrency limits how many requests can be in the prefill stage, "
                "which cannot exceed the total concurrent requests."
            )

        # Validate warmup_prefill_concurrency <= warmup_concurrency (or concurrency)
        if warmup_prefill_concurrency is not None:
            effective_warmup_concurrency = (
                self.loadgen.warmup_concurrency or self.loadgen.concurrency
            )
            if (
                effective_warmup_concurrency is not None
                and warmup_prefill_concurrency > effective_warmup_concurrency
            ):
                raise ValueError(
                    f"--warmup-prefill-concurrency ({warmup_prefill_concurrency}) cannot be "
                    f"greater than warmup concurrency ({effective_warmup_concurrency}). "
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
