# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from aiperf.common.config.config_validators import coerce_value, parse_str_or_list
from aiperf.common.config.endpoint_config import EndpointConfig
from aiperf.common.config.groups import Groups
from aiperf.common.config.input_config import InputConfig
from aiperf.common.config.loadgen_config import LoadGeneratorConfig
from aiperf.common.config.output_config import OutputConfig
from aiperf.common.config.tokenizer_config import TokenizerConfig
from aiperf.common.enums import CustomDatasetType, GPUTelemetryMode
from aiperf.common.enums.plugin_enums import EndpointType
from aiperf.common.enums.timing_enums import RequestRateMode, TimingMode
from aiperf.common.utils import load_json_str

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
            args = [f'"{x}"' if _should_quote_arg(x) else str(x) for x in args]
            self.cli_command = " ".join(["aiperf", *args])
        return self

    @model_validator(mode="after")
    def validate_timing_mode(self) -> Self:
        """Set the timing mode based on the user config. Will be called after all user config is set."""
        if self.input.fixed_schedule:
            self._timing_mode = TimingMode.FIXED_SCHEDULE
        elif self._should_use_fixed_schedule_for_mooncake_trace():
            self._timing_mode = TimingMode.FIXED_SCHEDULE
            _logger.info(
                "Automatically enabling fixed schedule mode for mooncake_trace dataset with timestamps"
            )
        elif self.loadgen.request_rate is not None:
            # Request rate is checked first, as if user has provided request rate and concurrency,
            # we will still use the request rate strategy.
            self._timing_mode = TimingMode.REQUEST_RATE
            if self.loadgen.request_rate_mode == RequestRateMode.CONCURRENCY_BURST:
                raise ValueError(
                    f"Request rate mode cannot be {RequestRateMode.CONCURRENCY_BURST!r} when a request rate is specified."
                )
        else:
            # Default to concurrency burst mode if no request rate or schedule is provided
            if self.loadgen.concurrency is None:
                # If user has not provided a concurrency value, set it to 1
                self.loadgen.concurrency = 1
            self._timing_mode = TimingMode.REQUEST_RATE
            self.loadgen.request_rate_mode = RequestRateMode.CONCURRENCY_BURST

        return self

    @model_validator(mode="after")
    def validate_benchmark_mode(self) -> Self:
        """Validate benchmarking is count-based or timing-based, plus associated args are correctly set."""
        if (
            "benchmark_duration" in self.loadgen.model_fields_set
            and "request_count" in self.loadgen.model_fields_set
        ):
            raise ValueError(
                "Count-based and duration-based benchmarking cannot be used together. "
                "Use either --request-count or --benchmark-duration."
            )

        if (
            "benchmark_grace_period" in self.loadgen.model_fields_set
            and "benchmark_duration" not in self.loadgen.model_fields_set
        ):
            raise ValueError(
                "--benchmark-grace-period can only be used with "
                "duration-based benchmarking (--benchmark-duration)."
            )

        return self

    def get_effective_request_count(self) -> int:
        """Get the effective number of requests to send.

        For mooncake_trace custom datasets, always use the dataset size to ensure
        exact trace replay. For all other scenarios, use the configured request_count.

        Returns:
            int: The number of requests that should be sent
        """
        if self.input.custom_dataset_type == CustomDatasetType.MOONCAKE_TRACE:
            try:
                dataset_size = self._count_dataset_entries()
                if dataset_size > 0:
                    return dataset_size
                else:
                    raise ValueError("Empty mooncake_trace dataset file")
            except Exception as e:
                raise ValueError(
                    f"Could not read mooncake_trace dataset file: {e}"
                ) from e

        return self.loadgen.request_count

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

    gpu_telemetry: Annotated[
        list[str] | None,
        Field(
            default=None,
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
    ]

    _gpu_telemetry_mode: GPUTelemetryMode = GPUTelemetryMode.SUMMARY
    _gpu_telemetry_urls: list[str] = []
    _gpu_telemetry_metrics_file: Path | None = None

    @model_validator(mode="after")
    def _parse_gpu_telemetry_config(self) -> Self:
        """Parse gpu_telemetry list into mode, URLs, and metrics file."""
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
            from aiperf.common.logging import AIPerfLogger

            _logger = AIPerfLogger(__name__)
            _logger.info(
                f"Model name '{model_name}' cannot be used to create artifact "
                f"directory. Instead, '{filtered_name}' will be used."
            )
            model_name = filtered_name
        return model_name

    def _get_artifact_service_kind(self) -> str:
        """Get the service kind name based on the endpoint config."""
        # Lazy import to avoid circular dependency
        from aiperf.common.factories import EndpointFactory
        from aiperf.module_loader import ensure_modules_loaded

        ensure_modules_loaded()

        metadata = EndpointFactory.get_metadata(self.endpoint.type)
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
            "request_count" in self.loadgen.model_fields_set
            and "num" in self.input.conversation.model_fields_set
        ):
            raise ValueError(
                "Both a request-count and number of conversations are set. This can result in confusing output. "
                "Use only --conversation-num for multi-turn scenarios."
            )

        return self

    @model_validator(mode="after")
    def validate_concurrency_limits(self) -> Self:
        """Validate that concurrency does not exceed the appropriate limit."""
        if self.loadgen.concurrency is None:
            return self

        # For multi-turn scenarios, check against conversation_num
        if self.input.conversation.num is not None:
            if self.loadgen.concurrency > self.input.conversation.num:
                raise ValueError(
                    f"Concurrency ({self.loadgen.concurrency}) cannot be greater than "
                    f"the number of conversations ({self.input.conversation.num}). "
                    "Either reduce --concurrency or increase --conversation-num."
                )
        # For single-turn scenarios, check against request_count if it is set
        elif (
            "request_count" in self.loadgen.model_fields_set
            and self.loadgen.concurrency > self.loadgen.request_count
        ):
            raise ValueError(
                f"Concurrency ({self.loadgen.concurrency}) cannot be greater than "
                f"the request count ({self.loadgen.request_count}). "
                "Either reduce --concurrency or increase --request-count."
            )

        return self

    @model_validator(mode="after")
    def validate_user_context_requires_sessions(self) -> Self:
        """Validate that user context prompt requires num-sessions to be specified."""
        if (
            self.input.prompt.prefix_prompt.user_context_prompt_length is not None
            and self.input.conversation.num is None
        ):
            raise ValueError(
                "--user-context-prompt-length requires --num-sessions to be specified. "
                "Each session needs a unique user context prompt, so the number of sessions must be defined."
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
                "Cannot use both --shared-system-prompt-length/--user-context-prompt-length "
                "and --prefix-prompt-length/--prefix-prompt-pool-size. "
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
                f"Rankings-specific options (--rankings-passages-mean, --rankings-passages-stddev, "
                "--rankings-passages-prompt-token-mean, --rankings-passages-prompt-token-stddev, "
                "--rankings-query-prompt-token-mean, --rankings-query-prompt-token-stddev) "
                "can only be used with rankings endpoint types "
                f"Rankings endpoints: ({', '.join(rankings_endpoints)})."
            )

        # Validate that prompt tokens and rankings tokens are not both set
        if prompt_tokens_modified and (
            rankings_tokens_modified or endpoint_type_is_rankings
        ):
            raise ValueError(
                "The --prompt-input-tokens-mean/--prompt-input-tokens-stddev options "
                "cannot be used together with rankings-specific token options or the rankings endpoints"
                "Ranking options: (--rankings-passages-prompt-token-mean, --rankings-passages-prompt-token-stddev, "
                "--rankings-query-prompt-token-mean, --rankings-query-prompt-token-stddev, ). "
                f"Rankings endpoints: ({', '.join(rankings_endpoints)})."
                "Please use only one set of options."
            )
        return self
