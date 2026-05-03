# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Any, Literal

from cyclopts import Parameter
from pydantic import Field, field_validator, model_validator

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.cli_parameter import CLIParameter
from aiperf.common.config.config_defaults import LoadGeneratorDefaults
from aiperf.common.config.groups import Groups
from aiperf.common.enums import ConvergenceMode, ConvergenceStat
from aiperf.plugin.enums import ArrivalPattern


class LoadGeneratorConfig(BaseConfig):
    """A configuration class for defining top-level load generator settings."""

    _CLI_GROUP = Groups.LOAD_GENERATOR

    @field_validator("concurrency", mode="before")
    @classmethod
    def parse_concurrency_list(
        cls, v: int | str | list[int] | None
    ) -> int | list[int] | None:
        """Parse comma-separated concurrency values from CLI input.

        Converts comma-separated strings like "10,20,30" into lists [10, 20, 30].
        Single values like "10" or 10 remain as integers for backward compatibility.

        Args:
            v: Input value from CLI (can be int, str, list[int], or None)

        Returns:
            - None if input is None
            - int if input is a single integer value (>= 1)
            - list[int] if input is a comma-separated string or already a list (all >= 1)

        Raises:
            ValueError: If string contains non-integer values or values < 1
        """
        if v is None:
            return None

        # Already an int - validate and return
        if isinstance(v, int):
            if v < 1:
                raise ValueError(
                    f"Invalid concurrency value: {v}. "
                    f"Must be a positive integer (>= 1)."
                )
            return v

        # Already a list - validate all elements and return
        if isinstance(v, list):
            validated = []
            for item in v:
                try:
                    # Reject non-integer floats to prevent silent truncation
                    if isinstance(item, float) and not item.is_integer():
                        raise ValueError(
                            f"Invalid concurrency list element: '{item}'. "
                            f"All values must be positive integers (>= 1)."
                        )
                    val = int(item)
                    if val < 1:
                        raise ValueError(
                            f"Invalid concurrency value: {val}. "
                            f"All concurrency values must be at least 1."
                        )
                    validated.append(val)
                except (ValueError, TypeError) as err:
                    raise ValueError(
                        f"Invalid concurrency list element: '{item}'. "
                        f"All values must be positive integers (>= 1)."
                    ) from err
            return validated

        # String input - parse comma-separated values
        if isinstance(v, str):
            # Split by comma and strip whitespace
            parts = [part.strip() for part in v.split(",")]

            # Single value without comma - return as int
            if len(parts) == 1:
                try:
                    val = int(parts[0])
                    if val < 1:
                        raise ValueError(
                            f"Invalid concurrency value: {val}. "
                            f"Must be a positive integer (>= 1)."
                        )
                    return val
                except ValueError as err:
                    raise ValueError(
                        f"Invalid concurrency value: '{parts[0]}'. "
                        f"Must be a positive integer (>= 1). "
                        f"Examples: --concurrency 10, --concurrency 20"
                    ) from err

            # Multiple values - parse as list
            try:
                validated = []
                for part in parts:
                    val = int(part)
                    if val < 1:
                        raise ValueError(
                            f"Invalid concurrency value: {val}. "
                            f"All concurrency values must be at least 1."
                        )
                    validated.append(val)
                return validated
            except ValueError as err:
                # Try to identify which value failed
                invalid_value = None
                for part in parts:
                    try:
                        val = int(part)
                        if val < 1:
                            invalid_value = part
                            break
                    except ValueError:
                        invalid_value = part
                        break

                if invalid_value:
                    raise ValueError(
                        f"Invalid concurrency list: '{v}'. "
                        f"Failed to parse value: '{invalid_value}'. "
                        f"All values must be positive integers (>= 1). "
                        f"Examples: --concurrency 10,20,30 or --concurrency 5,10,15,20"
                    ) from err
                else:
                    raise ValueError(
                        f"Invalid concurrency list: '{v}'. "
                        f"All values must be positive integers (>= 1). "
                        f"Examples: --concurrency 10,20,30 or --concurrency 5,10,15,20"
                    ) from err

        raise ValueError(
            f"Internal error: Invalid concurrency type {type(v).__name__}. "
            f"This is a bug - please report it. "
            f"Expected int, str, list[int], or None."
        )

    benchmark_duration: Annotated[
        float | None,
        Field(
            gt=0,
            description="Maximum benchmark runtime in seconds. When set, AIPerf stops issuing new requests after this duration, "
            "Responses received within `--benchmark-grace-period` after duration ends are included in metrics.",
        ),
        CLIParameter(
            name=("--benchmark-duration",),
            group=_CLI_GROUP,
        ),
    ] = None

    benchmark_grace_period: Annotated[
        float,
        Field(
            ge=0,
            description="The grace period in seconds to wait for responses after benchmark duration ends. "
            "Only applies when --benchmark-duration is set. Responses received within this period "
            "are included in metrics. Use 'inf' to wait indefinitely for all responses.",
        ),
        CLIParameter(
            name=("--benchmark-grace-period",),
            group=_CLI_GROUP,
        ),
    ] = LoadGeneratorDefaults.BENCHMARK_GRACE_PERIOD

    concurrency: Annotated[
        Any,  # CLI accepts string, validator converts to Union[int, list[int], None]
        Field(
            description="Number of concurrent requests to maintain OR list of concurrency values for parameter sweep. "
            "AIPerf issues a new request immediately when one completes, maintaining this level of in-flight requests. "
            "Can be combined with `--request-rate` to control the request rate. "
            "When a list is provided (e.g., [10, 20, 30]), AIPerf runs benchmarks sequentially for each value.",
        ),
        CLIParameter(
            name=(
                "--concurrency",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
        ),
    ] = None

    parameter_sweep_mode: Annotated[
        Literal["repeated", "independent"],
        Field(
            description="Sweep execution mode: 'repeated' (default) runs full sweep N times, "
            "'independent' runs N trials at each sweep value"
        ),
        CLIParameter(
            name=("--parameter-sweep-mode",),
            group=Groups.PARAMETER_SWEEP,
        ),
    ] = "repeated"

    parameter_sweep_cooldown_seconds: Annotated[
        float,
        Field(
            ge=0,
            description="Cooldown duration between sweep values (seconds). "
            "Only applies when sweeping parameters (e.g., --concurrency 10,20,30). "
            "Allows the system to stabilize between different parameter values. "
            "Default is 0 (no cooldown).",
        ),
        CLIParameter(
            name=("--parameter-sweep-cooldown-seconds",),
            group=Groups.PARAMETER_SWEEP,
        ),
    ] = 0.0

    parameter_sweep_same_seed: Annotated[
        bool,
        Field(
            description="Use same random seed across all sweep values (default: derive different seeds). "
            "Only applies when sweeping parameters (e.g., --concurrency 10,20,30). "
            "When False (default), each sweep value uses a different derived seed (base_seed + sweep_index) "
            "to avoid artificial correlation between measurements. "
            "When True, all sweep values use the same base seed for correlated workload comparisons.",
        ),
        CLIParameter(
            name=("--parameter-sweep-same-seed",),
            group=Groups.PARAMETER_SWEEP,
        ),
    ] = False

    prefill_concurrency: Annotated[
        int | None,
        Field(
            ge=1,
            description="Max concurrent requests waiting for first token (prefill phase). "
            "Limits how many requests can be in the prefill/prompt-processing stage simultaneously.",
        ),
        CLIParameter(
            name=("--prefill-concurrency",),
            group=_CLI_GROUP,
        ),
    ] = None

    request_rate: Annotated[
        float | None,
        Field(
            gt=0,
            description="Target request rate in requests per second. AIPerf generates request timing according to `--request-rate-mode` "
            "to achieve this average rate. Can be combined with `--concurrency` to control the number of concurrent requests. "
            "Supports fractional rates (e.g., `0.5` = 1 request every 2 seconds).",
        ),
        CLIParameter(
            name=(
                "--request-rate",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
        ),
    ] = None

    arrival_pattern: Annotated[
        ArrivalPattern,
        Field(
            description="Sets the arrival pattern for the load generated by AIPerf. Valid values: constant, poisson, gamma.\n"
            "`constant`: Generate requests at a fixed rate.\n"
            "`poisson`: Generate requests using a poisson distribution.\n"
            "`gamma`: Generate requests using a gamma distribution with tunable smoothness."
        ),
        CLIParameter(
            name=("--arrival-pattern", "--request-rate-mode"),
            group=_CLI_GROUP,
        ),
    ] = LoadGeneratorDefaults.ARRIVAL_PATTERN

    arrival_smoothness: Annotated[
        float | None,
        Field(
            gt=0,
            description="Smoothness parameter for gamma distribution arrivals (--arrival-pattern gamma). "
            "Controls the shape of the arrival pattern:\n"
            "- 1.0: Poisson-like (exponential inter-arrivals, default)\n"
            "- <1.0: Bursty/clustered arrivals (higher variance)\n"
            "- >1.0: Smooth/regular arrivals (lower variance)\n"
            "Compatible with vLLM's --burstiness parameter (same value = same distribution).",
        ),
        CLIParameter(
            name=("--arrival-smoothness", "--vllm-burstiness"),
            group=_CLI_GROUP,
        ),
    ] = None

    request_count: Annotated[
        int | None,
        Field(
            ge=1,
            description="The maximum number of requests to send. If not set, will be automatically determined based "
            "on the timing mode and dataset size. For synthetic datasets, this will be `max(10, concurrency * 2)`.",
        ),
        CLIParameter(
            name=(
                "--request-count",  # GenAI-Perf
                "--num-requests",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
        ),
    ] = None

    warmup_request_count: Annotated[
        int | None,
        Field(
            gt=0,
            description="The maximum number of warmup requests to send before benchmarking. "
            "If not set and no --warmup-duration is set, then no warmup phase will be used.",
        ),
        CLIParameter(
            name=(
                "--warmup-request-count",  # GenAI-Perf
                "--num-warmup-requests",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
        ),
    ] = None

    warmup_duration: Annotated[
        float | None,
        Field(
            gt=0,
            description="The maximum duration in seconds for the warmup phase. If not set, it will use the `--warmup-request-count` value. "
            "If neither are set, no warmup phase will be used.",
        ),
        CLIParameter(
            name=("--warmup-duration",),
            group=_CLI_GROUP,
        ),
    ] = None

    warmup_num_sessions: Annotated[
        int | None,
        Field(
            ge=1,
            description="The number of sessions to use for the warmup phase. If not set, it will use the `--warmup-request-count` value.",
        ),
        CLIParameter(
            name=("--num-warmup-sessions",),
            group=_CLI_GROUP,
        ),
    ] = None

    warmup_concurrency: Annotated[
        int | None,
        Field(
            ge=1,
            description="The concurrency value to use for the warmup phase. If not set, it will use the `--concurrency` value.",
        ),
        CLIParameter(
            name=("--warmup-concurrency",),
            group=_CLI_GROUP,
        ),
    ] = None

    warmup_prefill_concurrency: Annotated[
        int | None,
        Field(
            ge=1,
            description="The prefill concurrency value to use for the warmup phase. "
            "If not set, it will use the `--prefill-concurrency` value.",
        ),
        CLIParameter(
            name=("--warmup-prefill-concurrency",),
            group=_CLI_GROUP,
        ),
    ] = None

    warmup_request_rate: Annotated[
        float | None,
        Field(
            gt=0,
            description="The request rate to use for the warmup phase. If not set, it will use the `--request-rate` value.",
        ),
        CLIParameter(
            name=("--warmup-request-rate",),
            group=_CLI_GROUP,
        ),
    ] = None

    warmup_arrival_pattern: Annotated[
        ArrivalPattern | None,
        Field(
            default=None,
            description="The arrival pattern to use for the warmup phase. "
            "If not set, it will use the `--arrival-pattern` value. "
            "Valid values: constant, poisson, gamma.",
        ),
        CLIParameter(
            name=("--warmup-arrival-pattern",),
            group=_CLI_GROUP,
            show_choices=False,
        ),
    ] = None

    warmup_grace_period: Annotated[
        float | None,
        Field(
            ge=0,
            description="The grace period in seconds to wait for responses after warmup phase ends. "
            "Only applies when warmup is enabled. Responses received within this period "
            "are included in warmup completion. If not set, waits indefinitely for all warmup responses.",
        ),
        CLIParameter(
            name=("--warmup-grace-period",),
            group=_CLI_GROUP,
        ),
    ] = None

    # TODO: We should add a warning for values below 1.0, to ensure the user is aware that the value is a percentage.
    request_cancellation_rate: Annotated[
        float | None,
        Field(
            gt=0.0,
            le=100.0,
            description="Percentage (0-100) of requests to cancel for testing cancellation handling. Cancelled requests are sent normally "
            "but aborted after `--request-cancellation-delay` seconds. Useful for testing graceful degradation and resource cleanup.",
        ),
        CLIParameter(
            name=("--request-cancellation-rate",),
            group=_CLI_GROUP,
        ),
    ] = None

    request_cancellation_delay: Annotated[
        float,
        Field(
            ge=0.0,
            description="Seconds to wait after the request is fully sent before cancelling. "
            "A delay of 0 means 'send the full request, then immediately disconnect'. "
            "Requires --request-cancellation-rate to be set.",
        ),
        CLIParameter(
            name=("--request-cancellation-delay",),
            group=_CLI_GROUP,
        ),
    ] = 0.0

    user_centric_rate: Annotated[
        float | None,
        Field(
            gt=0,
            description="Enable user-centric rate limiting mode with the specified request rate (QPS). "
            "Each user has a gap = num_users / qps between turns. "
            "Users block on their previous turn (no interleaving within a user). "
            "New users are spawned on a fixed schedule to maintain steady-state throughput. "
            "Designed for KV cache benchmarking with realistic multi-user patterns. "
            "Requires --num-users to be set.",
        ),
        CLIParameter(
            name=("--user-centric-rate",),
            group=_CLI_GROUP,
        ),
    ] = None

    num_users: Annotated[
        int | None,
        Field(
            ge=1,
            description="The number of initial users to use for --user-centric-rate mode.",
        ),
        CLIParameter(
            name=("--num-users",),
            group=_CLI_GROUP,
        ),
    ] = None

    concurrency_ramp_duration: Annotated[
        float | None,
        Field(
            gt=0,
            description="Duration in seconds to ramp session concurrency from 1 to target. "
            "Useful for gradual warm-up of the target system.",
        ),
        CLIParameter(
            name=("--concurrency-ramp-duration",),
            group=_CLI_GROUP,
        ),
    ] = None

    prefill_concurrency_ramp_duration: Annotated[
        float | None,
        Field(
            gt=0,
            description="Duration in seconds to ramp prefill concurrency from 1 to target.",
        ),
        CLIParameter(
            name=("--prefill-concurrency-ramp-duration",),
            group=_CLI_GROUP,
        ),
    ] = None

    warmup_concurrency_ramp_duration: Annotated[
        float | None,
        Field(
            gt=0,
            description="Duration in seconds to ramp warmup session concurrency from 1 to target. "
            "If not set, uses `--concurrency-ramp-duration` value.",
        ),
        CLIParameter(
            name=("--warmup-concurrency-ramp-duration",),
            group=_CLI_GROUP,
        ),
    ] = None

    warmup_prefill_concurrency_ramp_duration: Annotated[
        float | None,
        Field(
            gt=0,
            description="Duration in seconds to ramp warmup prefill concurrency from 1 to target. "
            "If not set, uses `--prefill-concurrency-ramp-duration` value.",
        ),
        CLIParameter(
            name=("--warmup-prefill-concurrency-ramp-duration",),
            group=_CLI_GROUP,
        ),
    ] = None

    request_rate_ramp_duration: Annotated[
        float | None,
        Field(
            gt=0,
            description="Duration in seconds to ramp request rate from a proportional minimum to target. "
            "Start rate is calculated as target * (update_interval / duration), ensuring correct "
            "behavior for target rates below 1 QPS. Useful for gradual warm-up of the target system.",
        ),
        CLIParameter(
            name=("--request-rate-ramp-duration",),
            group=_CLI_GROUP,
        ),
    ] = None

    warmup_request_rate_ramp_duration: Annotated[
        float | None,
        Field(
            gt=0,
            description="Duration in seconds to ramp warmup request rate from a proportional minimum to target. "
            "Start rate is calculated as target * (update_interval / duration). "
            "If not set, uses `--request-rate-ramp-duration` value.",
        ),
        CLIParameter(
            name=("--warmup-request-rate-ramp-duration",),
            group=_CLI_GROUP,
        ),
    ] = None

    # Upper limit of 10 runs balances statistical validity with practical considerations:
    # - Statistical: 10 samples provide reasonable confidence intervals (t-distribution)
    # - Practical: Limits total benchmark time (10 runs can take hours for long benchmarks)
    # - Diminishing returns: Confidence interval width decreases with sqrt(n), so gains
    #   beyond 10 runs are marginal compared to the additional time investment
    # - Resource efficiency: Reduces compute/GPU costs while maintaining statistical rigor
    num_profile_runs: Annotated[
        int,
        Field(
            ge=1,
            le=10,
            description="Number of profile runs to execute for confidence reporting. "
            "Must be between 1 and 10. "
            "When set to 1 (default), runs a single benchmark. "
            "When set to >1, runs multiple benchmarks and computes aggregate statistics "
            "(mean, std, confidence intervals, coefficient of variation) across runs. "
            "Useful for quantifying variance and establishing confidence in results.",
        ),
        CLIParameter(
            name=("--num-profile-runs",),
            group=Groups.MULTI_RUN,
        ),
    ] = 1
    profile_run_cooldown_seconds: Annotated[
        float,
        Field(
            ge=0,
            description="Cooldown duration in seconds between profile runs. "
            "Only applies when --num-profile-runs > 1. "
            "Allows the system to stabilize between runs (e.g., clear caches, cool down GPUs). "
            "Default is 0 (no cooldown).",
        ),
        CLIParameter(
            name=("--profile-run-cooldown-seconds",),
            group=Groups.MULTI_RUN,
        ),
    ] = 0.0

    confidence_level: Annotated[
        float,
        Field(
            gt=0,
            lt=1,
            description="Confidence level for computing confidence intervals (0-1). "
            "Only applies when --num-profile-runs > 1. "
            "Common values: 0.90 (90%), 0.95 (95%, default), 0.99 (99%). "
            "Higher values produce wider confidence intervals.",
        ),
        CLIParameter(
            name=("--confidence-level",),
            group=Groups.MULTI_RUN,
        ),
    ] = 0.95

    profile_run_disable_warmup_after_first: Annotated[
        bool,
        Field(
            description="Disable warmup for profile runs after the first. "
            "Only applies when --num-profile-runs > 1. "
            "When True (default), only the first run includes warmup, subsequent runs "
            "measure steady-state performance for more accurate aggregate statistics. "
            "When False, all runs include warmup (useful for long cooldown periods "
            "or when testing cold-start performance).",
        ),
        Parameter(
            name=("--profile-run-disable-warmup-after-first",),
            group=Groups.MULTI_RUN,
            show_env_var=False,
            negative="--no-profile-run-disable-warmup-after-first",
        ),
    ] = True

    set_consistent_seed: Annotated[
        bool,
        Field(
            description="Automatically set random seed for consistent workloads across runs. "
            "Only applies when --num-profile-runs > 1. "
            "When True (default), automatically sets --random-seed=42 if not specified, "
            "ensuring identical workloads across all runs for valid statistical comparison. "
            "When False, preserves None seed, resulting in different workloads per run "
            "(not recommended for confidence reporting as it produces invalid statistics). "
            "If --random-seed is explicitly set, that value is always used regardless of this setting.",
        ),
        Parameter(
            name=("--set-consistent-seed",),
            group=Groups.MULTI_RUN,
            show_env_var=False,
            negative="--no-set-consistent-seed",
        ),
    ] = True

    convergence_metric: Annotated[
        str | None,
        Field(
            description="Target metric name for adaptive convergence stopping. "
            "When set with --num-profile-runs > 1, enables adaptive mode that stops "
            "early once the metric stabilizes according to --convergence-mode. "
            "Uses --num-profile-runs as the maximum run cap. "
            "Example metrics: time_to_first_token, request_latency, inter_token_latency.",
        ),
        CLIParameter(
            name=("--convergence-metric",),
            group=Groups.MULTI_RUN,
        ),
    ] = None

    convergence_stat: Annotated[
        ConvergenceStat,
        Field(
            description="Statistic to evaluate for convergence when using ci_width or cv mode. "
            "Common values: avg, p50, p90, p95, p99. "
            "Only applies when --convergence-metric is set.",
        ),
        CLIParameter(
            name=("--convergence-stat",),
            group=Groups.MULTI_RUN,
        ),
    ] = ConvergenceStat.AVG

    convergence_threshold: Annotated[
        float,
        Field(
            gt=0,
            lt=1,
            description="Threshold for convergence detection. "
            "For ci_width mode: maximum CI width as a fraction of the mean (default 0.10 = 10%). "
            "For cv mode: maximum coefficient of variation (default 0.10 = 10%). "
            "For distribution mode: KS test p-value threshold (default 0.10). "
            "Only applies when --convergence-metric is set.",
        ),
        CLIParameter(
            name=("--convergence-threshold",),
            group=Groups.MULTI_RUN,
        ),
    ] = 0.10

    convergence_mode: Annotated[
        ConvergenceMode,
        Field(
            description="Statistical method for convergence detection. "
            "ci_width: Stop when Student's t confidence interval width relative to mean is below threshold. "
            "cv: Stop when coefficient of variation (std/mean) is below threshold. "
            "distribution: Stop when KS test p-value indicates latest run matches prior runs "
            "(requires --export-level records or --export-level raw; rejected with --export-level summary). "
            "Only applies when --convergence-metric is set.",
        ),
        CLIParameter(
            name=("--convergence-mode",),
            group=Groups.MULTI_RUN,
        ),
    ] = ConvergenceMode.CI_WIDTH

    def get_sweep_parameter(self) -> tuple[str, list] | None:
        """Detect which parameter is being swept (if any).

        This method checks all sweepable parameters to determine if any
        are configured as lists, which indicates a parameter sweep.

        Returns:
            Tuple of (parameter_name, values) if sweeping, None otherwise.
            For example: ("concurrency", [10, 20, 30])

        Note:
            Currently only concurrency supports sweep mode. Future parameters
            can be added here as they become sweepable.
        """
        # Check concurrency
        if isinstance(self.concurrency, list):
            return ("concurrency", self.concurrency)

        # Future: Add other sweepable parameters here
        # if isinstance(self.request_rate, list):
        #     return ("request_rate", self.request_rate)

        return None

    def disable_warmup(self) -> None:
        """Disable all warmup-related parameters.

        This method explicitly sets all warmup fields to None, ensuring
        that no warmup phase runs. This is the authoritative list of
        warmup fields - if a new warmup field is added, it MUST be
        added to this method.

        This design makes it explicit which fields are warmup-related
        and ensures the list is maintained in one place (the config class).
        """
        # Core warmup parameters
        self.warmup_request_count = None
        self.warmup_duration = None
        self.warmup_num_sessions = None

        # Warmup load parameters
        self.warmup_concurrency = None
        self.warmup_prefill_concurrency = None
        self.warmup_request_rate = None
        self.warmup_arrival_pattern = None

        # Warmup timing parameters
        self.warmup_grace_period = None

        # Warmup ramp parameters
        self.warmup_concurrency_ramp_duration = None
        self.warmup_prefill_concurrency_ramp_duration = None
        self.warmup_request_rate_ramp_duration = None

    @model_validator(mode="after")
    def validate_concurrency_list(self) -> "LoadGeneratorConfig":
        """Validate that concurrency values are all >= 1 and lists have at least 2 elements.

        Raises:
            ValueError: If concurrency is < 1 (single value), any list value is < 1,
                       or list has fewer than 2 elements.
        """
        if isinstance(self.concurrency, int):
            if self.concurrency < 1:
                raise ValueError(
                    f"Invalid concurrency value: {self.concurrency}. "
                    f"Concurrency must be at least 1 (cannot have zero or negative concurrent requests). "
                    f"Use --concurrency 1 or higher."
                )
        elif isinstance(self.concurrency, list):
            # Check minimum list length for parameter sweep
            if len(self.concurrency) < 2:
                raise ValueError(
                    f"Invalid concurrency list: {self.concurrency}. "
                    f"Parameter sweep requires at least 2 values to compare. "
                    f"For a single concurrency value, use --concurrency {self.concurrency[0] if self.concurrency else 1} (without comma). "
                    f"For parameter sweep, provide multiple values: --concurrency 10,20,30"
                )

            # Check for duplicate values
            if len(set(self.concurrency)) < len(self.concurrency):
                duplicates = sorted(
                    {v for v in self.concurrency if self.concurrency.count(v) > 1}
                )
                raise ValueError(
                    f"Invalid concurrency list: {self.concurrency}. "
                    f"Duplicate values would overwrite each other's artifacts: {duplicates}. "
                    f"For variance / confidence reporting at a single concurrency, use "
                    f"--num-profile-runs N instead of repeating the value."
                )

            # Check all values are >= 1
            invalid_values = [v for v in self.concurrency if v < 1]
            if invalid_values:
                # Build helpful message showing positions
                positions = [i for i, v in enumerate(self.concurrency, 1) if v < 1]
                # Create a suggested fix
                fixed_list = ",".join(str(max(1, v)) for v in self.concurrency)
                raise ValueError(
                    f"Invalid concurrency values at position(s) {positions}: {invalid_values}. "
                    f"All concurrency values must be at least 1 (cannot have zero or negative concurrent requests). "
                    f"Current list: {self.concurrency}. "
                    f"Example fix: --concurrency {fixed_list}"
                )
        return self

    @model_validator(mode="after")
    def validate_multi_run_params(self) -> "LoadGeneratorConfig":
        """Validate that multi-run specific parameters are only set when num_profile_runs > 1.

        Raises:
            ValueError: If confidence_level, profile_run_disable_warmup_after_first,
                       profile_run_cooldown_seconds, or set_consistent_seed are explicitly
                       set when num_profile_runs == 1.
        """
        # Validate convergence_stat is not used with distribution mode
        # (distribution operates on per-request distributions, not summary stats)
        if (
            "convergence_stat" in self.model_fields_set
            and self.convergence_mode == ConvergenceMode.DISTRIBUTION
        ):
            raise ValueError(
                "--convergence-stat is not applicable with --convergence-mode distribution. "
                "Distribution mode uses per-request data directly and ignores the stat parameter. "
                "Remove --convergence-stat or use --convergence-mode ci_width or cv."
            )

        # Require --convergence-metric when any other convergence flag is explicitly set
        convergence_dependent_flags = {
            "convergence_mode": "--convergence-mode",
            "convergence_threshold": "--convergence-threshold",
            "convergence_stat": "--convergence-stat",
        }
        if self.convergence_metric is None:
            for field_name, flag_name in convergence_dependent_flags.items():
                if field_name in self.model_fields_set:
                    raise ValueError(
                        f"{flag_name} requires --convergence-metric to be set. "
                        f"Either add --convergence-metric or remove {flag_name}."
                    )

        if self.num_profile_runs == 1:
            # Check if confidence_level was explicitly set by the user
            if "confidence_level" in self.model_fields_set:
                raise ValueError(
                    "--confidence-level only applies when running multiple trials (--num-profile-runs > 1). "
                    "Confidence intervals require at least 2 runs to compute. "
                    "Either remove --confidence-level or add --num-profile-runs 5 (or higher)."
                )

            # Check if profile_run_disable_warmup_after_first was explicitly set by the user
            if "profile_run_disable_warmup_after_first" in self.model_fields_set:
                raise ValueError(
                    "--profile-run-disable-warmup-after-first only applies when running multiple trials (--num-profile-runs > 1). "
                    "This parameter controls whether warmup runs after the first trial. "
                    "Either remove --profile-run-disable-warmup-after-first or add --num-profile-runs 5 (or higher)."
                )

            # Check if profile_run_cooldown_seconds was explicitly set by the user
            if "profile_run_cooldown_seconds" in self.model_fields_set:
                raise ValueError(
                    "--profile-run-cooldown-seconds only applies when running multiple trials (--num-profile-runs > 1). "
                    "This parameter adds a pause between trials to let the system stabilize. "
                    "Either remove --profile-run-cooldown-seconds or add --num-profile-runs 5 (or higher)."
                )

            # Check if set_consistent_seed was explicitly set by the user
            if "set_consistent_seed" in self.model_fields_set:
                raise ValueError(
                    "--set-consistent-seed only applies when running multiple trials (--num-profile-runs > 1). "
                    "This parameter ensures identical workloads across trials for valid statistical comparison. "
                    "Either remove --set-consistent-seed or add --num-profile-runs 5 (or higher)."
                )

            # Check if convergence_metric was explicitly set by the user
            if "convergence_metric" in self.model_fields_set:
                raise ValueError(
                    "--convergence-metric only applies when --num-profile-runs > 1. "
                    "Remove --convergence-metric or increase --num-profile-runs."
                )

            # Check if other convergence flags were explicitly set
            convergence_dependent_flags = {
                "convergence_mode": "--convergence-mode",
                "convergence_threshold": "--convergence-threshold",
                "convergence_stat": "--convergence-stat",
            }
            for field_name, flag_name in convergence_dependent_flags.items():
                if field_name in self.model_fields_set:
                    raise ValueError(
                        f"{flag_name} only applies when --num-profile-runs > 1. "
                        f"Remove {flag_name} or increase --num-profile-runs."
                    )

        return self

    @model_validator(mode="after")
    def validate_sweep_params(self) -> "LoadGeneratorConfig":
        """Validate that parameter sweep specific parameters are only set when sweeping.

        Raises:
            ValueError: If parameter_sweep_mode, parameter_sweep_cooldown_seconds,
                       or parameter_sweep_same_seed are explicitly set when not sweeping.
        """
        is_sweep = isinstance(self.concurrency, list)

        if not is_sweep:
            # Check if parameter_sweep_mode was explicitly set by the user
            if "parameter_sweep_mode" in self.model_fields_set:
                raise ValueError(
                    "--parameter-sweep-mode only applies when sweeping parameters (e.g., --concurrency 10,20,30). "
                    "This parameter controls whether to run the full sweep repeatedly (repeated mode) "
                    "or run all trials at each value independently (independent mode). "
                    "Either remove --parameter-sweep-mode or provide a comma-separated list: --concurrency 10,20,30"
                )

            # Check if parameter_sweep_cooldown_seconds was explicitly set by the user
            if "parameter_sweep_cooldown_seconds" in self.model_fields_set:
                raise ValueError(
                    "--parameter-sweep-cooldown-seconds only applies when sweeping parameters (e.g., --concurrency 10,20,30). "
                    "This parameter adds a pause between different parameter values to let the system stabilize. "
                    "Either remove --parameter-sweep-cooldown-seconds or provide a comma-separated list: --concurrency 10,20,30"
                )

            # Check if parameter_sweep_same_seed was explicitly set by the user
            if "parameter_sweep_same_seed" in self.model_fields_set:
                raise ValueError(
                    "--parameter-sweep-same-seed only applies when sweeping parameters (e.g., --concurrency 10,20,30). "
                    "This parameter controls whether all sweep values use the same random seed for correlated workload comparisons. "
                    "Either remove --parameter-sweep-same-seed or provide a comma-separated list: --concurrency 10,20,30"
                )

        return self
