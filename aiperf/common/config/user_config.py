# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import Annotated, Any

from pydantic import Field, model_validator
from typing_extensions import Self

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.cli_parameter import DisableCLI
from aiperf.common.config.config_validators import coerce_value
from aiperf.common.config.endpoint_config import EndpointConfig
from aiperf.common.config.input_config import InputConfig
from aiperf.common.config.loadgen_config import LoadGeneratorConfig
from aiperf.common.config.output_config import OutputConfig
from aiperf.common.config.tokenizer_config import TokenizerConfig
from aiperf.common.enums import RequestRateMode, TimingMode


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

    @property
    def timing_mode(self) -> TimingMode:
        """Get the timing mode based on the user config."""
        return self._timing_mode
