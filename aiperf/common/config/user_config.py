# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Annotated

from pydantic import Field, model_validator
from typing_extensions import Self

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.endpoint_config import EndpointConfig
from aiperf.common.config.input_config import InputConfig
from aiperf.common.config.loadgen_config import LoadGeneratorConfig
from aiperf.common.config.output_config import OutputConfig
from aiperf.common.config.tokenizer_config import TokenizerConfig
from aiperf.common.enums.endpoints_enums import EndpointServiceKind
from aiperf.common.enums.timing_enums import RequestRateMode, TimingMode


class UserConfig(BaseConfig):
    """
    A configuration class for defining top-level user settings.
    """

    _timing_mode: TimingMode = TimingMode.REQUEST_RATE

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
        if self.endpoint.type.service_kind == EndpointServiceKind.OPENAI:
            return f"{self.endpoint.type.service_kind}-{self.endpoint.type}"
        else:
            raise ValueError(
                f"Unknown service kind '{self.endpoint.type.service_kind}'."
            )

    def _get_artifact_stimulus(self) -> str:
        """Get the stimulus name based on the timing mode."""
        match self._timing_mode:
            case TimingMode.REQUEST_RATE:
                return f"request_rate{self.loadgen.request_rate}"
            case TimingMode.FIXED_SCHEDULE:
                return "fixed_schedule"
            case _:
                raise ValueError(f"Unknown timing mode '{self._timing_mode}'.")
