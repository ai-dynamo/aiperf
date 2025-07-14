# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

import cyclopts
from pydantic import BeforeValidator, Field

from aiperf.common.config.base_config import ADD_TO_TEMPLATE, BaseConfig
from aiperf.common.config.config_defaults import UserDefaults
from aiperf.common.config.config_validators import (
    parse_str_or_list,
)
from aiperf.common.config.endpoint.endpoint_config import EndPointConfig
from aiperf.common.config.input.input_config import InputConfig
from aiperf.common.config.loadgen_config import LoadGeneratorConfig
from aiperf.common.config.measurement_config import MeasurementConfig
from aiperf.common.config.output.output_config import OutputConfig
from aiperf.common.config.tokenizer.tokenizer_config import TokenizerConfig


class UserConfig(BaseConfig):
    """
    A configuration class for defining top-level user settings.
    """

    model_names: Annotated[
        list[str],
        Field(
            ...,
            description="Model name(s) to be benchmarked. Can be a comma-separated list or a single model name.",
        ),
        BeforeValidator(parse_str_or_list),
        cyclopts.Parameter(
            name=("--model-names", "--model", "-m"),
        ),
    ]

    # TODO:: Should we move the verbose and template_filename to their own CLI config class?

    verbose: Annotated[
        bool,
        Field(
            description="Enable verbose output.",
            json_schema_extra={ADD_TO_TEMPLATE: False},
        ),
        cyclopts.Parameter(
            name=("--verbose", "-v"),
        ),
    ] = UserDefaults.VERBOSE

    template_filename: Annotated[
        str,
        Field(
            description="Path to the template file.",
            json_schema_extra={ADD_TO_TEMPLATE: False},
        ),
        cyclopts.Parameter(
            name=("--template-filename", "-t"),
        ),
    ] = UserDefaults.TEMPLATE_FILENAME

    endpoint: Annotated[
        EndPointConfig,
        Field(
            description="Endpoint configuration",
        ),
    ] = EndPointConfig()

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

    load: Annotated[
        LoadGeneratorConfig,
        Field(
            description="Load Generator configuration",
        ),
    ] = LoadGeneratorConfig()

    measurement: Annotated[
        MeasurementConfig,
        Field(
            description="Measurement configuration",
        ),
    ] = MeasurementConfig()
