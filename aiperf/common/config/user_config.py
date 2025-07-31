# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from cyclopts import Parameter
from pydantic import BeforeValidator, Field

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.config_validators import parse_str_or_list
from aiperf.common.config.endpoint_config import EndPointConfig
from aiperf.common.config.groups import Groups
from aiperf.common.config.input_config import InputConfig
from aiperf.common.config.loadgen_config import LoadGeneratorConfig
from aiperf.common.config.measurement_config import MeasurementConfig
from aiperf.common.config.output_config import OutputConfig
from aiperf.common.config.tokenizer_config import TokenizerConfig


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
        Parameter(
            name=(
                "--model-names",
                "--model",  # GenAI-Perf
                "-m",  # GenAI-Perf
            ),
            group=Groups.ENDPOINT,
        ),
    ]

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

    loadgen: Annotated[
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
