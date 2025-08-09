# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from cyclopts import Parameter
from pydantic import BeforeValidator, Field, model_validator
from typing_extensions import Self

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.config_defaults import EndpointDefaults
from aiperf.common.config.config_validators import parse_str_or_list
from aiperf.common.config.groups import Groups
from aiperf.common.enums import EndpointType, ModelSelectionStrategy

_logger = AIPerfLogger(__name__)


class EndpointConfig(BaseConfig):
    """
    A configuration class for defining endpoint related settings.
    """

    _CLI_GROUP = Groups.ENDPOINT

    @model_validator(mode="after")
    def validate_streaming(self) -> Self:
        if not self.type.supports_streaming:
            _logger.warning(
                f"Streaming is not supported for --endpoint-type {self.type}, setting streaming to False"
            )
            self.streaming = False
        return self

    model_names: Annotated[
        list[str],
        Field(
            ...,  # This must be set by the user
            description="Model name(s) to be benchmarked. Can be a comma-separated list or a single model name.",
        ),
        BeforeValidator(parse_str_or_list),
        Parameter(
            name=(
                "--model-names",
                "--model",  # GenAI-Perf
                "-m",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
        ),
    ]

    model_selection_strategy: Annotated[
        ModelSelectionStrategy,
        Field(
            description="When multiple models are specified, this is how a specific model should be assigned to a prompt.\n"
            "round_robin: nth prompt in the list gets assigned to n-mod len(models).\n"
            "random: assignment is uniformly random",
        ),
        Parameter(
            name=(
                "--model-selection-strategy",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
        ),
    ] = EndpointDefaults.MODEL_SELECTION_STRATEGY

    custom_endpoint: Annotated[
        str | None,
        Field(
            description="Set a custom endpoint that differs from the OpenAI defaults.",
        ),
        Parameter(
            name=(
                "--custom-endpoint",
                "--endpoint",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
        ),
    ] = EndpointDefaults.CUSTOM_ENDPOINT

    type: Annotated[
        EndpointType,
        Field(
            description="The type to send requests to on the server.",
        ),
        Parameter(
            name=(
                "--endpoint-type",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
        ),
    ] = EndpointDefaults.TYPE

    streaming: Annotated[
        bool,
        Field(
            description="An option to enable the use of the streaming API.",
        ),
        Parameter(
            name=(
                "--streaming",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
        ),
    ] = EndpointDefaults.STREAMING

    url: Annotated[
        str,
        Field(
            description="URL of the endpoint to target for benchmarking.",
        ),
        Parameter(
            name=(
                "--url",  # GenAI-Perf
                "-u",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
        ),
    ] = EndpointDefaults.URL

    # NEW AIPerf Option
    timeout_seconds: Annotated[
        float,
        Field(
            description="The timeout in floating points seconds for each request to the endpoint.",
        ),
        Parameter(
            name=("--request-timeout-seconds"),
            group=_CLI_GROUP,
        ),
    ] = EndpointDefaults.TIMEOUT

    # NEW AIPerf Option
    api_key: Annotated[
        str | None,
        Field(
            description="The API key to use for the endpoint. If provided, it will be sent with every request as "
            "a header: `Authorization: Bearer <api_key>`.",
        ),
        Parameter(
            name=("--api-key"),
            group=_CLI_GROUP,
        ),
    ] = EndpointDefaults.API_KEY
