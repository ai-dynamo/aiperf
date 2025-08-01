# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from cyclopts import Parameter
from pydantic import BeforeValidator, Field

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.config_defaults import EndpointDefaults
from aiperf.common.config.config_validators import parse_str_or_list
from aiperf.common.config.groups import Groups
from aiperf.common.enums import EndpointType, ModelSelectionStrategy


class EndpointConfig(BaseConfig):
    """
    A configuration class for defining endpoint related settings.
    """

    _CLI_GROUP = Groups.ENDPOINT

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
            parse=False,  # TODO: Not yet supported
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

    server_metrics_urls: Annotated[
        list[str],
        Field(
            description="The list of Triton server metrics URLs.\n"
            "These are used for Telemetry metric reporting with Triton.",
        ),
        BeforeValidator(parse_str_or_list),
        Parameter(
            name=(
                "--server-metrics-urls",  # GenAI-Perf
                "--server-metrics-url",  # GenAI-Perf
            ),
            parse=False,  # TODO: Not yet supported
            group=_CLI_GROUP,
        ),
    ] = EndpointDefaults.SERVER_METRICS_URLS

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

    grpc_method: Annotated[
        str,
        Field(
            description="A fully-qualified gRPC method name in "
            "'<package>.<service>/<method>' format.\n"
            "The option is only supported by dynamic gRPC service kind and is\n"
            "required to identify the RPC to use when sending requests to the server.",
        ),
        Parameter(
            name=(
                "--grpc-method",  # GenAI-Perf
            ),
            parse=False,  # TODO: Not yet supported
            group=_CLI_GROUP,
        ),
    ] = EndpointDefaults.GRPC_METHOD

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
