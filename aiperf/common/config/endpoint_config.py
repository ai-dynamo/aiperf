# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

import cyclopts
from pydantic import BeforeValidator, Field

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.config_defaults import EndPointDefaults
from aiperf.common.config.config_validators import parse_str_or_list
from aiperf.common.enums import EndpointType, ModelSelectionStrategy


class EndPointConfig(BaseConfig):
    """
    A configuration class for defining endpoint related settings.
    """

    _GROUP_NAME = "Endpoint"

    model_selection_strategy: Annotated[
        ModelSelectionStrategy,
        Field(
            description="When multiple models are specified, this is how a specific model should be assigned to a prompt.\n"
            "round_robin: nth prompt in the list gets assigned to n-mod len(models).\n"
            "random: assignment is uniformly random",
        ),
        cyclopts.Parameter(
            name=(
                "--model-selection-strategy",  # GenAI-Perf
            ),
            group=_GROUP_NAME,
        ),
    ] = EndPointDefaults.MODEL_SELECTION_STRATEGY

    custom_endpoint: Annotated[
        str | None,
        Field(
            description="Set a custom endpoint that differs from the OpenAI defaults.",
        ),
        cyclopts.Parameter(
            name=(
                "--custom-endpoint",
                "--endpoint",  # GenAI-Perf
            ),
            group=_GROUP_NAME,
        ),
    ] = EndPointDefaults.CUSTOM_ENDPOINT

    type: Annotated[
        EndpointType,
        Field(
            description="The type to send requests to on the server.",
        ),
        cyclopts.Parameter(
            name=(
                "--endpoint-type",  # GenAI-Perf
            ),
            group=_GROUP_NAME,
        ),
    ] = EndPointDefaults.TYPE

    streaming: Annotated[
        bool,
        Field(
            description="An option to enable the use of the streaming API.",
        ),
        cyclopts.Parameter(
            name=(
                "--streaming",  # GenAI-Perf
            ),
            group=_GROUP_NAME,
        ),
    ] = EndPointDefaults.STREAMING

    server_metrics_urls: Annotated[
        list[str],
        Field(
            description="The list of Triton server metrics URLs.\n"
            "These are used for Telemetry metric reporting with Triton.",
        ),
        BeforeValidator(parse_str_or_list),
        cyclopts.Parameter(
            name=(
                "--server-metrics-urls",  # GenAI-Perf
                "--server-metrics-url",  # GenAI-Perf
            ),
            group=_GROUP_NAME,
        ),
    ] = EndPointDefaults.SERVER_METRICS_URLS

    url: Annotated[
        str,
        Field(
            description="URL of the endpoint to target for benchmarking.",
        ),
        cyclopts.Parameter(
            name=(
                "--url",  # GenAI-Perf
                "-u",  # GenAI-Perf
            ),
            group=_GROUP_NAME,
        ),
    ] = EndPointDefaults.URL

    grpc_method: Annotated[
        str,
        Field(
            description="A fully-qualified gRPC method name in "
            "'<package>.<service>/<method>' format.\n"
            "The option is only supported by dynamic gRPC service kind and is\n"
            "required to identify the RPC to use when sending requests to the server.",
        ),
        cyclopts.Parameter(
            name=(
                "--grpc-method",  # GenAI-Perf
            ),
            group=_GROUP_NAME,
        ),
    ] = EndPointDefaults.GRPC_METHOD

    # NEW AIPerf Option
    timeout_seconds: Annotated[
        float,
        Field(
            description="The timeout in floating points seconds for each request to the endpoint.",
        ),
        cyclopts.Parameter(
            name=("--request-timeout-seconds"),
            group=_GROUP_NAME,
        ),
    ] = EndPointDefaults.TIMEOUT

    # NEW AIPerf Option
    api_key: Annotated[
        str | None,
        Field(
            description="The API key to use for the endpoint. If provided, it will be sent with every request as"
            "as a header: `Authorization: Bearer <api_key>`.",
        ),
        cyclopts.Parameter(
            name=("--api-key"),
            group=_GROUP_NAME,
        ),
    ] = EndPointDefaults.API_KEY
