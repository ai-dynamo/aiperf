# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

import cyclopts
from pydantic import BeforeValidator, Field

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.config_defaults import EndPointDefaults
from aiperf.common.config.config_validators import parse_str_or_list
from aiperf.common.enums import ModelSelectionStrategy, RequestPayloadType


class EndPointConfig(BaseConfig):
    """
    A configuration class for defining endpoint related settings.
    """

    model_selection_strategy: Annotated[
        ModelSelectionStrategy,
        Field(
            description="When multiple models are specified, this is how a specific model should be assigned to a prompt. \
            \nround_robin: nth prompt in the list gets assigned to n-mod len(models). \
            \nrandom: assignment is uniformly random",
        ),
        cyclopts.Parameter(
            name=("--model-selection-strategy"),
        ),
    ] = EndPointDefaults.MODEL_SELECTION_STRATEGY

    request_payload_type: Annotated[
        RequestPayloadType,
        Field(
            description="The type of request payload to send to the model.",
        ),
        cyclopts.Parameter(
            name=("--request-payload-type"),
        ),
    ] = EndPointDefaults.REQUEST_PAYLOAD_TYPE

    custom: Annotated[
        str,
        Field(
            description="Set a custom endpoint that differs from the OpenAI defaults.",
        ),
        cyclopts.Parameter(
            name=("--custom-endpoint"),
        ),
    ] = EndPointDefaults.CUSTOM

    type: Annotated[
        str,
        Field(
            description="The type to send requests to on the server.",
        ),
        cyclopts.Parameter(
            name=("--type"),
        ),
    ] = EndPointDefaults.TYPE

    streaming: Annotated[
        bool,
        Field(
            description="An option to enable the use of the streaming API.",
        ),
        cyclopts.Parameter(
            name=("--streaming"),
        ),
    ] = EndPointDefaults.STREAMING

    server_metrics_urls: Annotated[
        list[str],
        Field(
            description="The list of Triton server metrics URLs. \
            \nThese are used for Telemetry metric reporting with Triton.",
        ),
        BeforeValidator(parse_str_or_list),
        cyclopts.Parameter(
            name=("--server-metrics-urls"),
        ),
    ] = EndPointDefaults.SERVER_METRICS_URLS

    url: Annotated[
        str,
        Field(
            description="URL of the endpoint to target for benchmarking.",
        ),
        cyclopts.Parameter(
            name=("--url", "-u"),
        ),
    ] = EndPointDefaults.URL

    grpc_method: Annotated[
        str,
        Field(
            description="A fully-qualified gRPC method name in "
            "'<package>.<service>/<method>' format."
            "\nThe option is only supported by dynamic gRPC service kind and is"
            "\nrequired to identify the RPC to use when sending requests to the server.",
        ),
        cyclopts.Parameter(
            name=("--grpc-method"),
        ),
    ] = EndPointDefaults.GRPC_METHOD
