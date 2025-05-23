#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from typing import Annotated, List, Any

from pydantic import Field, BeforeValidator

from aiperf.common.models.config.base_config import BaseConfig
from aiperf.common.models.config.config_defaults import (
    EndPointDefaults,
    OutputFormat,
    ModelSelectionStrategy,
)

from aiperf.common.models.config.config_validators import parse_str_or_list


class EndPointConfig(BaseConfig):
    """
    A configuration class for defining endpoint related settings.
    """

    model_selection_strategy: Annotated[
        ModelSelectionStrategy,
        Field(
            default=EndPointDefaults.MODEL_SELECTION_STRATEGY,
            description="When multiple models are specified, this is how a specific model should be assigned to a prompt. \
            \nround_robin: nth prompt in the list gets assigned to n-mod len(models). \
            \nrandom: assignment is uniformly random",
        ),
    ]

    backend: Annotated[
        OutputFormat,
        Field(
            default=EndPointDefaults.BACKEND,
            description="When benchmarking Triton, this is the backend of the model.",
        ),
    ]

    custom: Annotated[
        str,
        Field(
            default=EndPointDefaults.CUSTOM,
            description="Set a custom endpoint that differs from the OpenAI defaults.",
        ),
    ]

    type: Annotated[
        str,
        Field(
            default=EndPointDefaults.TYPE,
            description="The type to send requests to on the server.",
        ),
    ]

    streaming: Annotated[
        bool,
        Field(
            default=EndPointDefaults.STREAMING,
            description="An option to enable the use of the streaming API.",
        ),
    ]

    server_metrics_urls: Annotated[
        List[str],
        Field(
            default=EndPointDefaults.SERVER_METRICS_URLS,
            description="The list of Triton server metrics URLs. \
            \nThese are used for Telemetry metric reporting with Triton.",
        ),
        BeforeValidator(parse_str_or_list),
    ]

    url: Annotated[
        str,
        Field(
            default=EndPointDefaults.URL,
            description="URL of the endpoint to target for benchmarking.",
        ),
    ]

    grpc_method: Annotated[
        str,
        Field(
            default=EndPointDefaults.GRPC_METHOD,
            description="A fully-qualified gRPC method name in "
            "'<package>.<service>/<method>' format."
            "\nThe option is only supported by dynamic gRPC service kind and is"
            "\nrequired to identify the RPC to use when sending requests to the server.",
        ),
    ]
