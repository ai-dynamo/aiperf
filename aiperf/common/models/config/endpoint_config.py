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

from typing import Annotated, List, Any, ClassVar

from pydantic import BaseModel, Field, BeforeValidator, model_serializer

from aiperf.common.models.config.base_config import BaseConfig
from aiperf.common.models.config.config_defaults import EndPointDefaults, UserDefaults


class EndPointConfig(BaseConfig):
    backend: Annotated[
        str,
        Field(
            default=EndPointDefaults.BACKEND,
            description="When benchmarking Triton, this is the backend of the model.",
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
    ]
