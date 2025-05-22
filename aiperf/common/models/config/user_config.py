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
from aiperf.common.models.config.config_defaults import UserDefaults
from aiperf.common.models.config.endpoint_config import EndPointConfig


def parse_model_names(model_names: Any) -> None:
    if type(model_names) is str:
        model_names = [model_name.strip() for model_name in model_names.split(",")]
    elif type(model_names) is list:
        model_names = model_names
    else:
        raise ValueError("User Config: model_names must be a string or list")

    return model_names


class UserConfig(BaseConfig):
    """
    UserConfig is a Pydantic model that represents the user configuration for the application.
    It includes fields for model names, batch size, and other parameters.
    """

    model_names: Annotated[
        List[str],
        Field(
            default=UserDefaults.MODEL_NAMES,
            description="Model name(s) to be benchmarked. Can be a comma-separated list or a single model name.",
        ),
        BeforeValidator(parse_model_names),
    ]

    verbose: Annotated[
        bool,
        Field(
            default=UserDefaults.VERBOSE,
            description="Enable verbose output.",
            json_schema_extra={"add_to_template": False},
        ),
    ]

    template_filename: Annotated[
        str,
        Field(
            default=UserDefaults.TEMPLATE_FILENAME,
            description="Path to the template file.",
            json_schema_extra={"add_to_template": False},
        ),
    ]

    endpoint: EndPointConfig = EndPointConfig()
