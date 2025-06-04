#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0


from typing import Annotated

from pydantic import AfterValidator, BeforeValidator, Field

from aiperf.common.config.base_config import ADD_TO_TEMPLATE, BaseConfig
from aiperf.common.config.config_defaults import UserDefaults
from aiperf.common.config.config_validators import (
    parse_str_or_list,
    print_str_or_list,
)
from aiperf.common.config.endpoint_config import EndPointConfig
from aiperf.common.config.input_config import InputConfig


class UserConfig(BaseConfig):
    """
    A configuration class for defining top-level user settings.
    """

    model_names: Annotated[
        str | list[str],
        Field(
            default=UserDefaults.MODEL_NAMES,
            description="Model name(s) to be benchmarked. Can be a comma-separated list or a single model name.",
        ),
        BeforeValidator(parse_str_or_list),
        AfterValidator(print_str_or_list),
    ]

    verbose: Annotated[
        bool,
        Field(
            default=UserDefaults.VERBOSE,
            description="Enable verbose output.",
            json_schema_extra={ADD_TO_TEMPLATE: False},
        ),
    ]

    template_filename: Annotated[
        str,
        Field(
            default=UserDefaults.TEMPLATE_FILENAME,
            description="Path to the template file.",
            json_schema_extra={ADD_TO_TEMPLATE: False},
        ),
    ]

    endpoint: EndPointConfig = EndPointConfig()
    input: InputConfig = InputConfig()
