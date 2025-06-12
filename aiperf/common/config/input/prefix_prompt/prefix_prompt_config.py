#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from pydantic import Field

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.config_defaults import PrefixPromptDefaults


class PrefixPromptConfig(BaseConfig):
    """
    A configuration class for defining prefix prompt related settings.
    """

    num: Annotated[
        int,
        Field(
            default=PrefixPromptDefaults.NUM,
            ge=0,
            description=(
                "The number of prefix prompts to select from.\n"
                "If this value is not zero, these are prompts that are prepended to input prompts.\n"
                "This is useful for benchmarking models that use a K-V cache."
            ),
        ),
    ]
    length: Annotated[
        int,
        Field(
            default=PrefixPromptDefaults.LENGTH,
            ge=0,
            description=(
                "The number of tokens in each prefix prompt.\n"
                'This is only used if "num" is greater than zero.\n'
                "Note that due to the prefix and user prompts being concatenated,\n"
                "the number of tokens in the final prompt may be off by one."
            ),
        ),
    ]
