#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from pydantic import Field

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.config_defaults import OutputTokensDefaults


class OutputTokensConfig(BaseConfig):
    """
    A configuration class for defining output token related settings.
    """

    mean: Annotated[
        int,
        Field(
            default=OutputTokensDefaults.MEAN,
            ge=0,
            description="The mean number of tokens in each output.",
        ),
    ]
    deterministic: Annotated[
        bool,
        Field(
            default=OutputTokensDefaults.DETERMINISTIC,
            description=(
                "This can be set to improve the precision of the mean by setting the\n"
                "minimum number of tokens equal to the requested number of tokens.\n"
                "This is currently supported with Triton."
            ),
        ),
    ]
    stddev: Annotated[
        float,
        Field(
            default=OutputTokensDefaults.STDDEV,
            ge=0,
            description="The standard deviation of the number of tokens in each output.",
        ),
    ]
