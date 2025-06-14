#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from pydantic import Field

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.config_defaults import SyntheticTokensDefaults


class SyntheticTokensConfig(BaseConfig):
    """
    A configuration class for defining synthetic token related settings.
    """

    mean: Annotated[
        int,
        Field(
            default=SyntheticTokensDefaults.MEAN,
            ge=0,
            description="The mean of number of tokens in the generated prompts when using synthetic data.",
        ),
    ]

    stddev: Annotated[
        float,
        Field(
            default=SyntheticTokensDefaults.STDDEV,
            ge=0,
            description="The standard deviation of number of tokens in the generated prompts when using synthetic data.",
        ),
    ]
