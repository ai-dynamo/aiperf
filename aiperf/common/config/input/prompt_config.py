#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from pydantic import Field

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.config_defaults import (
    InputTokensDefaults,
    OutputTokensDefaults,
    PrefixPromptDefaults,
)


class InputTokensConfig(BaseConfig):
    """
    A configuration class for defining input token related settings.
    """

    mean: Annotated[
        int,
        Field(
            ge=0,
            description="The mean of number of tokens in the generated prompts when using synthetic data.",
        ),
    ] = InputTokensDefaults.MEAN

    stddev: Annotated[
        float,
        Field(
            ge=0,
            description="The standard deviation of number of tokens in the generated prompts when using synthetic data.",
        ),
    ] = InputTokensDefaults.STDDEV


class OutputTokensConfig(BaseConfig):
    """
    A configuration class for defining output token related settings.
    """

    mean: Annotated[
        int,
        Field(
            ge=0,
            description="The mean number of tokens in each output.",
        ),
    ] = OutputTokensDefaults.MEAN

    deterministic: Annotated[
        bool,
        Field(
            description=(
                "This can be set to improve the precision of the mean by setting the\n"
                "minimum number of tokens equal to the requested number of tokens.\n"
                "This is currently supported with Triton."
            ),
        ),
    ] = OutputTokensDefaults.DETERMINISTIC

    stddev: Annotated[
        float,
        Field(
            ge=0,
            description="The standard deviation of the number of tokens in each output.",
        ),
    ] = OutputTokensDefaults.STDDEV


class PrefixPromptConfig(BaseConfig):
    """
    A configuration class for defining prefix prompt related settings.
    """

    pool_size: Annotated[
        int,
        Field(
            ge=0,
            description=(
                "The total size of the prefix prompt pool to select prefixes from.\n"
                "If this value is not zero, these are prompts that are prepended to input prompts.\n"
                "This is useful for benchmarking models that use a K-V cache."
            ),
        ),
    ] = PrefixPromptDefaults.POOL_SIZE

    length: Annotated[
        int,
        Field(
            ge=0,
            description=(
                "The number of tokens in each prefix prompt.\n"
                'This is only used if "num" is greater than zero.\n'
                "Note that due to the prefix and user prompts being concatenated,\n"
                "the number of tokens in the final prompt may be off by one."
            ),
        ),
    ] = PrefixPromptDefaults.LENGTH


class PromptConfig(BaseConfig):
    """
    A configuration class for defining prompt related settings.
    """

    input_tokens: InputTokensConfig = InputTokensConfig()
    output_tokens: OutputTokensConfig = OutputTokensConfig()
    prefix_prompt: PrefixPromptConfig = PrefixPromptConfig()
