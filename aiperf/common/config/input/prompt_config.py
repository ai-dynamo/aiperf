# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

import cyclopts
from pydantic import Field

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.config_defaults import (
    InputTokensDefaults,
    OutputTokensDefaults,
    PrefixPromptDefaults,
    PromptDefaults,
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
        cyclopts.Parameter(
            name=(
                "--synthetic-input-tokens-mean",  # GenAI-Perf
                "--isl",  # GenAI-Perf
            ),
        ),
    ] = InputTokensDefaults.MEAN

    stddev: Annotated[
        float,
        Field(
            ge=0,
            description="The standard deviation of number of tokens in the generated prompts when using synthetic data.",
        ),
        cyclopts.Parameter(
            name=(
                "--synthetic-input-tokens-stddev",  # GenAI-Perf
            ),
        ),
    ] = InputTokensDefaults.STDDEV

    # NEW AIPerf Option
    block_size: Annotated[
        int,
        Field(
            default=512,
            description="The block size of the prompt.",
        ),
        cyclopts.Parameter(
            name=("--synthetic-input-tokens-block-size"),
        ),
    ] = InputTokensDefaults.BLOCK_SIZE


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
        cyclopts.Parameter(
            name=(
                "--output-tokens-mean",  # GenAI-Perf
                "--osl",  # GenAI-Perf
            ),
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
        cyclopts.Parameter(
            name=(
                "--output-tokens-mean-deterministic",  # GenAI-Perf
            ),
        ),
    ] = OutputTokensDefaults.DETERMINISTIC

    stddev: Annotated[
        float,
        Field(
            ge=0,
            description="The standard deviation of the number of tokens in each output.",
        ),
        cyclopts.Parameter(
            name=(
                "--output-tokens-stddev",  # GenAI-Perf
            ),
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
        cyclopts.Parameter(
            name=(
                "--prefix-prompt-pool-size",
                "--num-prefix-prompts",  # GenAI-Perf
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
        cyclopts.Parameter(
            name=(
                "--prefix-prompt-length",  # GenAI-Perf
            ),
        ),
    ] = PrefixPromptDefaults.LENGTH


class PromptConfig(BaseConfig):
    """
    A configuration class for defining prompt related settings.
    """

    batch_size: Annotated[
        int,
        Field(
            description="The batch size of text requests AIPerf should send.\n"
            "This is currently supported with the embeddings and rankings endpoint types",
        ),
        cyclopts.Parameter(
            name=(
                "--prompt-batch-size",
                "--batch-size-text",  # GenAI-Perf
                "--batch-size",  # GenAI-Perf
                "-b",  # GenAI-Perf
            ),
        ),
    ] = PromptDefaults.BATCH_SIZE

    input_tokens: InputTokensConfig = InputTokensConfig()
    output_tokens: OutputTokensConfig = OutputTokensConfig()
    prefix_prompt: PrefixPromptConfig = PrefixPromptConfig()
