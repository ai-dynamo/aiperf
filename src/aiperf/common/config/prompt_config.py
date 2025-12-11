# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from pydantic import Field, model_validator
from typing_extensions import Self

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.cli_parameter import CLIParameter
from aiperf.common.config.config_defaults import (
    InputTokensDefaults,
    OutputTokensDefaults,
    PrefixPromptDefaults,
    PromptDefaults,
)
from aiperf.common.config.groups import Groups


class InputTokensConfig(BaseConfig):
    """
    A configuration class for defining input token related settings.
    """

    _CLI_GROUP = Groups.INPUT_SEQUENCE_LENGTH

    mean: Annotated[
        int,
        Field(
            ge=0,
            description="The mean of number of tokens in the generated prompts when using synthetic data.",
        ),
        CLIParameter(
            name=(
                "--prompt-input-tokens-mean",
                "--synthetic-input-tokens-mean",  # GenAI-Perf
                "--isl",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
        ),
    ] = InputTokensDefaults.MEAN

    stddev: Annotated[
        float,
        Field(
            ge=0,
            description="The standard deviation of number of tokens in the generated prompts when using synthetic data.",
        ),
        CLIParameter(
            name=(
                "--prompt-input-tokens-stddev",
                "--synthetic-input-tokens-stddev",  # GenAI-Perf
                "--isl-stddev",
            ),
            group=_CLI_GROUP,
        ),
    ] = InputTokensDefaults.STDDEV

    # NEW AIPerf Option
    block_size: Annotated[
        int,
        Field(
            default=512,
            description="The block size of the prompt.",
        ),
        CLIParameter(
            name=(
                "--prompt-input-tokens-block-size",
                "--synthetic-input-tokens-block-size",
                "--isl-block-size",
            ),
            group=_CLI_GROUP,
        ),
    ] = InputTokensDefaults.BLOCK_SIZE


class OutputTokensConfig(BaseConfig):
    """
    A configuration class for defining output token related settings.
    """

    _CLI_GROUP = Groups.OUTPUT_SEQUENCE_LENGTH

    mean: Annotated[
        int | None,
        Field(
            default=None,
            ge=0,
            description="The mean number of tokens in each output.",
        ),
        CLIParameter(
            name=(
                "--prompt-output-tokens-mean",
                "--output-tokens-mean",  # GenAI-Perf
                "--osl",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
        ),
    ] = None

    stddev: Annotated[
        float | None,
        Field(
            default=None,
            ge=0,
            description="The standard deviation of the number of tokens in each output.",
        ),
        CLIParameter(
            name=(
                "--prompt-output-tokens-stddev",
                "--output-tokens-stddev",  # GenAI-Perf
                "--osl-stddev",
            ),
            group=_CLI_GROUP,
        ),
    ] = OutputTokensDefaults.STDDEV


class PrefixPromptConfig(BaseConfig):
    """
    A configuration class for defining prefix prompt related settings.
    """

    _CLI_GROUP = Groups.PREFIX_PROMPT

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
        CLIParameter(
            name=(
                "--prompt-prefix-pool-size",
                "--prefix-prompt-pool-size",
                "--num-prefix-prompts",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
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
        CLIParameter(
            name=(
                "--prompt-prefix-length",
                "--prefix-prompt-length",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
        ),
    ] = PrefixPromptDefaults.LENGTH

    shared_system_prompt_length: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description=(
                "Length of shared system prompt in tokens.\n"
                "This prompt is identical across all sessions and appears as a system message.\n"
                "Mutually exclusive with --prefix-prompt-length/--prefix-prompt-pool-size."
            ),
        ),
        CLIParameter(
            name=("--shared-system-prompt-length",),
            group=_CLI_GROUP,
        ),
    ] = None

    user_context_prompt_length: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description=(
                "Length of per-session user context prompt in tokens.\n"
                "Each session gets a unique user context prompt.\n"
                "Requires --num-sessions to be specified.\n"
                "Mutually exclusive with --prefix-prompt-length/--prefix-prompt-pool-size."
            ),
        ),
        CLIParameter(
            name=("--user-context-prompt-length",),
            group=_CLI_GROUP,
        ),
    ] = None


class PromptConfig(BaseConfig):
    """
    A configuration class for defining prompt related settings.
    """

    _CLI_GROUP = Groups.PROMPT

    @model_validator(mode="after")
    def validate_sequence_distribution_format(self) -> Self:
        """Validate sequence distribution format and ensure percentages sum correctly."""
        if self.sequence_distribution is not None:
            try:
                from aiperf.common.models.sequence_distribution import (
                    DistributionParser,
                )

                # Only validate the format, don't create the full distribution yet
                # This avoids requiring RNG initialization during config validation
                DistributionParser.validate(self.sequence_distribution)
            except Exception as e:
                raise ValueError(f"Invalid sequence distribution format: {e}") from e
        return self

    batch_size: Annotated[
        int,
        Field(
            description="The batch size of text requests AIPerf should send.\n"
            "This is currently supported with the embeddings and rankings endpoint types",
        ),
        CLIParameter(
            name=(
                "--prompt-batch-size",
                "--batch-size-text",  # GenAI-Perf
                "--batch-size",  # GenAI-Perf
                "-b",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
        ),
    ] = PromptDefaults.BATCH_SIZE

    input_tokens: InputTokensConfig = InputTokensConfig()
    output_tokens: OutputTokensConfig = OutputTokensConfig()
    prefix_prompt: PrefixPromptConfig = PrefixPromptConfig()

    sequence_distribution: Annotated[
        str | None,
        Field(
            default=None,
            description="Sequence length distribution specification for varying ISL/OSL pairs",
        ),
        CLIParameter(
            name=("--seq-dist", "--sequence-distribution"),
            group=Groups.INPUT_SEQUENCE_LENGTH,
        ),
    ] = None

    def get_sequence_distribution(self):
        """Get sequence distribution object, returning None if not specified."""
        if self.sequence_distribution is not None:
            from aiperf.common.models.sequence_distribution import DistributionParser

            return DistributionParser.parse(self.sequence_distribution)
        return None
