# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
            description="Mean number of tokens for synthetically generated input prompts. AIPerf generates prompts with lengths "
            "following a normal distribution around this mean (±`--prompt-input-tokens-stddev`). Applies only to synthetic datasets, "
            "not custom or public datasets.",
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
            description="Standard deviation for synthetic input prompt token lengths. Creates variability in prompt sizes when > 0, "
            "simulating realistic workloads with mixed request sizes. Lengths follow normal distribution. "
            "Set to 0 for uniform prompt lengths. Applies only to synthetic data generation.",
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

    block_size: Annotated[
        int,
        Field(
            default=512,
            description="Token block size for hash-based prompt caching in `mooncake_trace` datasets. When `hash_ids` are provided in trace entries, "
            "prompts are divided into blocks of this size. Each `hash_id` maps to a cached block of `block_size` tokens, enabling simulation "
            "of KV-cache sharing patterns from production workloads. The total prompt length equals `(num_hash_ids - 1) * block_size + final_block_size`.",
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
            description="Mean number of tokens to request in model outputs via `max_completion_tokens` field. "
            "Controls response length for synthetic and some custom datasets. If specified, included in request payload to limit "
            "generation length. When not set, model determines output length.",
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
            description="Standard deviation for output token length requests. Creates variability in `max_completion_tokens` field across requests, "
            "simulating mixed response length requirements. Lengths follow normal distribution. "
            "Only applies when `--prompt-output-tokens-mean` is set.",
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
            description="Number of distinct prefix prompts to generate for K-V cache testing. Each prefix is prepended to user prompts, "
            "simulating cached context scenarios. Prefixes randomly selected from pool per request. Set to 0 to disable prefix prompts. "
            "Mutually exclusive with `--shared-system-prompt-length`/`--user-context-prompt-length`.",
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
                "This is only used if `--num-prefix-prompts` is greater than zero.\n"
                "Note that due to the prefix and user prompts being concatenated,\n"
                "the number of tokens in the final prompt may be off by one."
                "Mutually exclusive with `--shared-system-prompt-length`/`--user-context-prompt-length`."
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
                "Mutually exclusive with `--prefix-prompt-length`/`--prefix-prompt-pool-size`."
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
                "Each dataset entry gets a unique user context prompt.\n"
                "Requires --num-dataset-entries to be specified.\n"
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
            description="Number of text inputs to include in each request for batch processing endpoints. Supported by `embeddings` "
            "and `rankings` endpoint types where models can process multiple inputs simultaneously for efficiency. "
            "Set to 1 for single-input requests. Not applicable to `chat` or `completions` endpoints.",
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
            description="Distribution of (ISL, OSL) pairs with probabilities for mixed workload simulation. "
            "Format: `ISL,OSL:prob;ISL,OSL:prob` (semicolons separate pairs, probabilities are percentages 0-100 that must sum to 100). "
            "Supports optional stddev: `ISL|stddev,OSL|stddev:prob`. "
            "Examples: `128,64:25;512,128:50;1024,256:25` or with variance: `256|10,128|5:40;512|20,256|10:60`. "
            "Also supports bracket `[(256,128):40,(512,256):60]` and JSON formats.",
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
