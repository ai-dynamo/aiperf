# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration for trace synthesis with prefix data generation."""

from typing import Annotated

from pydantic import Field

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.cli_parameter import CLIParameter
from aiperf.common.config.groups import Groups


class SynthesisConfig(BaseConfig):
    """Configuration for synthetic trace generation with prefix patterns."""

    _CLI_GROUP = Groups.SYNTHESIS

    speedup_ratio: Annotated[
        float,
        Field(
            default=1.0,
            ge=0.0,
            description="Multiplier for timestamp scaling in synthesized traces",
        ),
        CLIParameter(name=("--synthesis-speedup-ratio",), group=_CLI_GROUP),
    ] = 1.0

    prefix_len_multiplier: Annotated[
        float,
        Field(
            default=1.0,
            ge=0.0,
            description="Multiplier for core prefix branch lengths in radix tree",
        ),
        CLIParameter(name=("--synthesis-prefix-len-multiplier",), group=_CLI_GROUP),
    ] = 1.0

    prefix_root_multiplier: Annotated[
        int,
        Field(
            default=1,
            ge=1,
            description="Number of independent radix trees to distribute traces across",
        ),
        CLIParameter(name=("--synthesis-prefix-root-multiplier",), group=_CLI_GROUP),
    ] = 1

    prompt_len_multiplier: Annotated[
        float,
        Field(
            default=1.0,
            ge=0.0,
            description="Multiplier for leaf path (unique prompt) lengths",
        ),
        CLIParameter(name=("--synthesis-prompt-len-multiplier",), group=_CLI_GROUP),
    ] = 1.0

    max_isl: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description="Maximum input sequence length for filtering. Traces with input_length > max_isl are skipped.",
        ),
        CLIParameter(name=("--synthesis-max-isl",), group=_CLI_GROUP),
    ] = None

    max_osl: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description="Maximum output sequence length cap. Traces with output_length > max_osl are capped to max_osl.",
        ),
        CLIParameter(name=("--synthesis-max-osl",), group=_CLI_GROUP),
    ] = None

    def should_synthesize(self) -> bool:
        """Check if synthesis should be auto-triggered based on non-default values.

        max_isl and max_osl are filters and caps, not synthesis transformations, so they don't trigger synthesis.

        Returns:
            True if any synthesis parameter differs from defaults (excluding max_isl and max_osl)
        """
        return (
            self.speedup_ratio != 1.0
            or self.prefix_len_multiplier != 1.0
            or self.prefix_root_multiplier != 1
            or self.prompt_len_multiplier != 1.0
        )
