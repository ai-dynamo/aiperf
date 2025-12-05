# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from pydantic import Field

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.cli_parameter import CLIParameter
from aiperf.common.config.config_defaults import RankingsDefaults
from aiperf.common.config.groups import Groups


class RankingsPassagesConfig(BaseConfig):
    """
    A configuration class for defining rankings passages related settings.
    """

    _CLI_GROUP = Groups.RANKINGS

    mean: Annotated[
        int,
        Field(
            ge=1,
            description=(
                "Mean number of passages per rankings entry (per query)(default 1)."
            ),
        ),
        CLIParameter(
            name=("--rankings-passages-mean",),
            group=_CLI_GROUP,
        ),
    ] = RankingsDefaults.PASSAGES_MEAN

    stddev: Annotated[
        int,
        Field(
            ge=0,
            description=("Stddev for passages per rankings entry (default 0)."),
        ),
        CLIParameter(
            name=("--rankings-passages-stddev",),
            group=_CLI_GROUP,
        ),
    ] = RankingsDefaults.PASSAGES_STDDEV

    prompt_token_mean: Annotated[
        int,
        Field(
            ge=1,
            description=(
                "Mean number of tokens in a passage entry for rankings (default 550)."
            ),
        ),
        CLIParameter(
            name=("--rankings-passages-prompt-token-mean",),
            group=_CLI_GROUP,
        ),
    ] = RankingsDefaults.PASSAGES_PROMPT_TOKEN_MEAN

    prompt_token_stddev: Annotated[
        int,
        Field(
            ge=0,
            description=(
                "Stddev for number of tokens in a passage entry for rankings (default 0)."
            ),
        ),
        CLIParameter(
            name=("--rankings-passages-prompt-token-stddev",),
            group=_CLI_GROUP,
        ),
    ] = RankingsDefaults.PASSAGES_PROMPT_TOKEN_STDDEV


class RankingsQueryConfig(BaseConfig):
    """
    A configuration class for defining rankings query related settings.
    """

    _CLI_GROUP = Groups.RANKINGS

    prompt_token_mean: Annotated[
        int,
        Field(
            ge=1,
            description=(
                "Mean number of tokens in a query entry for rankings (default 550)."
            ),
        ),
        CLIParameter(
            name=("--rankings-query-prompt-token-mean",),
            group=_CLI_GROUP,
        ),
    ] = RankingsDefaults.QUERY_PROMPT_TOKEN_MEAN

    prompt_token_stddev: Annotated[
        int,
        Field(
            ge=0,
            description=(
                "Stddev for number of tokens in a query entry for rankings (default 0)."
            ),
        ),
        CLIParameter(
            name=("--rankings-query-prompt-token-stddev",),
            group=_CLI_GROUP,
        ),
    ] = RankingsDefaults.QUERY_PROMPT_TOKEN_STDDEV


class RankingsConfig(BaseConfig):
    """
    A configuration class for defining rankings related settings.
    """

    _CLI_GROUP = Groups.RANKINGS

    passages: RankingsPassagesConfig = RankingsPassagesConfig()
    query: RankingsQueryConfig = RankingsQueryConfig()
