# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
            description="Mean number of passages to include per ranking request. For `rankings` endpoint type, each request contains a query "
            "and multiple passages to rank. Passages follow normal distribution around this mean (±`--rankings-passages-stddev`). "
            "Higher values test ranking at scale but increase request payload size and processing time.",
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
            description="Standard deviation for number of passages per ranking request. Creates variability in ranking workload complexity. "
            "Passage counts follow normal distribution. Set to 0 for uniform passage counts across all requests.",
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
            description="Mean token length for each passage in ranking requests. Passages are synthetically generated text with lengths "
            "following normal distribution around this mean (±`--rankings-passages-prompt-token-stddev`). "
            "Longer passages increase input processing demands and request size.",
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
            description="Standard deviation for passage token lengths in ranking requests. Creates variability in passage sizes, simulating "
            "realistic heterogeneous document collections. Token lengths follow normal distribution. "
            "Set to 0 for uniform passage lengths.",
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
            description="Mean token length for query text in ranking requests. Each ranking request contains one query and multiple passages. "
            "Queries are synthetically generated with lengths following normal distribution around this mean (±`--rankings-query-prompt-token-stddev`). ",
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
            description="Standard deviation for query token lengths in ranking requests. Creates variability in query complexity, simulating "
            "realistic user search patterns. Token lengths follow normal distribution. "
            "Set to 0 for uniform query lengths.",
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
