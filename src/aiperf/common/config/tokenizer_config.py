# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from pydantic import Field

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.cli_parameter import CLIParameter
from aiperf.common.config.config_defaults import TokenizerDefaults
from aiperf.common.config.groups import Groups


class TokenizerConfig(BaseConfig):
    """
    A configuration class for defining tokenizer related settings.
    """

    _CLI_GROUP = Groups.TOKENIZER

    name: Annotated[
        str | None,
        Field(
            description=(
                "The HuggingFace tokenizer to use to interpret token metrics "
                "from prompts and responses.\nThe value can be the "
                "name of a tokenizer or the filepath of the tokenizer.\n"
                "The default value is the model name."
            ),
        ),
        CLIParameter(
            name=("--tokenizer"),
            group=_CLI_GROUP,
        ),
    ] = TokenizerDefaults.NAME

    revision: Annotated[
        str,
        Field(
            description=(
                "The specific model version to use.\n"
                "It can be a branch name, tag name, or commit ID."
            ),
        ),
        CLIParameter(
            name=("--tokenizer-revision"),
            group=_CLI_GROUP,
        ),
    ] = TokenizerDefaults.REVISION

    trust_remote_code: Annotated[
        bool,
        Field(
            description=(
                "Allows custom tokenizer to be downloaded and executed.\n"
                "This carries security risks and should only be used for repositories you trust.\n"
                "This is only necessary for custom tokenizers stored in HuggingFace Hub."
            ),
        ),
        CLIParameter(
            name=("--tokenizer-trust-remote-code"),
            group=_CLI_GROUP,
        ),
    ] = TokenizerDefaults.TRUST_REMOTE_CODE
