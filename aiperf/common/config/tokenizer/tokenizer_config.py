# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

import cyclopts
from pydantic import Field

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.config_defaults import TokenizerDefaults


class TokenizerConfig(BaseConfig):
    """
    A configuration class for defining tokenizer related settings.
    """

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
        cyclopts.Parameter(
            name=("--tokenizer-name"),
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
        cyclopts.Parameter(
            name=("--tokenizer-revision"),
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
        cyclopts.Parameter(
            name=("--tokenizer-trust-remote-code"),
        ),
    ] = TokenizerDefaults.TRUST_REMOTE_CODE
