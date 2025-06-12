#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Annotated, Any

from pydantic import BeforeValidator, Field

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.config_defaults import TokenizerDefaults


class TokenizerConfig(BaseConfig):
    """
    A configuration class for defining tokenizer related settings.
    """

    name: Annotated[
        str,
        Field(
            default=TokenizerDefaults.NAME,
            description=(
                "The HuggingFace tokenizer to use to interpret token metrics "
                "from prompts and responses. The value can be the "
                "name of a tokenizer or the filepath of the tokenizer. "
                "The default value is the model name."
            ),
        ),
    ]
    revision: Annotated[
        str,
        Field(
            default=TokenizerDefaults.REVISION,
            description=(
                "The specific model version to use. "
                "It can be a branch name, tag name, or commit ID."
            ),
        ),
    ]
    trust_remote_code: Annotated[
        bool,
        Field(
            default=TokenizerDefaults.TRUST_REMOTE_CODE,
            description=(
                "Allows custom tokenizer to be downloaded and executed. "
                "This carries security risks and should only be used for repositories you trust. "
                "This is only necessary for custom tokenizers stored in HuggingFace Hub."
            ),
        ),
    ]
