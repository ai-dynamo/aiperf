# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
            description="HuggingFace tokenizer identifier or local path for token counting in prompts and responses. "
            "Accepts model names (e.g., `meta-llama/Llama-2-7b-hf`) or filesystem paths to tokenizer files. "
            "If not specified, defaults to the value of `--model-names`. Essential for accurate token-based metrics "
            "(input/output token counts, token throughput).",
        ),
        CLIParameter(
            name=("--tokenizer"),
            group=_CLI_GROUP,
        ),
    ] = TokenizerDefaults.NAME

    revision: Annotated[
        str,
        Field(
            description="Specific tokenizer version to load from HuggingFace Hub. Can be a branch name (e.g., `main`), "
            "tag name (e.g., `v1.0`), or full commit hash. Ensures reproducible tokenization across runs by pinning "
            "to a specific version. Defaults to `main` branch if not specified.",
        ),
        CLIParameter(
            name=("--tokenizer-revision"),
            group=_CLI_GROUP,
        ),
    ] = TokenizerDefaults.REVISION

    trust_remote_code: Annotated[
        bool,
        Field(
            description="Allow execution of custom Python code from HuggingFace Hub tokenizer repositories. Required for tokenizers "
            "with custom implementations not in the standard `transformers` library. **Security Warning**: Only enable for "
            "trusted repositories, as this executes arbitrary code. Unnecessary for standard tokenizers.",
        ),
        CLIParameter(
            name=("--tokenizer-trust-remote-code"),
            group=_CLI_GROUP,
        ),
    ] = TokenizerDefaults.TRUST_REMOTE_CODE
