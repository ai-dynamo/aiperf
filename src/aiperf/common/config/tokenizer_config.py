# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from pydantic import Field

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.cli_parameter import CLIParameter, DisableCLI
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

    resolved_names: Annotated[
        dict[str, str] | None,
        Field(
            description="Mapping of model names to resolved tokenizer names after HuggingFace Hub alias resolution. "
            "Set by config validator during startup, before services are spawned. "
            "Services should use `get_tokenizer_name_for_model()` to look up the tokenizer for a specific model.",
        ),
        DisableCLI(reason="This is automatically set"),
    ] = None

    def get_tokenizer_name_for_model(self, model_name: str) -> str:
        """Get the tokenizer name to use for a given model.

        Resolution order:
        1. Pre-resolved name from `resolved_names` (set by CLI after alias resolution)
        2. Explicitly configured tokenizer name
        3. The model name itself (assumes model repo contains tokenizer)

        Args:
            model_name: The model name to get the tokenizer for.

        Returns:
            The tokenizer name to use for loading.
        """
        if self.resolved_names and model_name in self.resolved_names:
            return self.resolved_names[model_name]
        return self.name or model_name

    @property
    def should_resolve_alias(self) -> bool:
        """Whether alias resolution should be performed when loading tokenizers.

        Returns False if `resolved_names` is set (CLI already resolved aliases),
        True otherwise to enable HuggingFace Hub alias resolution.
        """
        return self.resolved_names is None
