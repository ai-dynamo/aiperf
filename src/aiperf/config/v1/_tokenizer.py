# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""v1 TokenizerConfig - CLI-only input DTO for tokenizer settings.

No validators. AIPerfConfig is the sole validation gate.
"""

from typing import Annotated

from pydantic import Field

from aiperf.config._base import BaseConfig
from aiperf.config.cli_parameter import CLIParameter, DisableCLI, Groups


class TokenizerConfig(BaseConfig):
    """A configuration class for defining tokenizer related settings."""

    _CLI_GROUP = Groups.TOKENIZER

    name: Annotated[
        str | None,
        Field(
            description="HuggingFace tokenizer identifier, local path, or `builtin` for token counting in prompts and responses. "
            "Accepts model names (e.g., `meta-llama/Llama-2-7b-hf`), filesystem paths to tokenizer files, "
            "or `builtin` for a zero-network-access tokenizer backed by tiktoken (o200k_base encoding). "
            "If not specified, defaults to the value of `--model-names`. "
            "If `--tokenizer` is not set and the model name looks like an obvious placeholder "
            "(e.g. `mock-model`, `test-model`, `fake-model`), AIPerf substitutes `builtin` automatically "
            "and emits a warning. Essential for accurate token-based metrics "
            "(input/output token counts, token throughput).",
        ),
        CLIParameter(
            name=("--tokenizer"),
            group=_CLI_GROUP,
        ),
    ] = None

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
    ] = "main"

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
    ] = False

    resolved_names: Annotated[
        dict[str, str] | None,
        Field(
            description="Mapping of model names to resolved tokenizer names after HuggingFace Hub alias resolution. "
            "Set by config validator during startup, before services are spawned. "
            "Services should use `get_tokenizer_name_for_model()` to look up the tokenizer for a specific model.",
        ),
        DisableCLI(reason="This is automatically set"),
    ] = None
