# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from aiperf.config.flags import CLIConfig
from aiperf.config.tokenizer import TokenizerDefaults


def test_tokenizer_config_defaults():
    """
    Test the default values of the tokenizer fields on CLIConfig.

    This test verifies that CLIConfig is initialized with the correct
    default values as defined in the TokenizerDefaults class.
    """
    config = CLIConfig()
    assert config.tokenizer_name == TokenizerDefaults.NAME
    assert config.tokenizer_revision == TokenizerDefaults.REVISION
    assert config.trust_remote_code == TokenizerDefaults.TRUST_REMOTE_CODE


def test_output_config_custom_values():
    """
    Test the tokenizer fields on CLIConfig with custom values.

    This test verifies that CLIConfig correctly initializes its tokenizer
    fields when provided with a dictionary of custom values.
    """
    custom_values = {
        "tokenizer_name": "custom_tokenizer",
        "tokenizer_revision": "v1.0.0",
        "trust_remote_code": True,
    }
    config = CLIConfig(**custom_values)

    for key, value in custom_values.items():
        assert getattr(config, key) == value


class TestGetTokenizerNameForModel:
    """Methods moved from v1 TokenizerConfig to v2 in commit bcc8fe384.

    The v1 TokenizerConfig is now flattened into CLIConfig (no methods, no
    validators); see `aiperf.config.tokenizer.TokenizerConfig` for the
    `get_tokenizer_name_for_model` / `should_resolve_alias` behavior tests.
    """


class TestShouldResolveAlias:
    """See note on TestGetTokenizerNameForModel — behavior lives on v2."""
