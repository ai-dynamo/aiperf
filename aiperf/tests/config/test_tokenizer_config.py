#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0


from aiperf.common.config.config_defaults import TokenizerDefaults
from aiperf.common.config.tokenizer.tokenizer_config import TokenizerConfig


def test_tokenizer_config_defaults():
    """
    Test the default values of the TokenizerConfig class.

    This test verifies that the TokenizerConfig object is initialized with the correct
    default values as defined in the TokenizerDefaults class.
    """
    config = TokenizerConfig()
    assert config.name == TokenizerDefaults.NAME
    assert config.revision == TokenizerDefaults.REVISION
    assert config.trust_remote_code == TokenizerDefaults.TRUST_REMOTE_CODE


def test_output_config_custom_values():
    """
    Test the OutputConfig class with custom values.

    This test verifies that the OutputConfig class correctly initializes its attributes
    when provided with a dictionary of custom values.
    """
    custom_values = {
        "name": "custom_tokenizer",
        "revision": "v1.0.0",
        "trust_remote_code": True,
    }
    config = TokenizerConfig(**custom_values)

    for key, value in custom_values.items():
        assert getattr(config, key) == value
