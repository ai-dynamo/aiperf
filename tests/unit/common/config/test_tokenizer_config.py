# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from aiperf.common.config import TokenizerConfig, TokenizerDefaults


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


class TestGetTokenizerNameForModel:
    """Tests for TokenizerConfig.get_tokenizer_name_for_model method."""

    def test_uses_resolved_names_when_available(self):
        """Test that resolved_names takes priority over name and model_name."""
        config = TokenizerConfig(
            name="explicit-tokenizer",
            resolved_names={
                "model-a": "resolved/model-a",
                "model-b": "resolved/model-b",
            },
        )
        assert config.get_tokenizer_name_for_model("model-a") == "resolved/model-a"
        assert config.get_tokenizer_name_for_model("model-b") == "resolved/model-b"

    def test_falls_back_to_name_when_model_not_in_resolved(self):
        """Test fallback to explicit name when model not in resolved_names."""
        config = TokenizerConfig(
            name="explicit-tokenizer",
            resolved_names={"other-model": "resolved/other"},
        )
        assert (
            config.get_tokenizer_name_for_model("unknown-model") == "explicit-tokenizer"
        )

    def test_falls_back_to_name_when_no_resolved_names(self):
        """Test fallback to explicit name when resolved_names is None."""
        config = TokenizerConfig(name="explicit-tokenizer")
        assert config.get_tokenizer_name_for_model("any-model") == "explicit-tokenizer"

    def test_falls_back_to_model_name_when_no_name(self):
        """Test fallback to model_name when no name or resolved_names."""
        config = TokenizerConfig()
        assert config.get_tokenizer_name_for_model("my-model") == "my-model"

    def test_falls_back_to_model_name_with_empty_resolved(self):
        """Test fallback to model_name with empty resolved_names dict."""
        config = TokenizerConfig(resolved_names={})
        assert config.get_tokenizer_name_for_model("my-model") == "my-model"


class TestShouldResolveAlias:
    """Tests for TokenizerConfig.should_resolve_alias property."""

    def test_returns_true_when_no_resolved_names(self):
        """Test that alias resolution is enabled when resolved_names is None."""
        config = TokenizerConfig()
        assert config.should_resolve_alias is True

    def test_returns_false_when_resolved_names_set(self):
        """Test that alias resolution is disabled when resolved_names is set."""
        config = TokenizerConfig(resolved_names={"model": "resolved/model"})
        assert config.should_resolve_alias is False

    def test_returns_false_with_empty_resolved_names(self):
        """Test that alias resolution is disabled even with empty dict."""
        config = TokenizerConfig(resolved_names={})
        assert config.should_resolve_alias is False
