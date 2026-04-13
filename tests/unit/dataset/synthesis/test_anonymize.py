# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for trace anonymization."""

from unittest.mock import MagicMock

import pytest

from aiperf.common.tokenizer import Tokenizer


class TestTokenizerApplyChatTemplate:
    """Tests for Tokenizer.apply_chat_template."""

    def test_apply_chat_template_delegates_to_underlying(self) -> None:
        """Test that apply_chat_template calls the underlying tokenizer."""
        tokenizer = Tokenizer()
        mock_hf = MagicMock()
        mock_hf.apply_chat_template.return_value = "<|user|>Hello<|end|>"
        tokenizer._tokenizer = mock_hf

        messages = [{"role": "user", "content": "Hello"}]
        result = tokenizer.apply_chat_template(messages)

        assert result == "<|user|>Hello<|end|>"
        mock_hf.apply_chat_template.assert_called_once_with(
            messages, tokenize=False, add_generation_prompt=True
        )

    def test_apply_chat_template_not_initialized_raises(self) -> None:
        """Test that calling apply_chat_template before init raises."""
        tokenizer = Tokenizer()

        with pytest.raises(Exception, match="not initialized"):
            tokenizer.apply_chat_template([{"role": "user", "content": "Hi"}])
