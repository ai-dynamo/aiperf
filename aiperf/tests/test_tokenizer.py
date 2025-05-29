# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.exceptions import TokenizerInitializationError
from aiperf.common.tokenizer import Tokenizer


class TestTokenizer:
    def test_empty_tokenizer(self):
        tokenizer = Tokenizer()
        assert tokenizer._tokenizer is None

        with pytest.raises(TokenizerInitializationError):
            tokenizer("test")
        with pytest.raises(TokenizerInitializationError):
            tokenizer.encode("test")
        with pytest.raises(TokenizerInitializationError):
            tokenizer.decode([1])
        with pytest.raises(TokenizerInitializationError):
            tokenizer.bos_token_id()

    def test_non_empty_tokenizer(self, mock_hf_tokenizer):
        tokenizer = Tokenizer.from_pretrained("gpt2")
        assert tokenizer._tokenizer is not None

        assert tokenizer("This is a test")["input_ids"] == [10, 11, 12, 13]
        assert tokenizer.encode("This is a test") == [10, 11, 12, 13]
        assert tokenizer.decode([10, 11, 12, 13]) == "10 11 12 13"
        assert tokenizer.bos_token_id() == 1

    def test_all_args(self, mock_hf_tokenizer):
        tokenizer = Tokenizer.from_pretrained(
            name="gpt2",
            trust_remote_code=True,
            revision="11c5a3d5811f50298f278a704980280950aedb10",
        )
        assert tokenizer._tokenizer is not None
