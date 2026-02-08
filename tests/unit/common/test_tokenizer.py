# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest

from aiperf.common.exceptions import NotInitializedError
from aiperf.common.tokenizer import Tokenizer


class TestTokenizer:
    def test_empty_tokenizer(self):
        tokenizer = Tokenizer()
        assert tokenizer._tokenizer is None

        with pytest.raises(NotInitializedError):
            tokenizer("test")
        with pytest.raises(NotInitializedError):
            tokenizer.encode("test")
        with pytest.raises(NotInitializedError):
            tokenizer.decode([1])
        with pytest.raises(NotInitializedError):
            _ = tokenizer.bos_token_id
