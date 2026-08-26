# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from tests.harness.fake_tokenizer import FakeTokenizer


def test_fake_tokenizer_has_no_prompt_special_tokens() -> None:
    assert FakeTokenizer().num_prompt_special_tokens() == 0
