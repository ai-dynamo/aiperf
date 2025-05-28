# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from contextlib import nullcontext as does_not_raise

import pytest

from aiperf.common.exceptions import GeneratorConfigurationError
from aiperf.common.tokenizer import Tokenizer
from aiperf.services.dataset.generator.prompt import PromptGenerator


class TestPromptGenerator:
    def test_synthetic_prompt_default(self):
        tokenizer = Tokenizer.from_pretrained("gpt2")
        _ = PromptGenerator.create_synthetic_prompt(tokenizer)

    def test_synthetic_prompt_zero_token(self):
        tokenizer = Tokenizer.from_pretrained("gpt2")
        prompt = PromptGenerator.create_synthetic_prompt(
            tokenizer=tokenizer,
            prompt_tokens_mean=0,
            prompt_tokens_stddev=0,
        )

        assert prompt == ""
        assert len(tokenizer.encode(prompt)) == 0

    def test_synthetic_prompt_nonzero_tokens(self):
        prompt_tokens = 123
        tokenizer = Tokenizer.from_pretrained("gpt2")
        prompt = PromptGenerator.create_synthetic_prompt(
            tokenizer=tokenizer,
            prompt_tokens_mean=prompt_tokens,
            prompt_tokens_stddev=0,
        )
        assert len(tokenizer.encode(prompt)) == prompt_tokens

    @pytest.mark.parametrize(
        "test_num_tokens, context",
        [
            (12, does_not_raise()),
            (9, pytest.raises(GeneratorConfigurationError)),
            (16, pytest.raises(GeneratorConfigurationError)),
        ],
    )
    def test_generate_prompt_with_token_reuse(self, test_num_tokens, context):
        tokenizer = Tokenizer.from_pretrained("gpt2")
        with context:
            _ = PromptGenerator._generate_prompt_with_token_reuse(
                tokenizer=tokenizer,
                num_tokens=test_num_tokens,
                prompt_hash_list=[1, 2, 3],
                block_size=5,
            )
