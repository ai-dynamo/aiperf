#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from contextlib import nullcontext as does_not_raise

import pytest

from aiperf.common.exceptions import SyntheticDataConfigurationException

# TODO: Need ConfigCommand to run the tests
# from genai_perf.config.input.config_command import ConfigCommand
from aiperf.common.tokenizer import get_tokenizer
from aiperf.services.dataset_manager.data_generator import SyntheticPromptGenerator


class TestSyntheticPromptGenerator:
    # TODO: Uncomment when ConfigCommand is ready
    def test_synthetic_prompt_default(self):
        # config = ConfigCommand({"model_name": "test_model"})
        # config.tokenizer.name = "gpt2"
        tokenizer = get_tokenizer("gpt2")
        _ = SyntheticPromptGenerator.create_synthetic_prompt(tokenizer)

    def test_synthetic_prompt_zero_token(self):
        # config = ConfigCommand({"model_name": "test_model"})
        # config.tokenizer.name = "gpt2"
        tokenizer = get_tokenizer("gpt2")
        prompt = SyntheticPromptGenerator.create_synthetic_prompt(
            tokenizer=tokenizer,
            prompt_tokens_mean=0,
            prompt_tokens_stddev=0,
        )

        assert prompt == ""
        assert len(tokenizer.encode(prompt)) == 0

    def test_synthetic_prompt_nonzero_tokens(self):
        prompt_tokens = 123
        tolerance = 2
        # config = ConfigCommand({"model_name": "test_model"})
        # config.tokenizer.name = "gpt2"
        tokenizer = get_tokenizer("gpt2")
        prompt = SyntheticPromptGenerator.create_synthetic_prompt(
            tokenizer=tokenizer,
            prompt_tokens_mean=prompt_tokens,
            prompt_tokens_stddev=0,
        )
        assert len(tokenizer.encode(prompt)) <= 123 + tolerance
        assert len(tokenizer.encode(prompt)) >= 123 - tolerance

    @pytest.mark.parametrize(
        "test_num_tokens, context",
        [
            (12, does_not_raise()),
            (9, pytest.raises(SyntheticDataConfigurationException)),
            (16, pytest.raises(SyntheticDataConfigurationException)),
        ],
    )
    def test_generate_prompt_with_token_reuse(self, test_num_tokens, context):
        # config = ConfigCommand({"model_name": "test_model"})
        # config.tokenizer.name = "gpt2"
        tokenizer = get_tokenizer("gpt2")
        with context:
            _ = SyntheticPromptGenerator._generate_prompt_with_token_reuse(
                tokenizer=tokenizer,
                num_tokens=test_num_tokens,
                prompt_hash_list=[1, 2, 3],
                block_size=5,
            )
