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

# TODO: uncomment when ConfigCommand and TokenizerDefaults are implemented
# from genai_perf.config.input.config_command import ConfigCommand
# from genai_perf.config.input.config_defaults import TokenizerDefaults
from aiperf.common.tokenizer import get_tokenizer


class TestTokenizer:
    # TODO: uncomment when ConfigCommand and TokenizerDefaults are implemented
    # def _create_tokenizer_config(
    #    self, name, trust_remote_code=False, revision=TokenizerDefaults.REVISION
    # ):
    #    config = ConfigCommand({"model_name": "test_model"})
    #    config.tokenizer.name = name
    #    config.tokenizer.trust_remote_code = trust_remote_code
    #    config.tokenizer.revision = revision

    #    return config

    def test_default_tokenizer(self):
        # config = self._create_tokenizer_config(name="gpt2")
        get_tokenizer(name="gpt2")

    def test_non_default_tokenizer(self):
        # config = self._create_tokenizer_config(name="gpt2")
        get_tokenizer(name="gpt2")

    def test_default_tokenizer_all_args(self):
        # config = self._create_tokenizer_config(
        #    name="gpt2",
        #    trust_remote_code=False,
        #    revision=TokenizerDefaults.REVISION,
        # )
        get_tokenizer(name="gpt2")

    def test_non_default_tokenizer_all_args(self):
        # config = self._create_tokenizer_config(
        #    name="gpt2",
        #    trust_remote_code=False,
        #    revision="11c5a3d5811f50298f278a704980280950aedb10",
        # )
        get_tokenizer(
            name="gpt2",
            trust_remote_code=False,
            revision="11c5a3d5811f50298f278a704980280950aedb10",
        )

    def test_default_args(self):
        # config = self._create_tokenizer_config(
        #    name="hf-internal-testing/llama-tokenizer"
        # )
        tokenizer = get_tokenizer(name="hf-internal-testing/llama-tokenizer")

        # There are 3 special tokens in the default tokenizer
        #  - <unk>: 0  (unknown)
        #  - <s>: 1  (beginning of sentence)
        #  - </s>: 2  (end of sentence)
        special_tokens = list(tokenizer._tokenizer.added_tokens_encoder.keys())
        special_token_ids = list(tokenizer._tokenizer.added_tokens_encoder.values())

        # special tokens are disabled by default
        text = "This is test."
        tokens = tokenizer(text)["input_ids"]
        assert all([s not in tokens for s in special_token_ids])

        tokens = tokenizer.encode(text)
        assert all([s not in tokens for s in special_token_ids])

        output = tokenizer.decode(tokens)
        assert all([s not in output for s in special_tokens])

        # check special tokens is enabled
        text = "This is test."
        tokens = tokenizer(text, add_special_tokens=True)["input_ids"]
        assert any([s in tokens for s in special_token_ids])

        tokens = tokenizer.encode(text, add_special_tokens=True)
        assert any([s in tokens for s in special_token_ids])

        output = tokenizer.decode(tokens, skip_special_tokens=False)
        assert any([s in output for s in special_tokens])
