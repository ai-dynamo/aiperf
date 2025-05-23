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

from aiperf.common.tokenizer import Tokenizer


class TestTokenizer:
    def test_empty_tokenizer(self):
        tokenizer = Tokenizer()
        assert tokenizer._tokenizer is None

    def test_non_empty_tokenizer(self):
        tokenizer = Tokenizer.from_pretrained("gpt2")
        assert tokenizer._tokenizer is not None

    def test_all_args(self):
        tokenizer = Tokenizer.from_pretrained(
            name="gpt2",
            trust_remote_code=True,
            revision="11c5a3d5811f50298f278a704980280950aedb10",
        )
        assert tokenizer._tokenizer is not None

    def test_default_args(self):
        tokenizer = Tokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
        assert tokenizer._tokenizer is not None

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
