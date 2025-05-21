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

import contextlib
import io
from typing import TYPE_CHECKING

# Use TYPE_CHECKING to import BatchEncoding only during static type checks
if TYPE_CHECKING:
    from transformers import BatchEncoding

# TODO: Need ConfigCommand for get_tokenizer
# from genai_perf.config.input.config_command import ConfigCommand
from aiperf.common.exceptions import TokenizerInitializationException


class Tokenizer:
    """
    This class provides a simplified interface for using Huggingface
    tokenizers, with default arguments for common operations.
    """

    def __init__(self) -> None:
        """
        Initialize the tokenizer with default values for call, encode, and decode.
        """
        self._call_args = {"add_special_tokens": False}
        self._encode_args = {"add_special_tokens": False}
        self._decode_args = {"skip_special_tokens": True}

    def set_tokenizer(self, name: str, trust_remote_code: bool, revision: str) -> None:
        """
        Set the tokenizer from Huggingface.co or local filesystem.

        Args:
            name: The name or path of the tokenizer.
            trust_remote_code: Whether to trust remote code when loading the tokenizer.
            revision: The specific model version to use.
        """
        try:
            # Silence tokenizer warning on import and first use
            with (
                contextlib.redirect_stdout(io.StringIO()) as _,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                from transformers import AutoTokenizer
                from transformers import logging as token_logger

                token_logger.set_verbosity_error()
                tokenizer = AutoTokenizer.from_pretrained(
                    name, trust_remote_code=trust_remote_code, revision=revision
                )
        except Exception as e:
            raise TokenizerInitializationException(e) from e
        self._tokenizer = tokenizer

    def __call__(self, text, **kwargs) -> "BatchEncoding":
        """
        Call the underlying Huggingface tokenizer with default arguments,
        which can be overridden by kwargs.

        Args:
            text: The input text to tokenize.

        Returns:
            A BatchEncoding object containing the tokenized output.
        """
        return self._tokenizer(text, **{**self._call_args, **kwargs})

    def encode(self, text, **kwargs) -> list[int]:
        """
        Encode the input text into a list of token IDs.

        This method calls the underlying Huggingface tokenizer's encode
        method with default arguments, which can be overridden by kwargs.

        Args:
            text: The input text to encode.

        Returns:
            A list of token IDs.
        """
        return self._tokenizer.encode(text, **{**self._encode_args, **kwargs})

    def decode(self, token_ids, **kwargs) -> str:
        """
        Decode a list of token IDs back into a string.

        This method calls the underlying Huggingface tokenizer's decode
        method with default arguments, which can be overridden by kwargs.

        Args:
            token_ids: A list of token IDs to decode.

        Returns:
            The decoded string.
        """
        return self._tokenizer.decode(token_ids, **{**self._decode_args, **kwargs})

    def bos_token_id(self) -> int:
        """
        Get the beginning-of-sequence (BOS) token ID.

        Returns:
            The BOS token ID.
        """
        return self._tokenizer.bos_token_id

    def __repr__(self) -> str:
        """
        Return a string representation of the underlying tokenizer.

        Returns:
            The string representation of the tokenizer.
        """
        return self._tokenizer.__repr__()


def get_empty_tokenizer() -> Tokenizer:
    """
    Return a Tokenizer without a tokenizer set
    """
    return Tokenizer()


# TODO: Need ConfigCommand for get_tokenizer
# def get_tokenizer(config: ConfigCommand) -> Tokenizer:
def get_tokenizer(
    name: str,
    trust_remote_code: bool = False,
    revision: str = "main",
) -> Tokenizer:
    """
    Return tokenizer for the given model name
    """
    tokenizer = Tokenizer()
    tokenizer.set_tokenizer(name, trust_remote_code, revision)
    return tokenizer
