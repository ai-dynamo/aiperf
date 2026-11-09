# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline SentencePiece-style tokenizer that reproduces hash-block token drift.

Like Mistral v0.3 and Llama-2, it has no usable newline separator, and its
continuation pieces have no space-prefixed twin. Each corpus word ``waaa``
encodes as ``[▁w, aa, a]``. Two drift modes follow (NVBug 7067483):

* Text that starts on a continuation piece gains a token on re-encode,
  because the first word gets a ``▁`` prefix: ``[aa, a]`` -> ``[▁, aa, a]``.
* Two ``a`` pieces joined at a seam merge into one ``aa`` and lose a token.
"""

from tokenizers import Tokenizer as BackendTokenizer
from tokenizers import decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from aiperf.common.tokenizer import Tokenizer

SENTENCEPIECE_CORPUS = "waaa " * 512
WORD_START_ID = 3


def make_sentencepiece_tokenizer() -> Tokenizer:
    """Return an AIPerf Tokenizer backed by the toy Metaspace BPE model."""
    vocab = {"▁": 0, "w": 1, "a": 2, "▁w": WORD_START_ID, "aa": 4}
    backend = BackendTokenizer(models.BPE(vocab, [("▁", "w"), ("a", "a")]))
    backend.pre_tokenizer = pre_tokenizers.Metaspace()
    backend.decoder = decoders.Metaspace()
    tokenizer = Tokenizer()
    tokenizer._tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
    return tokenizer
