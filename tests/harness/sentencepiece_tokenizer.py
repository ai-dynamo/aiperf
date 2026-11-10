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
from tokenizers import decoders, models, normalizers, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from aiperf.common.tokenizer import Tokenizer

SENTENCEPIECE_CORPUS = "waaa " * 512
WORD_START_ID = 3
# Single and quadruple spaces between words, like indented code.
SENTENCEPIECE_WHITESPACE_CORPUS = "waaa  waaa waaa    waaa " * 128


def make_sentencepiece_tokenizer() -> Tokenizer:
    """Return an AIPerf Tokenizer backed by the toy Metaspace BPE model."""
    vocab = {"▁": 0, "w": 1, "a": 2, "▁w": WORD_START_ID, "aa": 4}
    backend = BackendTokenizer(models.BPE(vocab, [("▁", "w"), ("a", "a")]))
    backend.pre_tokenizer = pre_tokenizers.Metaspace()
    backend.decoder = decoders.Metaspace()
    tokenizer = Tokenizer()
    tokenizer._tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
    return tokenizer


def make_sentencepiece_whitespace_tokenizer() -> Tokenizer:
    """Return a toy Llama-2-style tokenizer that merges runs of spaces.

    Like Llama-2, it has no pre-tokenizer, so ``▁▁`` and ``▁▁▁▁`` are pieces of
    their own. They carry the ``▁`` marker but are not word starts: a block
    that starts on one merges with whitespace at the end of the previous block,
    or loses a space when the decoder strips the prompt's leading space.
    """
    vocab = {"▁": 0, "w": 1, "a": 2, "▁w": WORD_START_ID, "aa": 4, "▁▁": 5, "▁▁▁▁": 6}
    merges = [("▁", "▁"), ("▁▁", "▁▁"), ("▁", "w"), ("a", "a")]
    backend = BackendTokenizer(models.BPE(vocab, merges))
    backend.normalizer = normalizers.Sequence(
        [normalizers.Prepend("▁"), normalizers.Replace(" ", "▁")]
    )
    backend.decoder = decoders.Sequence(
        [decoders.Replace("▁", " "), decoders.Fuse(), decoders.Strip(" ", 1, 0)]
    )
    tokenizer = Tokenizer()
    tokenizer._tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
    return tokenizer
