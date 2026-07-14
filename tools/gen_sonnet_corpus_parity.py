# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the sonnet-corpus cross-language parity fixture.

Runs the *real* Python ``PromptGenerator._initialize_corpus``
(``src/aiperf/dataset/generator/prompt.py``) with the built-in tiktoken
``o200k_base`` tokenizer and records a digest of the tokenized Shakespeare
corpus. The Rust integration test ``rust/aiperf/tests/sonnet_corpus_parity.rs``
rebuilds the same corpus via ``aiperf::dataset::corpus::tokenize_sonnet_corpus``
and asserts an identical token count, head/tail, and SHA-256 — proving the two
implementations tokenize the embedded corpus byte-for-byte identically.

Usage::

    python tools/gen_sonnet_corpus_parity.py \
        > rust/aiperf/tests/data/sonnet_corpus_parity.json
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys

from aiperf.common import random_generator as rng
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.generator.prompt import PromptGenerator

TOKENIZER = "o200k_base"


def main() -> int:
    # PromptGenerator derives seeded RNG streams in its constructor; the global
    # manager must be initialized first. The seed is irrelevant here — corpus
    # tokenization does not consume any RNG stream.
    rng.init(seed=0)

    tokenizer = Tokenizer.from_pretrained(TOKENIZER)
    generator = PromptGenerator(prompts=None, prefix_prompts=None, tokenizer=tokenizer)
    corpus = generator._tokenized_corpus
    assert corpus, "tokenized corpus is empty"
    assert generator._corpus_size == len(corpus)

    raw = b"".join(struct.pack("<I", token_id) for token_id in corpus)
    fixture = {
        "tokenizer": TOKENIZER,
        "corpus_token_count": len(corpus),
        "sha256_le_u32": hashlib.sha256(raw).hexdigest(),
        "first_16": list(corpus[:16]),
        "last_16": list(corpus[-16:]),
    }
    json.dump(fixture, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
