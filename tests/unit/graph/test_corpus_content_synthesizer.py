# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CorpusContentSynthesizer must fail loud on tokenizer load, never fall back to builtin."""

import pytest

from aiperf.common.tokenizer import BUILTIN_TOKENIZER_NAME, Tokenizer
from aiperf.dataset import _tokenizer_preload
from aiperf.dataset.graph.adapters.shared import content as _shared_content


def test_tokenizer_load_failure_fails_loud_no_builtin_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient tokenizer load failure propagates instead of silently using builtin."""
    # Substituting builtin would emit builtin-sized content while claiming the
    # requested tokenizer, so the failure must surface to the caller.
    _tokenizer_preload.clear_preloaded()
    _shared_content.CorpusContentSynthesizer.reset_worker_cache()

    requested = "some-org/some-private-tokenizer"
    boom = RuntimeError("HF unreachable")

    def _always_fail(name: str, **kwargs: object) -> Tokenizer:
        raise boom

    monkeypatch.setattr(Tokenizer, "from_pretrained", staticmethod(_always_fail))

    with pytest.raises(RuntimeError, match="HF unreachable"):
        _shared_content.CorpusContentSynthesizer._build_generator(
            requested, _shared_content.PROMPT_CORPUS_CODING
        )

    # Sanity: the requested name is not the builtin name, so the old code path
    # would have substituted builtin instead of raising.
    assert requested != BUILTIN_TOKENIZER_NAME
