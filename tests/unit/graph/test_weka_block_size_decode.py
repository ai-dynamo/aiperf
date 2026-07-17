# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weka content decode honors the TRACE's ``block_size``.

The trie GEOMETRY (covered-block math) uses ``trace.block_size``; if the
production content callbacks instead decoded every hash block at the
synthesizer's hardcoded 64 tokens, a ``block_size: 32`` trace with ``in: 64``
would synthesize
a 128-token prompt (2 covered blocks x 64) instead of 64, silently corrupting
every ISL / prefix-cache measurement on non-default-block-size corpora. These
tests pin that ``_default_callbacks`` (and therefore ``build_trie_graph``)
decode at the trace's own block size, mirroring the dynamo adapter's
``dynamo_recon_callbacks``.

Hermetic: ``fake_tokenizer`` pins ``Tokenizer.from_pretrained`` to the
deterministic ``FakeTokenizer`` (decode = ``"tok$" * n_tokens``), so prompt
token counts are recoverable from content length with no real tokenizer.
"""

from __future__ import annotations

import orjson
import pytest

from aiperf.dataset.graph.adapters.shared.content import CorpusContentSynthesizer
from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace
from aiperf.dataset.graph.adapters.weka.trie_build import _default_callbacks
from aiperf.dataset.graph.segment_ir.envelope import read_prompt_segment_ids
from tests.harness.fake_tokenizer import TOKEN_LEN

_BLOCK32_TRACE = {
    "id": "trace_bs32",
    "models": ["M"],
    "block_size": 32,
    "hash_id_scope": "local",
    "requests": [
        {
            "t": 0.0,
            "type": "n",
            "model": "M",
            "in": 64,
            "out": 8,
            "hash_ids": [1, 2],
            "api_time": 0.5,
        },
    ],
}


@pytest.fixture(autouse=True)
def _fresh_synth_cache():
    """Isolate the process-level synthesizer cache from other tests."""
    CorpusContentSynthesizer.reset_worker_cache()
    yield
    CorpusContentSynthesizer.reset_worker_cache()


@pytest.mark.parametrize("block_size", [16, 32, 64])
def test_default_callbacks_decode_at_trace_block_size(
    fake_tokenizer: None,  # noqa: ARG001
    block_size: int,
) -> None:
    """Each hash id decodes to exactly ``block_size`` tokens (not the legacy 64)."""
    callbacks = _default_callbacks(
        "fake-tok", "coding", 0, trace_id="trace_bs", block_size=block_size
    )
    assert len(callbacks.decode_block_tokens([1])) == block_size
    assert len(callbacks.decode_block_tokens([1, 2])) == 2 * block_size


def test_build_block32_trace_synthesizes_recorded_isl(
    fake_tokenizer: None,  # noqa: ARG001
    tmp_path,
) -> None:
    """A ``block_size=32`` trace with ``in=64`` yields a 64-token prompt (was 128)."""
    trace_file = tmp_path / "bs32.json"
    trace_file.write_bytes(orjson.dumps(_BLOCK32_TRACE))

    parsed = from_weka_trace(trace_file, content_root_seed=0)

    # The trie route carries NO inline prompt; the content is in the segment
    # pool, addressed via the node's prompt_segment_ids path.
    node = parsed.graph.nodes["trace_bs32:0"]
    messages = parsed.segment_pool.materialize(read_prompt_segment_ids(node))
    prompt_tokens = sum(len(m["content"]) // TOKEN_LEN for m in messages)
    assert prompt_tokens == 64, (
        f"covered-count ISL must equal recorded in=64 at block_size=32, "
        f"got {prompt_tokens} (a 64-token legacy decode doubles it to 128)"
    )


def test_from_weka_trace_threads_max_osl_to_dispatch(
    fake_tokenizer: None,  # noqa: ARG001
    tmp_path,
) -> None:
    """``max_osl`` reaches the trie build through the parse seam (finding W2)."""
    capped_trace = dict(_BLOCK32_TRACE, id="trace_capped")
    capped_trace["requests"] = [dict(_BLOCK32_TRACE["requests"][0], out=5000)]
    trace_file = tmp_path / "capped.json"
    trace_file.write_bytes(orjson.dumps(capped_trace))

    parsed = from_weka_trace(trace_file, content_root_seed=0, max_osl=100)
    assert parsed.graph.nodes["trace_capped:0"].max_tokens == 100
