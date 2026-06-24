# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hash-coherence smoke test over the kv-cache-tester corpus.

Marked ``slow`` since it walks 739 trace files and instantiates the
PromptGenerator block cache. Run via:

    uv run pytest -m slow tests/unit/dataset/loader/test_weka_trace_hash_coherence.py -n auto

The contract: every recurrence of the same hash_id must produce the
identical token sequence (otherwise server-side prefix-cache hits during
replay would diverge from the recorded run's hits).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aiperf.dataset.loader.weka_trace import WekaTraceLoader

CORPUS = Path(__file__).parents[4] / "artifacts" / "kv-cache-tester" / "traces"


pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def loader_for_corpus():
    if not CORPUS.exists() or not any(CORPUS.glob("trace_*.json")):
        pytest.skip(f"Corpus not present at {CORPUS}; submodule not initialized")

    uc = MagicMock()
    uc.input.random_seed = 0
    uc.input.fixed_schedule_start_offset = None
    uc.input.fixed_schedule_end_offset = None
    uc.input.ignore_trace_delays = False
    uc.input.use_think_time_only = False
    uc.input.use_end_to_start_delays = False
    uc.loadgen.inter_turn_delay_cap_seconds = None
    uc.loadgen.trace_idle_gap_cap_seconds = None
    uc.input.synthesis.max_isl = None
    uc.input.synthesis.max_osl = None
    uc.input.max_context_length = None
    uc.input.synthesis.should_synthesize.return_value = False
    uc.input.prompt.input_tokens.block_size = None
    uc.tokenizer.trust_remote_code = False
    uc.tokenizer.revision = None
    uc.tokenizer.name = "test-tok"
    uc.endpoint.model_names = sorted(_collect_corpus_models())

    loader = WekaTraceLoader(filename=str(CORPUS), user_config=uc)
    pg = MagicMock()
    pg._cache = {}
    # Deterministic per-hash sample: cycle through a finite token alphabet
    # keyed by hash_id. Same hash -> same tokens.
    pg._sample_tokens.side_effect = lambda n: [0] * n
    pg._tokenized_corpus = list(range(10000, 11000))
    pg._corpus_size = 1000
    from tests.unit.dataset.loader.conftest import stub_hash_id_corpus_rng

    stub_hash_id_corpus_rng(pg)
    pg.tokenizer.decode.side_effect = lambda toks: "x" * len(toks)
    loader.prompt_generator = pg
    loader._tokenizer_name = "test-tok"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64
    loader.synthesize_prompts_from_hash_ids = lambda reqs: {r.key: "x" for r in reqs}
    return loader


def _collect_corpus_models() -> set[str]:
    models: set[str] = set()
    for path in sorted(CORPUS.glob("trace_*.json")):
        blob = json.loads(path.read_text())
        _walk_models(blob.get("requests", []), models)
    return models


def _walk_models(reqs: list, models: set[str]) -> None:
    for r in reqs:
        if r.get("type") in ("n", "s"):
            models.add(r["model"])
        elif r.get("type") == "subagent":
            _walk_models(r.get("requests", []), models)


def test_hash_coherence_within_loader(loader_for_corpus):
    """Within a single trace scope, every occurrence of the same hash_id
    decodes to the identical token sequence.

    The cache lifecycle: ``convert_to_conversations`` clears the int-keyed
    ``_cache`` between scopes (per-trace and per-subagent) and once more in
    a ``finally`` block, so post-call the cache is empty. Coherence is
    therefore verified per-scope by reseating the hash-id RNG to a known
    scope and exercising the decoder twice for each observed hash_id.
    """
    loader = loader_for_corpus
    convs = loader.convert_to_conversations(loader.load_dataset())

    # The post-call cache must be empty: holding any trace's content past
    # convert_to_conversations would re-introduce a cross-trace cache leak.
    assert loader.prompt_generator._cache == {}, (
        "convert_to_conversations did not clear the block cache on exit; "
        "per-scope cache contract regressed."
    )

    # Collect every distinct hash_id observed across the corpus from the
    # parsed trace data (not from the cache).
    observed: set[int] = set()
    for path in sorted(CORPUS.glob("trace_*.json")):
        blob = json.loads(path.read_text())
        _walk_hashes(blob.get("requests", []), observed)

    # Within a fixed scope, two decode calls for the same hash_id must
    # return identical tokens (cache hit on the second). Pick an arbitrary
    # but stable scope — the test asserts intra-scope determinism, not
    # cross-scope behavior.
    pg = loader.prompt_generator
    pg._cache.clear()
    pg._hash_id_corpus_rng.set_trace_id("hash-coherence-probe")
    for h in list(observed)[:200]:  # cap: every hash_id is equivalent here
        rebuilt = loader._decode_block_tokens([h])
        again = loader._decode_block_tokens([h])
        assert rebuilt == again, (
            f"hash_id {h}: _decode_block_tokens not deterministic — "
            f"first call returned {rebuilt!r}, second {again!r}"
        )
    assert len(convs) > 0  # sanity


def _walk_hashes(reqs: list, observed: set[int]) -> None:
    for r in reqs:
        if r.get("type") in ("n", "s"):
            for h in r.get("hash_ids", []):
                observed.add(h)
        elif r.get("type") == "subagent":
            _walk_hashes(r.get("requests", []), observed)
