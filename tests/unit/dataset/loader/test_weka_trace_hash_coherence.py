# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hash-coherence smoke test over the kv-cache-tester corpus."""

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

    from tests.unit.dataset.loader.conftest import make_weka_run

    run = make_weka_run(model_names=sorted(_collect_corpus_models()))

    loader = WekaTraceLoader(filename=str(CORPUS), run=run)
    pg = MagicMock()
    pg._cache = {}
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
    """Within a single trace scope, every occurrence of the same hash_id"""
    loader = loader_for_corpus
    convs = loader.convert_to_conversations(loader.load_dataset())

    assert loader.prompt_generator._cache == {}, (
        "convert_to_conversations did not clear the block cache on exit; "
        "per-scope cache contract regressed."
    )

    observed: set[int] = set()
    for path in sorted(CORPUS.glob("trace_*.json")):
        blob = json.loads(path.read_text())
        _walk_hashes(blob.get("requests", []), observed)

    pg = loader.prompt_generator
    pg._cache.clear()
    pg._hash_id_corpus_rng.set_trace_id("hash-coherence-probe")
    for h in list(observed)[:200]:
        rebuilt = loader._decode_block_tokens([h])
        again = loader._decode_block_tokens([h])
        assert rebuilt == again, (
            f"hash_id {h}: _decode_block_tokens not deterministic — "
            f"first call returned {rebuilt!r}, second {again!r}"
        )
    assert len(convs) > 0


def _walk_hashes(reqs: list, observed: set[int]) -> None:
    for r in reqs:
        if r.get("type") in ("n", "s"):
            for h in r.get("hash_ids", []):
                observed.add(h)
        elif r.get("type") == "subagent":
            _walk_hashes(r.get("requests", []), observed)
