# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate every shipped kv-cache-tester trace parses and holds expected invariants (opt-in via ``pytest -m slow``)."""

from pathlib import Path

import orjson
import pytest

from aiperf.dataset.loader.weka_trace_models import (
    WekaNormalRequest,
    WekaStreamingRequest,
    WekaSubagentEntry,
    WekaTrace,
)

CORPUS = Path(__file__).parents[4] / "artifacts" / "kv-cache-tester" / "traces"


pytestmark = pytest.mark.slow


@pytest.mark.skipif(not CORPUS.exists(), reason=f"corpus missing at {CORPUS}")
def test_all_corpus_files_parse():
    files = sorted(CORPUS.glob("trace_*.json"))
    assert len(files) > 0, "expected at least one trace in corpus"
    failures: list[tuple[str, str]] = []
    for path in files:
        try:
            WekaTrace.model_validate(orjson.loads(path.read_bytes()))
        except Exception as e:
            failures.append((path.name, repr(e)))
    assert not failures, f"{len(failures)} parse failures: {failures[:3]}"


@pytest.mark.skipif(not CORPUS.exists(), reason=f"corpus missing at {CORPUS}")
def test_corpus_invariants():
    for path in sorted(CORPUS.glob("trace_*.json")):
        t = WekaTrace.model_validate(orjson.loads(path.read_bytes()))
        assert t.hash_id_scope == "local", (
            f"{path.name}: unexpected hash_id_scope={t.hash_id_scope}"
        )
        for req in t.requests:
            if isinstance(req, WekaSubagentEntry):
                for inner in req.requests:
                    # Subagent inner requests are always non-streaming in this corpus.
                    assert isinstance(inner, WekaNormalRequest)
                    assert inner.model in req.models, (
                        f"{path.name}: subagent inner model {inner.model} not in "
                        f"declared models {req.models}"
                    )
            else:
                assert isinstance(req, WekaNormalRequest | WekaStreamingRequest)
