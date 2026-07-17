# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guard: weka ingest produces REAL-content prompts; no node holds a placeholder.

The segment-trie IR is the only weka prompt path: ``from_weka_trace`` synthesizes
the agentx-faithful multi-turn conversation into the :class:`SegmentPool` at
INGEST time and stamps each ``LlmNode`` with a ``prompt_segment_ids`` path that
materializes to that real content. This guard asserts every materialized prompt
message carries real, non-empty text -- never an empty or placeholder prompt.
"""

from pathlib import Path

from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace
from aiperf.dataset.graph.models import LlmNode

FIX = Path(__file__).parent / "fixtures" / "weka_min.json"
FIX_SUB = Path(__file__).parent / "fixtures" / "weka_subagent.json"

# Legacy transient placeholder the deleted delta-synthesis path stamped before
# filling real content. The trie path never emits it; assert it is absent.
_LEGACY_UNFILLED_PLACEHOLDER = "<aiperf-weka-unfilled-delta>"


def _materialized_prompts(parsed) -> list[list[dict]]:
    """Every LlmNode's materialized prompt messages on the trie top graph."""
    pool = parsed.segment_pool
    assert pool is not None, "trie ingest must attach a SegmentPool"
    out: list[list[dict]] = []
    for node in parsed.graph.nodes.values():
        if isinstance(node, LlmNode):
            out.append(pool.materialize(node.metadata["trie"]["prompt_segment_ids"]))
    return out


def test_default_ingest_yields_real_content_no_placeholder():
    parsed = from_weka_trace(str(FIX))
    prompts = _materialized_prompts(parsed)
    assert prompts, "expected at least one LlmNode prompt"

    all_msgs = [m for prompt in prompts for m in prompt]
    assert all_msgs, "expected materialized prompt messages"
    assert any(m.get("role") == "user" for m in all_msgs), (
        "expected at least one user message"
    )
    for m in all_msgs:
        content = m.get("content")
        assert isinstance(content, str) and content, "prompt content must be real text"
        assert _LEGACY_UNFILLED_PLACEHOLDER not in content, (
            "real-content ingest must NOT emit the transient placeholder"
        )


def test_default_ingest_multi_turn_content_is_real_no_placeholder():
    # The subagent fixture exercises multi-turn + subagent-inner requests, all
    # flattened into the trie top graph; every materialized prompt is real text.
    parsed = from_weka_trace(str(FIX_SUB))
    prompts = _materialized_prompts(parsed)
    assert len(prompts) > 1, "expected multiple turns flattened into the trie graph"
    for prompt in prompts:
        assert prompt, "every node must materialize a non-empty prompt"
        for m in prompt:
            assert isinstance(m["content"], str) and m["content"]
            assert _LEGACY_UNFILLED_PLACEHOLDER not in m["content"]
