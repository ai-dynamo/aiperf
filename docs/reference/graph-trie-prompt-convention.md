<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Graph Trie-Route `prompt=[]` Convention

Internal developer reference for the inline-prompt convention on the
segment-trie IR route (the `dag_jsonl`, `dynamo`, and `weka` trace adapters).

## Convention

Trie-route adapters stamp an EMPTY inline prompt (`LlmNode.prompt == []`).
Prompt content lives ONLY in the run's content-addressed `SegmentPool`, reached
through the node's `metadata["trie"]["prompt_segment_ids"]` path. The inline
`prompt` field stays a required `LlmNode` field (authored native graphs still
carry real inline prompts), but on the trie route it is deliberately left empty.

Producers (all stamp `prompt=[]`):

- `src/aiperf/dataset/graph/adapters/dag_jsonl/lowering.py`
- `src/aiperf/dataset/graph/adapters/dynamo/trie_lowering.py` (`build_dynamo_llm_node`)
- `src/aiperf/dataset/graph/adapters/weka/trie_build.py` (`_build_llm_node`)

## Why

The inline prompt is dead weight on the trie route: it is one
`{"role", "content"}` dict per prompt message per node -- O(sum of path lengths)
across the graph, held for the entire store build -- while the deduplicated
`SegmentPool` already holds the content once. Nothing on the trie route reads
`node.prompt`:

- The unified store build (`segment_ir/store_builder.py`) drains only the
  segment pool plus the per-node trie envelope (`prompt_segment_ids`,
  the native dispatch fields, `stream`).
- The `graph_meta` sidecar (`graph/graph_meta_sidecar.py`, `strip_replay_text`)
  forces `prompt=[]` unconditionally.
- The worker materializes prompts from the mmap segment store, not the node.

## How consumers reach content

- Build/worker plane: walk `metadata["trie"]["prompt_segment_ids"]` against the
  store / `SegmentPool`.
- In-process debugging: `segment_pool.materialize(read_prompt_segment_ids(node))`
  (`segment_ir/envelope.py`).

New trie-route consumers MUST go through the segment path; reading `node.prompt`
on a trie graph yields `[]`.

## Invariant

The persisted store bytes are a function of `(segment pool, trie envelope)` only
-- never the inline `node.prompt`. This is pinned by
`tests/unit/dataset/test_dynamo_streaming_store_parity.py::test_store_bytes_independent_of_inline_prompt`,
which builds byte-identical stores through both the eager and streaming drains
from a real-content parse and a sentinel-prompt copy.
