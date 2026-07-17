# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native-graph lowering onto the unified segment store (Phase 0: static only).

Covers the canonicalization/interning happy paths (linear-chat shorthand,
explicit static prompts, per-trace init splices, content-block concatenation),
the NotImplementedError gates for un-lowerable constructs, and the
store-roundtrip parity: a lowered parse drained through
`build_unified_trie_store_interned` must materialize byte-correct messages.
"""

from pathlib import Path

import pytest
from pytest import param

from aiperf.dataset.graph.models import (
    ChannelSpec,
    ChannelType,
    GraphRecord,
    LlmNode,
    ParsedGraph,
    ReducerName,
    StaticEdge,
    TraceRecord,
)
from aiperf.dataset.graph.native_lowering import lower_native_to_unified
from aiperf.dataset.graph.parser import parse_native
from aiperf.dataset.graph.segment_ir.envelope import read_prompt_segment_ids
from aiperf.dataset.graph.segment_ir.store_builder import (
    build_unified_trie_store_interned,
)
from aiperf.dataset.graph_segment_unified_store import (
    GraphSegmentUnifiedBackingStore,
    GraphSegmentUnifiedClient,
)


def _write_yaml(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "workload.yaml"
    p.write_text(text)
    return p


def _materialized(parsed: ParsedGraph, graph: GraphRecord, node_id: str) -> list[dict]:
    node = graph.nodes[node_id]
    assert isinstance(node, LlmNode)
    path = read_prompt_segment_ids(node)
    assert path is not None
    assert parsed.segment_pool is not None
    return parsed.segment_pool.materialize(path)


class TestNativeLoweringHappyPaths:
    def test_linear_chat_shorthand_lowers_static_per_trace(
        self, tmp_path: Path
    ) -> None:
        parsed = parse_native(
            _write_yaml(
                tmp_path,
                """
graph:
  system: be brief
traces:
  - id: t1
    messages:
      - {role: user, content: hello}
  - id: t2
    messages:
      - {role: user, content: goodbye}
""",
            )
        )
        assert parsed.segment_pool is not None
        assert set(parsed.graphs) == {"t1", "t2"}
        assert [t.graph_ref for t in parsed.traces] == ["t1", "t2"]
        msgs = _materialized(parsed, parsed.graphs["t1"], "_llm")
        assert msgs == [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hello"},
        ]
        msgs2 = _materialized(parsed, parsed.graphs["t2"], "_llm")
        assert msgs2[1] == {"role": "user", "content": "goodbye"}

    def test_static_prompts_share_one_graph(self, tmp_path: Path) -> None:
        parsed = parse_native(
            _write_yaml(
                tmp_path,
                """
graph:
  nodes:
    a:
      prompt:
        - {role: user, content: "question one"}
      output: a_out
    b:
      prompt:
        - {role: user, content: "question two"}
      output: b_out
  edges:
    - {source: START, target: a}
    - {source: a, target: b}
    - {source: b, target: END}
traces:
  - id: t1
  - id: t2
""",
            )
        )
        assert parsed.segment_pool is not None
        assert parsed.graphs == {}
        assert all(t.graph_ref is None for t in parsed.traces)
        assert _materialized(parsed, parsed.graph, "a") == [
            {"role": "user", "content": "question one"}
        ]

    def test_content_block_concat_and_escape(self, tmp_path: Path) -> None:
        parsed = parse_native(
            _write_yaml(
                tmp_path,
                """
graph:
  nodes:
    a:
      prompt:
        - role: user
          content: ["Summarize: ", "@topic", " ", "@@literal"]
      output: a_out
  edges:
    - {source: START, target: a}
    - {source: a, target: END}
traces:
  - id: t1
    initial_state:
      topic: "graph stores"
""",
            )
        )
        msgs = _materialized(parsed, parsed.graphs["t1"], "a")
        assert msgs == [{"role": "user", "content": "Summarize: graph stores @literal"}]

    def test_trace_messages_shorthand_honored_with_explicit_nodes(
        self, tmp_path: Path
    ) -> None:
        # G5: traces[].messages is documented as equivalent to
        # initial_state.messages; it must be lifted for explicit-node graphs
        # too, not only on the linear-chat synthesis path.
        parsed = parse_native(
            _write_yaml(
                tmp_path,
                """
graph:
  nodes:
    a:
      prompt: ["@messages"]
      output: a_out
  edges:
    - {source: START, target: a}
    - {source: a, target: END}
traces:
  - id: t1
    messages:
      - {role: user, content: authored}
""",
            )
        )
        assert parsed.traces[0].messages is None
        assert parsed.traces[0].initial_state["messages"] == [
            {"role": "user", "content": "authored"}
        ]
        assert _materialized(parsed, parsed.graphs["t1"], "a") == [
            {"role": "user", "content": "authored"}
        ]

    def test_trace_messages_shorthand_defers_to_explicit_initial_state(
        self, tmp_path: Path
    ) -> None:
        parsed = parse_native(
            _write_yaml(
                tmp_path,
                """
graph:
  nodes:
    a:
      prompt: ["@messages"]
      output: a_out
  edges:
    - {source: START, target: a}
    - {source: a, target: END}
traces:
  - id: t1
    messages:
      - {role: user, content: shorthand}
    initial_state:
      messages:
        - {role: user, content: explicit}
""",
            )
        )
        assert _materialized(parsed, parsed.graphs["t1"], "a") == [
            {"role": "user", "content": "explicit"}
        ]

    def test_self_written_channel_reads_init(self, tmp_path: Path) -> None:
        parsed = parse_native(
            _write_yaml(
                tmp_path,
                """
graph:
  state:
    hist: {type: messages, reducer: add_messages}
  nodes:
    a:
      prompt: ["@hist"]
      output: hist
  edges:
    - {source: START, target: a}
    - {source: a, target: END}
traces:
  - id: t1
    initial_state:
      hist:
        - {role: user, content: hi}
""",
            )
        )
        msgs = _materialized(parsed, parsed.graphs["t1"], "a")
        assert msgs == [{"role": "user", "content": "hi"}]

    @pytest.mark.asyncio
    async def test_store_roundtrip_materializes_canonical_messages(
        self, tmp_path: Path
    ) -> None:
        parsed = parse_native(
            _write_yaml(
                tmp_path,
                """
graph:
  system: sys
traces:
  - id: t1
    messages:
      - {role: user, content: hello}
""",
            )
        )
        store = GraphSegmentUnifiedBackingStore(
            base_path=tmp_path, benchmark_id="bench"
        )
        catalog = await build_unified_trie_store_interned(parsed, store)
        assert set(catalog) == {"t1"}
        ordinal = catalog["t1"]["_llm"]
        client = GraphSegmentUnifiedClient(tmp_path, "bench").open()
        import orjson

        envelope = orjson.loads(client.get_node_envelope("t1", ordinal))
        assert client.materialize_handles(envelope["handles"]) == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]


class TestNativeLoweringGates:
    @pytest.mark.parametrize(
        ("yaml_text", "match"),
        [
            param(
                """
graph:
  nodes:
    a:
      prompt: [{role: user, content: q}]
      output: shared
    b:
      prompt: ["@shared"]
      output: b_out
  edges:
    - {source: START, target: a}
    - {source: START, target: b}
    - {source: a, target: END}
    - {source: b, target: END}
traces:
  - id: t1
""",
                "concurrent with",
                id="non_ancestor_writer_gated",
            ),
            param(
                """
graph:
  nodes:
    a:
      prompt: ["plain text at array level"]
      output: a_out
  edges:
    - {source: START, target: a}
    - {source: a, target: END}
traces:
  - id: t1
""",
                "top-level string items",
                id="literal_array_item",
            ),
            param(
                """
graph:
  nodes:
    a:
      prompt:
        - role: user
          content: [{synth_tokens: 5}]
      output: a_out
  edges:
    - {source: START, target: a}
    - {source: a, target: END}
traces:
  - id: t1
""",
                "non-string content blocks",
                id="directive_block",
            ),
            param(
                """
graph:
  nodes:
    a:
      prompt: ["@hist"]
      output: a_out
  edges:
    - {source: START, target: a}
    - {source: a, target: END}
traces:
  - id: t1
""",
                "init-seeded content",
                id="missing_init",
            ),
            param(
                """
graph:
  nodes:
    a:
      prompt: []
      output: a_out
  edges:
    - {source: START, target: a}
    - {source: a, target: END}
traces:
  - id: t1
""",
                "empty prompt",
                id="empty_prompt",
            ),
            param(
                """
graph:
  nodes:
    a:
      prompt: ["@hist"]
      output: a_out
  edges:
    - {source: START, target: a}
    - {source: a, target: END}
traces:
  - id: t1
    initial_state:
      hist: "not a list"
""",
                "list of message dicts",
                id="messages_init_not_list",
            ),
            param(
                """
graph:
  nodes:
    a:
      prompt:
        - role: user
          content: ["@topic"]
      output: a_out
  edges:
    - {source: START, target: a}
    - {source: a, target: END}
traces:
  - id: t1
    initial_state:
      topic: [not, a, string]
""",
                "string initial_state value",
                id="text_init_not_str",
            ),
            param(
                """
graph:
  nodes:
    a:
      prompt:
        - {role: user, content: q, name: alice}
      output: a_out
  edges:
    - {source: START, target: a}
    - {source: a, target: END}
traces:
  - id: t1
""",
                "not representable in the unified store",
                id="extra_message_keys",
            ),
            param(
                """
graph:
  nodes:
    a:
      prompt:
        - {role: user, content: q}
      output: a_out
      metadata:
        replay_reducers: {a_out: overwrite}
  edges:
    - {source: START, target: a}
    - {source: a, target: END}
traces:
  - id: t1
""",
                "reducer overrides",
                id="replay_reducers_metadata",
            ),
            param(
                """
graph:
  nodes:
    a:
      prompt: ["@hist"]
      output: a_out
  edges:
    - {source: START, target: a}
    - {source: a, target: END}
traces:
  - id: t1
    initial_state:
      hist:
        - {role: user, content: hi, id: m1}
""",
                "not representable in the unified store",
                id="extra_init_message_keys",
            ),
            param(
                # G12: a rule-55-invalid edge (first-token anchor with no
                # dispatch fallback) must fail loudly at lowering, not be
                # silently lowered as a completion edge.
                """
graph:
  nodes:
    a:
      prompt: [{role: user, content: q}]
      output: a_out
    b:
      prompt: [{role: user, content: q}]
      output: b_out
  edges:
    - {source: START, target: a}
    - {source: a, target: b, delay_after_predecessor_first_token_us: 1000}
    - {source: b, target: END}
traces:
  - id: t1
""",
                r"graph\.edges\[a->b\].*sets no delay_after_predecessor_start_us",
                id="first_token_anchor_without_fallback",
            ),
        ],
    )  # fmt: skip
    def test_parse_native_gates(
        self, tmp_path: Path, yaml_text: str, match: str
    ) -> None:
        with pytest.raises(NotImplementedError, match=match):
            parse_native(_write_yaml(tmp_path, yaml_text))

    def test_retired_stream_reducer_rejected_at_parse(self, tmp_path: Path) -> None:
        from aiperf.dataset.graph.decode import GraphDecodeError

        with pytest.raises(GraphDecodeError, match="stream_passthrough"):
            parse_native(
                _write_yaml(
                    tmp_path,
                    """
graph:
  state:
    s: {type: text, reducer: stream_passthrough}
  nodes:
    a:
      prompt:
        - {role: user, content: q}
      output: a_out
traces:
  - id: t1
""",
                )
            )

    def test_unknown_node_kind_rejected_at_decode(self) -> None:
        from aiperf.dataset.graph.decode import GraphDecodeError, decode_node

        with pytest.raises(GraphDecodeError, match="unknown node node_type"):
            decode_node({"node_type": "spawn", "graph_ref": "child"})

    def test_conditional_edge_rejected_at_decode(self) -> None:
        from aiperf.dataset.graph.decode import GraphDecodeError, decode_edge

        with pytest.raises(GraphDecodeError, match="branches"):
            decode_edge({"source": "a", "branches": {"x": "END"}})

    def test_subgraph_record_rejected_at_parse(self, tmp_path: Path) -> None:
        from aiperf.dataset.graph.parser import GraphParseError

        path = tmp_path / "sub.jsonl"
        path.write_text(
            '{"kind":"graph","nodes":{}}\n'
            '{"kind":"subgraph","name":"child","nodes":{}}\n'
        )
        with pytest.raises(GraphParseError, match="unknown kind"):
            parse_native(path)

    def test_refs_without_traces_gated(self) -> None:
        node = LlmNode(
            prompt=["@hist"],
            output="a_out",
        )
        graph = GraphRecord(
            nodes={"a": node},
            state={
                "hist": ChannelSpec(
                    type=ChannelType.MESSAGES, reducer=ReducerName.ADD_MESSAGES
                )
            },
            edges=[
                StaticEdge(source="START", target="a"),
                StaticEdge(source="a", target="END"),
            ],
        )
        parsed = ParsedGraph(graph=graph, traces=[])
        with pytest.raises(NotImplementedError, match="no trace records"):
            lower_native_to_unified(parsed)

    def test_duplicate_trace_ids_gated(self) -> None:
        node = LlmNode(prompt=["@hist"], output="a_out")
        graph = GraphRecord(
            nodes={"a": node},
            edges=[
                StaticEdge(source="START", target="a"),
                StaticEdge(source="a", target="END"),
            ],
        )
        init = {"hist": [{"role": "user", "content": "x"}]}
        traces = [
            TraceRecord(id="t1", initial_state=init),
            TraceRecord(id="t1", initial_state=init),
        ]
        parsed = ParsedGraph(graph=graph, traces=traces)
        with pytest.raises(NotImplementedError, match="duplicate trace ids"):
            lower_native_to_unified(parsed)


class TestSegmentIdDedup:
    def test_shared_prefix_dedups_across_traces(self, tmp_path: Path) -> None:
        parsed = parse_native(
            _write_yaml(
                tmp_path,
                """
graph:
  system: same system
traces:
  - id: t1
    messages:
      - {role: user, content: alpha}
  - id: t2
    messages:
      - {role: user, content: beta}
""",
            )
        )
        assert parsed.segment_pool is not None
        p1 = read_prompt_segment_ids(parsed.graphs["t1"].nodes["_llm"])
        p2 = read_prompt_segment_ids(parsed.graphs["t2"].nodes["_llm"])
        assert p1 is not None and p2 is not None
        assert p1[0] == p2[0]
        assert p1[1] != p2[1]


class TestNativeModelFallback:
    """Native nodes carry no per-node model; the run ``--model`` must be folded
    into the wire body on both materialize paths, else the server 422s on a
    missing ``model`` field (a Phase-0 gap the mock-server E2E surfaced)."""

    @pytest.mark.asyncio
    async def test_run_model_folded_when_node_has_none(self, tmp_path: Path) -> None:
        from aiperf.common.models import EndpointInfo
        from aiperf.graph.worker_materialize import (
            materialize_graph_request_unified,
            materialize_graph_request_unified_bytes,
        )

        parsed = parse_native(
            _write_yaml(
                tmp_path,
                """
graph:
  system: sys
traces:
  - id: t1
    messages:
      - {role: user, content: hello}
""",
            )
        )
        store = GraphSegmentUnifiedBackingStore(base_path=tmp_path, benchmark_id="m")
        catalog = await build_unified_trie_store_interned(parsed, store)
        ordinal = catalog["t1"]["_llm"]
        client = GraphSegmentUnifiedClient(tmp_path, "m").open()
        endpoint = EndpointInfo(type="chat", streaming=False, extra=[])

        # dict path
        payload = materialize_graph_request_unified(
            client, "t1", ordinal, "profiling", default_model="run-model"
        )
        assert payload["model"] == "run-model"

        # bytes path
        import orjson

        body, model, _ = materialize_graph_request_unified_bytes(
            client,
            "t1",
            ordinal,
            "profiling",
            endpoint=endpoint,
            default_model="run-model",
        )
        assert model == "run-model"
        assert orjson.loads(body)["model"] == "run-model"

        # No default supplied => no model injected (weka/dynamo stamp their own).
        bare = materialize_graph_request_unified(client, "t1", ordinal, "profiling")
        assert "model" not in bare
        client.close()
