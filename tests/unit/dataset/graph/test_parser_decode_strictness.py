# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parser / decoder strictness for hand-authored native graph input.

Locks the strictness contract: unknown top-level YAML
keys and unknown record fields fail loudly instead of silently dropping data;
a node with both ``prompt`` and ``messages`` is rejected; non-finite delay
values are rejected at decode time; the dead synth-token prompt fabrication
is a clear error; a non-dict ``provenance.extra`` gets file/loc context.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest import param

from aiperf.dataset.graph.codecs import (
    decode_parsed_graph_msgpack,
    encode_parsed_graph_msgpack,
)
from aiperf.dataset.graph.decode import (
    GraphDecodeError,
    decode_edge,
    decode_graph,
    decode_node,
)
from aiperf.dataset.graph.models import (
    GraphRecord,
    LlmNode,
    ParsedGraph,
    StaticEdge,
    TraceRecord,
)
from aiperf.dataset.graph.parser import GraphParseError, parse_native


def _write_yaml(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "workload.yaml"
    p.write_text(text)
    return p


class TestUnknownTopLevelKeys:
    """G1: unknown top-level single-document keys must fail, not vanish."""

    def test_typo_trace_singular_rejected_with_suggestion(self, tmp_path: Path) -> None:
        with pytest.raises(GraphParseError, match=r"'trace'.*did you mean 'traces'"):
            parse_native(
                _write_yaml(
                    tmp_path,
                    """
graph:
  system: s
trace:
  - id: t1
""",
                )
            )

    def test_bare_unwrapped_graph_doc_rejected(self, tmp_path: Path) -> None:
        # nodes:/edges: at top level (missing the graph: wrapper) must fail
        # loudly rather than be discarded wholesale.
        with pytest.raises(GraphParseError, match=r"unknown top-level key.*'nodes'"):
            parse_native(
                _write_yaml(
                    tmp_path,
                    """
nodes:
  a:
    prompt: [{role: user, content: q}]
    output: a_out
edges:
  - {source: START, target: a}
""",
                )
            )

    def test_doc_with_none_of_the_known_sections_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(GraphParseError, match="contains none of"):
            parse_native(_write_yaml(tmp_path, "{}\n"))

    @pytest.mark.parametrize(
        "section",
        [
            param("mix", id="mix"),
            param("subgraphs", id="subgraphs"),
        ],
    )  # fmt: skip
    def test_retired_mix_subgraphs_sections_rejected(
        self, tmp_path: Path, section: str
    ) -> None:
        # mix:/subgraphs: were retired with their record kinds; they now fail
        # at the top-level-key gate instead of expanding to a kind that
        # _assemble rejects one step later.
        with pytest.raises(GraphParseError, match=rf"unknown top-level key.*{section}"):
            parse_native(
                _write_yaml(
                    tmp_path,
                    f"""
graph:
  system: s
{section}:
  child: {{}}
""",
                )
            )

    def test_known_sections_still_parse(self, tmp_path: Path) -> None:
        parsed = parse_native(
            _write_yaml(
                tmp_path,
                """
graph:
  system: s
traces:
  - id: t1
    messages:
      - {role: user, content: hi}
""",
            )
        )
        assert [t.id for t in parsed.traces] == ["t1"]


class TestUnknownRecordFields:
    """G2: LlmNode/GraphRecord/TraceRecord forbid unknown fields like edges do."""

    @pytest.mark.parametrize(
        ("yaml_text", "match"),
        [
            param(
                """
graph:
  nodes:
    a:
      prompt: [{role: user, content: q}]
      output: a_out
      streming: false
traces:
  - id: t1
""",
                "streming",
                id="node_field_typo",
            ),
            param(
                """
graph:
  verzion: "2.0"
  nodes:
    a:
      prompt: [{role: user, content: q}]
      output: a_out
traces:
  - id: t1
""",
                "verzion",
                id="graph_field_typo",
            ),
            param(
                """
graph:
  nodes:
    a:
      prompt: [{role: user, content: q}]
      output: a_out
traces:
  - id: t1
    replay_output: {}
""",
                "replay_output",
                id="trace_field_typo",
            ),
        ],
    )  # fmt: skip
    def test_unknown_fields_rejected_at_parse(
        self, tmp_path: Path, yaml_text: str, match: str
    ) -> None:
        with pytest.raises((GraphParseError, GraphDecodeError), match=match):
            parse_native(_write_yaml(tmp_path, yaml_text))

    def test_node_with_both_prompt_and_messages_rejected(self) -> None:
        with pytest.raises(GraphDecodeError, match="both 'prompt' and 'messages'"):
            decode_node(
                {
                    "prompt": [{"role": "user", "content": "a"}],
                    "messages": [{"role": "user", "content": "b"}],
                    "output": "o",
                },
                "n1",
            )

    def test_messages_alias_still_accepted_alone(self) -> None:
        node = decode_node(
            {"messages": [{"role": "user", "content": "a"}], "output": "o"}
        )
        assert isinstance(node, LlmNode)
        assert node.prompt == [{"role": "user", "content": "a"}]

    def test_msgpack_codec_round_trips_typed_structs(self) -> None:
        # forbid_unknown_fields must not break the cross-process msgpack path
        # (the encoder emits the node_type tag; the decoder must accept it).
        node = LlmNode(
            prompt=[{"role": "user", "content": "q"}],
            output="o",
            metadata={"trie": {"prompt_segment_ids": ["ab"]}},
        )
        pg = ParsedGraph(
            graph=GraphRecord(
                nodes={"a": node},
                edges=[
                    StaticEdge(source="START", target="a"),
                    StaticEdge(source="a", target="END"),
                ],
            ),
            traces=[TraceRecord(id="t1", replay_outputs={"a": {"o": "y"}})],
        )
        assert decode_parsed_graph_msgpack(encode_parsed_graph_msgpack(pg)) == pg


class TestNonFiniteDelaysAtDecode:
    """G3: +/-inf and NaN delay values must fail at decode, not hang at run."""

    @pytest.mark.parametrize(
        "field",
        [
            param("delay_after_predecessor_us", id="completion"),
            param("min_start_delay_us", id="min_start"),
            param("delay_after_predecessor_start_us", id="start_anchor"),
            param("delay_after_predecessor_first_token_us", id="first_token"),
        ],
    )  # fmt: skip
    def test_inf_edge_delay_rejected(self, field: str) -> None:
        raw: dict = {"source": "a", "target": "b", field: float("inf")}
        if field == "delay_after_predecessor_first_token_us":
            raw["delay_after_predecessor_start_us"] = 1.0
        with pytest.raises(GraphDecodeError, match=f"{field} must be finite"):
            decode_edge(raw)

    def test_nan_edge_delay_rejected(self) -> None:
        # NaN fails the msgspec ge=0 constraint; either way it must not decode.
        with pytest.raises(GraphDecodeError):
            decode_edge(
                {"source": "a", "target": "b", "min_start_delay_us": float("nan")}
            )

    def test_inf_node_min_start_delay_rejected(self) -> None:
        with pytest.raises(GraphDecodeError, match="min_start_delay_us must be finite"):
            decode_node(
                {"prompt": [], "output": "o", "min_start_delay_us": float("inf")},
                "n1",
            )

    def test_inf_delay_in_yaml_rejected_at_parse(self, tmp_path: Path) -> None:
        with pytest.raises((GraphParseError, GraphDecodeError), match="must be finite"):
            parse_native(
                _write_yaml(
                    tmp_path,
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
    - {source: a, target: b, delay_after_predecessor_us: .inf}
    - {source: b, target: END}
traces:
  - id: t1
""",
                )
            )

    def test_finite_delays_still_decode(self) -> None:
        edge = decode_edge(
            {"source": "a", "target": "b", "delay_after_predecessor_us": 1500.0}
        )
        assert edge.delay_after_predecessor_us == 1500.0


class TestSynthTokenFabricationRemoved:
    """G6: prompt-less nodes with expected.input_tokens fail clearly."""

    def test_promptless_node_with_expected_input_tokens_errors(self) -> None:
        with pytest.raises(
            GraphDecodeError,
            match=(
                r"graph\.nodes\.n1: node has no prompt; synth-token fabrication "
                r"is not supported"
            ),
        ):
            decode_node({"expected": {"input_tokens": 128}, "output": "o"}, "n1")

    def test_promptless_node_via_parse_names_the_node(self, tmp_path: Path) -> None:
        with pytest.raises(
            (GraphParseError, GraphDecodeError), match=r"graph\.nodes\.a.*no prompt"
        ):
            parse_native(
                _write_yaml(
                    tmp_path,
                    """
graph:
  nodes:
    a:
      expected: {input_tokens: 64}
      output: a_out
traces:
  - id: t1
""",
                )
            )


class TestProvenanceExtraCoercion:
    """G10: a non-dict provenance.extra fails with loc context, not a raw
    ValueError/TypeError."""

    @pytest.mark.parametrize(
        "bad_extra",
        [
            param("a string", id="str"),
            param(["vendor", "keys"], id="list"),
            param(7, id="int"),
        ],
    )  # fmt: skip
    def test_non_dict_extra_rejected_with_loc(self, bad_extra: object) -> None:
        with pytest.raises(
            GraphDecodeError, match=r"graph\.provenance\.extra: must be a mapping"
        ):
            decode_graph({"provenance": {"source": "weka_trace", "extra": bad_extra}})

    def test_vendor_keys_still_fold_into_extra(self) -> None:
        graph = decode_graph(
            {"provenance": {"source": "weka_trace", "tool": "x/1", "vendor_key": 3}}
        )
        assert graph.provenance.extra == {"vendor_key": 3}
