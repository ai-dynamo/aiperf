# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter registration, detection, and parse-seam tests for ``dag_jsonl``.

Covers the adapter selection surface on top of the (already tested) tree
expansion + lowering core:

* ``DagJsonlGraphAdapter.can_load`` -- a strict, bounded, never-raising sniff
  that claims real dag files and rejects dynamo / native / mooncake / empty
  inputs (kept mutually exclusive with dynamo's sniff).
* plugin-registry resolution of the new ``graph_adapter.dag_jsonl`` entry.
* autodetect exclusion (a dag file is NEVER auto-claimed) paired with explicit
  ``--graph-format dag_jsonl`` forcing, driven through a REAL ``BenchmarkRun``
  (MagicMock auto-creates attribute paths and would hide config drift).
* determinism: repeated lowering of the same dag file produces byte-identical
  build catalogs (the parse is a pure function of file + run config).
"""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest
from pytest import param

from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.graph.adapters.dag_jsonl.trace import (
    DagJsonlGraphAdapter,
    _assert_dag_zero_arrival_offsets,
    from_dag_jsonl,
)
from aiperf.dataset.graph.graph_path_catalog import build_graph_path_catalog
from aiperf.dataset.graph.models import GraphRecord, LlmNode, ParsedGraph
from aiperf.dataset.graph.workload_detect import (
    _detect_graph_workload_format,
    is_graph_workload_path,
    parse_graph_workload,
)
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType
from tests.unit.conftest import make_run_from_cli

FIXTURES_DIR = Path(__file__).parents[5] / "fixtures" / "dag"

ALL_DAG_FIXTURES = [
    "background_fork.dag.jsonl",
    "bg_fork_fanout.dag.jsonl",
    "bg_fork_nested.dag.jsonl",
    "bg_fork_with_spawn_join.dag.jsonl",
    "full.dag.jsonl",
    "multi_root_single_turn.dag.jsonl",
    "small.dag.jsonl",
    "spawn_minimal.dag.jsonl",
]


def _write_jsonl(tmp_path: Path, name: str, lines: list[dict]) -> Path:
    path = tmp_path / name
    path.write_bytes(b"\n".join(orjson.dumps(line) for line in lines))
    return path


# --- (a) can_load sniff -----------------------------------------------------


class TestCanLoad:
    @pytest.mark.parametrize("fixture", ALL_DAG_FIXTURES)
    def test_true_on_real_dag_fixtures(self, fixture: str) -> None:
        assert DagJsonlGraphAdapter.can_load(FIXTURES_DIR / fixture) is True

    def test_true_on_bare_jsonl_suffix(self, tmp_path: Path) -> None:
        path = _write_jsonl(
            tmp_path,
            "conv.jsonl",
            [{"session_id": "root", "turns": [{"messages": [{"role": "user"}]}]}],
        )
        assert DagJsonlGraphAdapter.can_load(path) is True

    def test_false_on_dynamo_first_line(self, tmp_path: Path) -> None:
        # A genuine dynamo trace line -- claimed by DynamoTraceAdapter, never us.
        path = _write_jsonl(
            tmp_path,
            "dynamo.jsonl",
            [
                {
                    "schema": "dynamo.request.trace.v1",
                    "event_type": "request_end",
                    "agent_context": {},
                }
            ],
        )
        assert DagJsonlGraphAdapter.can_load(path) is False

    def test_false_on_dynamo_discriminator_even_with_dag_keys(
        self, tmp_path: Path
    ) -> None:
        # Mutual-exclusivity guard: the dynamo discriminator wins even if the
        # (adversarial) line also carries session_id/turns.
        path = _write_jsonl(
            tmp_path,
            "hybrid.jsonl",
            [
                {
                    "schema": "dynamo.request.trace.v1",
                    "event_type": "request_end",
                    "session_id": "root",
                    "turns": [],
                }
            ],
        )
        assert DagJsonlGraphAdapter.can_load(path) is False

    def test_false_on_dynamo_sink_envelope(self, tmp_path: Path) -> None:
        # Dynamo file sinks wrap each record in {"timestamp", "event"}; unwrap
        # before matching the discriminator.
        path = _write_jsonl(
            tmp_path,
            "wrapped.jsonl",
            [
                {
                    "timestamp": 12,
                    "event": {
                        "schema": "dynamo.request.trace.v1",
                        "event_type": "request_end",
                    },
                }
            ],
        )
        assert DagJsonlGraphAdapter.can_load(path) is False

    def test_false_on_native_yaml(self, tmp_path: Path) -> None:
        # Native graph workloads are .yaml/.yml -- the dag sniff only claims
        # .jsonl, so a native file (even one carrying graph-shaped content) is
        # rejected on suffix.
        path = tmp_path / "graph.yaml"
        path.write_text("version: '2.0'\nnodes: {}\nedges: []\n")
        assert DagJsonlGraphAdapter.can_load(path) is False

    def test_false_on_mooncake_style_jsonl(self, tmp_path: Path) -> None:
        path = _write_jsonl(
            tmp_path,
            "mooncake.jsonl",
            [{"timestamp": 0, "input_length": 10, "output_length": 5, "hash_ids": [1]}],
        )
        assert DagJsonlGraphAdapter.can_load(path) is False

    @pytest.mark.parametrize(
        "name, content",
        [
            param("empty.jsonl", b"", id="empty"),
            param("blank_lines.jsonl", b"\n\n  \n", id="blank-lines-only"),
            param("garbage.jsonl", b"not json at all\n", id="invalid-json"),
            param(
                "missing_turns.jsonl",
                b'{"session_id": "root"}\n',
                id="missing-turns-key",
            ),
            param(
                "missing_sid.jsonl",
                b'{"turns": [{"messages": []}]}\n',
                id="missing-session-id-key",
            ),
        ],
    )  # fmt: skip
    def test_false_on_malformed_inputs(
        self, tmp_path: Path, name: str, content: bytes
    ) -> None:
        path = tmp_path / name
        path.write_bytes(content)
        assert DagJsonlGraphAdapter.can_load(path) is False

    def test_false_on_missing_file(self, tmp_path: Path) -> None:
        assert DagJsonlGraphAdapter.can_load(tmp_path / "nope.jsonl") is False

    def test_false_on_wrong_suffix(self, tmp_path: Path) -> None:
        path = _write_jsonl(
            tmp_path,
            "conv.json",
            [{"session_id": "root", "turns": [{"messages": []}]}],
        )
        assert DagJsonlGraphAdapter.can_load(path) is False


# --- (b) plugin-registry resolution ----------------------------------------


class TestRegistry:
    def test_registry_resolves_adapter_class(self) -> None:
        cls = plugins.get_class(PluginType.GRAPH_ADAPTER, "dag_jsonl")
        assert cls is DagJsonlGraphAdapter

    def test_adapter_parse_produces_parsed_graph(self) -> None:
        parsed = DagJsonlGraphAdapter.parse(FIXTURES_DIR / "small.dag.jsonl")
        assert isinstance(parsed, ParsedGraph)
        assert set(parsed.graphs) == {"root"}


# --- (c) autodetect exclusion + explicit --graph-format forcing -------------


def _dag_run(fixture: str, **cli_overrides):
    return _dag_run_from_path(FIXTURES_DIR / fixture, **cli_overrides)


def _dag_run_from_path(path: Path, **cli_overrides):
    cfg = CLIConfig(
        model_names=["test-model"],
        input_file=str(path),
        tokenizer_name="builtin",
        **cli_overrides,
    )
    return make_run_from_cli(cfg)


class TestSelectionSeam:
    @pytest.mark.parametrize("fixture", ALL_DAG_FIXTURES)
    def test_autodetect_never_claims_dag_file(self, fixture: str) -> None:
        path = FIXTURES_DIR / fixture
        # Excluded from autodetect: no adapter (dag_jsonl skipped, weka/dynamo
        # reject .jsonl dag content) claims the file.
        assert _detect_graph_workload_format(path) is None
        assert is_graph_workload_path(path) is False

    def test_forced_graph_format_returns_parsed_graph(self) -> None:
        run = _dag_run("small.dag.jsonl", graph_format="dag_jsonl")
        parsed = parse_graph_workload(run, FIXTURES_DIR / "small.dag.jsonl")
        assert isinstance(parsed, ParsedGraph)
        assert set(parsed.graphs) == {"root"}

    def test_forced_graph_format_threads_run_streaming(self) -> None:
        run = _dag_run("small.dag.jsonl", graph_format="dag_jsonl", streaming=True)
        parsed = parse_graph_workload(run, FIXTURES_DIR / "small.dag.jsonl")
        node = next(iter(parsed.graph.nodes.values()))
        assert node.streaming is True

    def test_forced_graph_format_threads_streaming_false(self) -> None:
        # Discriminating counterpart to the streaming=True case above: a
        # hardcoded ``True`` would survive that test but fail here. Every node's
        # body ``stream`` AND its ``LlmNode.streaming`` must track the resolved
        # run flag.
        run = _dag_run("small.dag.jsonl", graph_format="dag_jsonl", streaming=False)
        parsed = parse_graph_workload(run, FIXTURES_DIR / "small.dag.jsonl")
        nodes = [
            node
            for record in (parsed.graph, *parsed.graphs.values())
            for node in record.nodes.values()
        ]
        assert nodes
        for node in nodes:
            assert node.streaming is False

    def test_default_model_stamped_when_turn_has_no_authored_model(
        self, tmp_path: Path
    ) -> None:
        # Same-run determinism can't catch a wrong-but-consistent model knob.
        # A turn with NO authored ``model`` must inherit the run's resolved
        # primary model (``test-model`` here), not some hardcoded default.
        path = _write_jsonl(
            tmp_path,
            "no_model.dag.jsonl",
            [
                {
                    "session_id": "root",
                    "turns": [{"messages": [{"role": "user", "content": "hi"}]}],
                }
            ],
        )
        run = _dag_run_from_path(path, graph_format="dag_jsonl")
        parsed = parse_graph_workload(run, path)
        node = next(iter(parsed.graph.nodes.values()))
        assert node.model == "test-model"

    def test_forced_graph_format_threads_extra_inputs(self, tmp_path: Path) -> None:
        # The dag branch threads ``endpoint.extra`` (--extra-inputs) into the
        # lowering so run-level vendor keys are folded into extra_body
        # at parse (endpoint extras first, turn extras last; turn ``extra``
        # wins on overlap) and every node carries the
        # ``endpoint_extra_applied`` stamp for the worker's skip.
        path = _write_jsonl(
            tmp_path,
            "extras.dag.jsonl",
            [
                {
                    "session_id": "root",
                    "turns": [
                        {
                            "messages": [{"role": "user", "content": "hi"}],
                            "extra": {"temperature": 0.5},
                        }
                    ],
                }
            ],
        )
        run = _dag_run_from_path(
            path,
            graph_format="dag_jsonl",
            extra_inputs=["temperature:0.9", "min_p:0.05"],
        )
        parsed = parse_graph_workload(run, path)
        node = parsed.graph.nodes["root:0"]
        assert list(node.extra_body) == [
            "temperature",
            "min_p",
        ]
        assert node.extra_body["temperature"] == 0.5, (
            "turn extra must win over --extra-inputs on overlap"
        )
        assert node.extra_body["min_p"] == 0.05
        assert node.metadata["dispatch"]["endpoint_extra_applied"] is True

    def test_inter_turn_delay_cap_clamps_sequential_edge(self, tmp_path: Path) -> None:
        # ``--inter-turn-delay-cap-seconds`` (seconds) clamps the authored
        # per-turn ``delay`` (milliseconds) before it becomes the sequential
        # edge's ``delay_after_predecessor_us``. Cap 2s => 2_000_000us; the
        # uncapped 60s delay would instead surface as 60_000_000us, so this
        # discriminates a missing/ignored cap.
        path = _write_jsonl(
            tmp_path,
            "big_delay.dag.jsonl",
            [
                {
                    "session_id": "root",
                    "turns": [
                        {"messages": [{"role": "user", "content": "t0"}]},
                        {
                            "messages": [{"role": "user", "content": "t1"}],
                            "delay": 60000.0,
                        },
                    ],
                }
            ],
        )
        run = _dag_run_from_path(
            path, graph_format="dag_jsonl", inter_turn_delay_cap_seconds=2.0
        )
        parsed = parse_graph_workload(run, path)
        edge = next(
            e
            for e in parsed.graph.edges
            if e.source == "root:0" and e.target == "root:1"
        )
        assert edge.delay_after_predecessor_us == 2_000_000.0


class TestArrivalOffsetGuard:
    """The t*/dynamic-slot gate carves out graphs whose every node carries an
    explicit-zero ``arrival_offset_us`` -- the shape dag lowering stamps.
    ``DagJsonlGraphAdapter.parse`` wraps ``from_dag_jsonl`` in
    ``_assert_dag_zero_arrival_offsets`` to hold that invariant at the one
    seam production dispatches through; these tests exercise the guard
    directly and through the full run-parse seam."""

    def _one_node_graph(self, arrival_offset_us: int) -> ParsedGraph:
        node = LlmNode(prompt=[], output="n_out", arrival_offset_us=arrival_offset_us)
        record = GraphRecord(nodes={"n": node})
        return ParsedGraph(graph=record, graphs={"root": record})

    def test_guard_passes_on_zero_offsets(self) -> None:
        _assert_dag_zero_arrival_offsets(self._one_node_graph(0))

    def test_guard_raises_on_nonzero_offset(self) -> None:
        with pytest.raises(ValueError, match="arrival_offset_us"):
            _assert_dag_zero_arrival_offsets(self._one_node_graph(5))

    def test_parse_seam_invokes_guard(self, monkeypatch) -> None:
        # Prove the seam actually calls the guard: force ``from_dag_jsonl`` to
        # return a nonzero-offset graph and assert the adapter's parse rejects
        # it. Through the registry seam the guard's ValueError surfaces as
        # GraphParseError (a ValueError subclass; message text preserved).
        bad = self._one_node_graph(7)
        monkeypatch.setattr(
            "aiperf.dataset.graph.adapters.dag_jsonl.trace.from_dag_jsonl",
            lambda *args, **kwargs: bad,
        )
        run = _dag_run("small.dag.jsonl", graph_format="dag_jsonl")
        with pytest.raises(ValueError, match="arrival_offset_us"):
            parse_graph_workload(run, FIXTURES_DIR / "small.dag.jsonl")


# --- (d) determinism --------------------------------------------------------


class TestDeterminism:
    def test_default_knobs_deterministic(self) -> None:
        # ``from_dag_jsonl`` with protocol defaults is a pure function of file.
        path = FIXTURES_DIR / "full.dag.jsonl"
        first = from_dag_jsonl(str(path))
        second = from_dag_jsonl(str(path))
        assert build_graph_path_catalog(first) == build_graph_path_catalog(second)
