# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tightened weka workload auto-detection predicate tests.

Pins the false-positive surface called out in ``adv2-integration.md`` F3:

* ``_is_weka_trace_object`` must require the genuine :class:`WekaTrace`
  discriminator strictly and REJECT objects carrying foreign/contradictory
  top-level keys (a mooncake/sharegpt-shaped object that happens to also carry
  the five weka keys must NOT be detected), and
* ``_looks_like_hf_dataset_id`` must NOT treat an arbitrary ``org/name`` string
  as a weka HuggingFace dataset id -- only repo ids that carry the weka corpus
  marker are routed to ``datasets.load_dataset``.

Genuine weka inputs (the existing fixtures and the canonical weka HF corpus id)
MUST still be detected so :func:`resolve_graph_workload` stays correct.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pytest import param

from aiperf.dataset.graph.adapters.weka.trace import (
    WekaTraceAdapter,
    _is_weka_trace_object,
    _looks_like_hf_dataset_id,
)

FIXTURES = Path(__file__).parent / "fixtures"
WEKA_MIN = FIXTURES / "weka_min.json"

# A minimal but genuine weka trace object (the on-disk single-document shape).
_GENUINE_WEKA_OBJECT: dict = {
    "id": "trace_genuine",
    "models": ["claude-opus-4-5-20251101"],
    "block_size": 64,
    "hash_id_scope": "local",
    "requests": [
        {"t": 0.0, "type": "n", "model": "m", "in": 10, "out": 5},
    ],
}


# --------------------------------------------------------------------------- #
# _is_weka_trace_object: genuine still detected                               #
# --------------------------------------------------------------------------- #
def test_genuine_weka_object_detected():
    assert _is_weka_trace_object(_GENUINE_WEKA_OBJECT) is True


def test_genuine_weka_object_with_optional_fields_detected():
    # tool_tokens / system_tokens / totals are legitimate WekaTrace fields.
    doc = dict(_GENUINE_WEKA_OBJECT)
    doc["tool_tokens"] = 12
    doc["system_tokens"] = 8
    doc["totals"] = {"requests": 1}
    assert _is_weka_trace_object(doc) is True


def test_genuine_weka_jsonl_kind_marker_detected():
    # The native JSONL writer stamps a ``kind`` marker on serialized rows; a
    # weka object that also carries ``kind`` must still be accepted.
    doc = dict(_GENUINE_WEKA_OBJECT)
    doc["kind"] = "trace"
    assert _is_weka_trace_object(doc) is True


def test_global_hash_scope_detected():
    # 'global' is a supported hash_id_scope (cross-trace shared hash
    # namespace), so a global-scope trace must detect as weka.
    doc = {**_GENUINE_WEKA_OBJECT, "hash_id_scope": "global"}
    assert _is_weka_trace_object(doc) is True


def test_weka_min_fixture_file_detected():
    assert WekaTraceAdapter.can_load(WEKA_MIN) is True


# --------------------------------------------------------------------------- #
# _is_weka_trace_object: near-misses and foreign-shaped objects rejected      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "doc",
    [
        param(
            {**_GENUINE_WEKA_OBJECT, "hash_id_scope": "cluster"},
            id="unrecognized_hash_scope",
        ),
        param(
            {**_GENUINE_WEKA_OBJECT, "conversations": [{"from": "human"}]},
            id="sharegpt_foreign_key",
        ),
        param(
            {**_GENUINE_WEKA_OBJECT, "mooncake_field": True, "timestamp": 1},
            id="mooncake_foreign_keys",
        ),
        param(
            {
                "id": "x",
                "models": ["m"],
                "block_size": 64,
                "hash_id_scope": "local",
                "requests": [],
                "prompt": "synthetic-style",
                "completion": "x",
            },
            id="synthetic_foreign_keys",
        ),
    ],
)  # fmt: skip
def test_near_miss_and_foreign_objects_not_detected(doc: dict):
    assert _is_weka_trace_object(doc) is False


@pytest.mark.parametrize(
    "doc",
    [
        param({"id": "x", "models": ["m"], "block_size": 64}, id="missing_keys"),
        param("not-a-dict", id="not_a_dict"),
        param(
            {**_GENUINE_WEKA_OBJECT, "block_size": True},
            id="bool_block_size",
        ),
        param(
            {**_GENUINE_WEKA_OBJECT, "models": "claude"},
            id="models_not_list",
        ),
    ],
)  # fmt: skip
def test_malformed_objects_not_detected(doc: object):
    assert _is_weka_trace_object(doc) is False


def test_foreign_keyed_object_not_detected_via_file(tmp_path: Path):
    # End-to-end through the file sniff: a JSON file carrying the five weka keys
    # PLUS foreign keys must NOT route to the graph pipeline.
    f = tmp_path / "mooncake.json"
    doc = {**_GENUINE_WEKA_OBJECT, "mooncake_field": True, "conversations": []}
    f.write_text(json.dumps(doc))
    assert WekaTraceAdapter.can_load(f) is False


# --------------------------------------------------------------------------- #
# _looks_like_hf_dataset_id: only weka-marked repo ids match                  #
# --------------------------------------------------------------------------- #
def test_genuine_weka_hf_id_detected():
    assert _looks_like_hf_dataset_id("semianalysisai/cc-traces-weka-061526") is True


@pytest.mark.parametrize(
    "repo_id",
    [
        param("meta-llama/Llama-3", id="meta_llama"),
        param("my-team/notes", id="team_notes"),
        param("a/b", id="short"),
        param("openai/gpt", id="openai_gpt"),
        param("anthropic/claude", id="anthropic_claude"),
    ],
)  # fmt: skip
def test_arbitrary_org_name_not_detected_as_hf_weka_id(repo_id: str):
    assert _looks_like_hf_dataset_id(repo_id) is False


def test_existing_parent_dir_not_detected_as_hf_weka_id(tmp_path, monkeypatch):
    """A typo'd relative path under an EXISTING local dir is not an HF id.

    ``traces/weka-061526`` (weka-marked, org/name-shaped, non-existent) under a
    real ``traces/`` directory is a local-path mistake; routing it to
    ``datasets.load_dataset`` buries the typo behind a confusing HF error.
    """
    (tmp_path / "traces").mkdir()
    monkeypatch.chdir(tmp_path)
    assert _looks_like_hf_dataset_id("traces/weka-061526") is False


@pytest.mark.parametrize(
    "candidate",
    [
        param("./traces/weka-061526", id="dot_slash_prefix"),
        param("traces/weka-061526/", id="trailing_slash"),
        param("a/b/weka-061526", id="multi_component_path"),
    ],
)  # fmt: skip
def test_pathlike_markers_not_detected_as_hf_weka_id(candidate: str):
    assert _looks_like_hf_dataset_id(candidate) is False


def test_hf_load_failure_error_names_both_interpretations(monkeypatch):
    """When the weka HF heuristic fires and the load fails, the error must
    present BOTH readings (missing local path vs bad HF repo id)."""
    import sys
    import types

    from aiperf.dataset.graph.adapters.weka.trace import (
        WekaTraceAdapterError,
        _load_hf_rows,
    )

    def load_dataset(*_args, **_kwargs):  # noqa: ANN001
        raise FileNotFoundError("dataset not found on the hub")

    monkeypatch.setitem(
        sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset)
    )

    rows = _load_hf_rows("no-such-org/cc-weka-typo", split="train", revision=None)
    with pytest.raises(WekaTraceAdapterError) as excinfo:
        next(rows)
    message = str(excinfo.value)
    assert "no such local file or directory" in message
    assert "HuggingFace" in message


# --------------------------------------------------------------------------- #
# _detect_graph_workload_format: registry-driven detection + native exclusion #
# --------------------------------------------------------------------------- #
def test_detect_graph_workload_format_dynamo(tmp_path) -> None:
    from pathlib import Path

    from aiperf.dataset.graph.workload_detect import (
        _detect_graph_workload_format,
    )

    fixture = (
        Path(__file__).resolve().parents[1]
        / "dataset/graph/adapters/fixtures/dynamo_nested/nested_2_level.jsonl.gz"
    )
    assert _detect_graph_workload_format(fixture) == "dynamo_trace"


def test_detect_graph_workload_format_excludes_native(tmp_path) -> None:
    # A plain native-looking .jsonl must NOT be auto-detected as a graph
    # workload (native is explicit --graph only); returns None so the linear
    # pipeline keeps it.
    from aiperf.dataset.graph.workload_detect import (
        _detect_graph_workload_format,
    )

    f = tmp_path / "plain.jsonl"
    f.write_text('{"messages": [{"role": "user", "content": "hi"}]}\n')
    assert _detect_graph_workload_format(f) is None


# --------------------------------------------------------------------------- #
# graph_format override: forces graph classification + parse format           #
# --------------------------------------------------------------------------- #
def _write_minimal_native_graph(tmp_path: Path) -> Path:
    """Write a minimal valid native graph JSONL (one LLM node + one trace).

    Native is auto-detect-EXCLUDED, so this file is only ever treated as a graph
    workload when ``--graph-format native`` forces it.
    """
    f = tmp_path / "native_min.jsonl"
    f.write_text(
        '{"kind": "graph", "nodes": '
        '{"a": {"node_type": "llm", '
        '"prompt": [{"role": "user", "content": "hi"}], "output": "out"}}}\n'
        '{"kind": "trace", "id": "t1"}\n'
    )
    return f


@pytest.fixture
def make_run():
    """Build a ``BenchmarkRun`` with a default ``FileDataset`` for the given path.

    Wraps the shared ``make_run_from_cli`` helper so the override tests reuse the
    production resolver (CLI ``--input-file`` / ``--graph-format`` -> resolved
    ``FileDataset.path`` / ``.graph_format``) instead of hand-building config.
    """
    from aiperf.config.flags.cli_config import CLIConfig
    from tests.unit.conftest import make_run_from_cli

    def _make(*, path: str, graph_format: str | None = None):
        cfg = CLIConfig(
            model_names=["test-model"],
            input_file=path,
            graph_format=graph_format,
        )
        return make_run_from_cli(cfg)

    return _make


def test_graph_format_override_forces_native(make_run, tmp_path: Path) -> None:
    from aiperf.dataset.graph.workload_detect import (
        parse_graph_workload,
        resolve_graph_workload,
    )

    native_file = _write_minimal_native_graph(tmp_path)
    run = make_run(path=str(native_file), graph_format="native")
    # native is auto-detect-EXCLUDED, so only the override makes this a graph
    # workload:
    ref = resolve_graph_workload(run)
    assert ref is not None
    assert ref.path == native_file
    assert ref.format == "native"
    pb = parse_graph_workload(run, native_file)
    assert pb.graph is not None


def test_graph_format_override_forces_dynamo(make_run) -> None:
    from aiperf.dataset.graph.workload_detect import resolve_graph_workload

    fixture = (
        Path(__file__).resolve().parents[1]
        / "dataset/graph/adapters/fixtures/dynamo_nested/nested_2_level.jsonl.gz"
    )
    run = make_run(path=str(fixture), graph_format="dynamo_trace")
    ref = resolve_graph_workload(run)
    assert ref is not None
    assert ref.path == fixture
    assert ref.format == "dynamo_trace"
