# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Registry-dispatch-with-ctx vs run-parse parity pins (dynamo)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pytest import param

from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.graph.models import ParsedGraph
from aiperf.dataset.graph.segment_trie.envelope import read_prompt_segment_ids
from aiperf.dataset.graph.workload_detect import (
    parse_graph_workload,
    resolve_graph_parse_context,
)
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType
from tests.unit.conftest import make_run_from_cli

_TESTS_DIR = Path(__file__).parents[2]

DYNAMO_NESTED = (
    _TESTS_DIR
    / "unit/dataset/graph/adapters/fixtures/dynamo_nested/nested_2_level.jsonl.gz"
)


def _run_for(path: Path, **cli_overrides: Any) -> Any:
    """A resolved run config over ``path`` with an offline tokenizer."""
    cfg = CLIConfig(
        model_names=["test-model"],
        input_file=str(path),
        # Offline builtin tokenizer: the fake "test-model" must not trigger a
        # HF load during content synthesis (dynamo).
        tokenizer_name="builtin",
        **cli_overrides,
    )
    return make_run_from_cli(cfg)


def _parse_identity(parsed: ParsedGraph) -> dict[str, Any]:
    """The run-visible parse identity the two dispatch routes must agree on."""
    records = {"": parsed.graph, **parsed.graphs}
    return {
        "trace_ids": [t.id for t in parsed.traces],
        "node_ids": {key: sorted(rec.nodes) for key, rec in records.items()},
        "prompt_segment_ids": {
            key: {nid: read_prompt_segment_ids(n) for nid, n in rec.nodes.items()}
            for key, rec in records.items()
        },
        "pool_keys": sorted(parsed.segment_pool._by_id)
        if parsed.segment_pool is not None
        else None,
    }


@pytest.mark.parametrize(
    ("path", "fmt", "cli_overrides"),
    [
        param(
            DYNAMO_NESTED,
            "dynamo_trace",
            {"random_seed": 7},
            id="dynamo-explicit-seed",
        ),
        param(DYNAMO_NESTED, "dynamo_trace", {}, id="dynamo"),
    ],
)  # fmt: skip
def test_registry_dispatch_with_ctx_matches_run_parse(
    path: Path, fmt: str, cli_overrides: dict[str, Any]
) -> None:
    """Registry dispatch with a resolved ctx yields the same parse as the run-parse path."""
    # Identity compared: trace ids, node ids, per-node prompt_segment_ids, and the
    # segment pool's ``_by_id`` keys.
    run = _run_for(path, **cli_overrides)

    via_run = parse_graph_workload(run, path)
    adapter_cls = plugins.get_class(PluginType.GRAPH_ADAPTER, fmt)
    via_registry = adapter_cls.parse(path, resolve_graph_parse_context(run))

    run_identity = _parse_identity(via_run)
    registry_identity = _parse_identity(via_registry)
    assert run_identity["pool_keys"], "parse must carry real content segments"
    assert run_identity == registry_identity
