# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified-store wiring through the GraphStoreBuilder build path.

Behavioral wiring proofs: the assertions monkeypatch the real builders and
drive the real methods, so a docstring mention of a symbol cannot satisfy them
(the trap the previous ``inspect.getsource`` substring checks fell into).
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from aiperf.config.resolution.plan import GraphWorkloadRef, ResolvedConfig
from aiperf.dataset import dataset_manager
from aiperf.dataset.graph import store_build
from aiperf.dataset.graph.models import GraphRecord, ParsedGraph, TraceRecord
from aiperf.dataset.graph.segment_ir import store_builder

# Recognized weka HF dataset id (marker + nonexistent org/name path); mirrors
# the id the real-ingest suite pins.
_HF_WEKA_ID = "semianalysisai/cc-traces-weka-062126"


def _bare_builder(
    graph_ref: GraphWorkloadRef | None = None,
) -> store_build.GraphStoreBuilder:
    """A GraphStoreBuilder on a stub run for direct method-wiring tests.

    ``build`` reads the run's memoized graph resolution
    (``resolve_graph_workload``), so the stub carries a real ``ResolvedConfig``;
    pass ``graph_ref`` to preset the memo the production resolver chain (or the
    DatasetManager's accessor call) populates before any build.
    """
    resolved = ResolvedConfig(
        graph_workload=graph_ref, graph_workload_resolved=graph_ref is not None
    )
    return store_build.GraphStoreBuilder(
        run=SimpleNamespace(benchmark_id="bid", resolved=resolved)
    )


@pytest.mark.asyncio
async def test_build_interned_unified_store_calls_interned_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The interned unified build routes through ``build_unified_trie_store_interned``.

    Monkeypatches the node-typed interned builder at its definition site (the
    method imports it locally) and asserts the flag path actually CALLS it with
    the parse + store it was handed, returning the builder's catalog verbatim.
    """
    calls: dict[str, tuple[object, object]] = {}

    async def _fake_builder(parsed: object, unified: object) -> dict:
        calls["args"] = (parsed, unified)
        return {"trace_a": {"n0": 0}}

    monkeypatch.setattr(
        store_builder, "build_unified_trie_store_interned", _fake_builder
    )

    sentinel_parsed, sentinel_store = object(), object()
    catalog = await _bare_builder()._build_interned_unified_store(
        sentinel_parsed, sentinel_store
    )

    assert calls["args"] == (sentinel_parsed, sentinel_store)
    assert catalog == {"trace_a": {"n0": 0}}


@pytest.mark.asyncio
async def test_build_routes_hf_id_to_streaming_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An HF weka dataset id reaches the ONE streaming store build as weka_trace.

    ``GraphStoreBuilder.build`` always calls ``_build_graph_store_streaming``;
    the wiring under test is the format threading -- the run's memoized
    ``weka_trace`` ref (populated from a recognized weka HF repo id at config
    resolution; see ``test_graph_workload_resolution``) must arrive as
    ``fmt == "weka_trace"`` so the builder takes the worker-pool payload
    branch.
    """
    builder = _bare_builder(
        GraphWorkloadRef(path=Path(_HF_WEKA_ID), format="weka_trace")
    )
    called: dict[str, object] = {}

    async def _fake_streaming(path: Path, base_path: Path, fmt: str | None) -> tuple:
        called["path"] = path
        called["fmt"] = fmt
        return {"trace_a": {"n0": 0}}, ParsedGraph(
            graph=GraphRecord(), traces=[TraceRecord(id="trace_a", tags=["x"])]
        )

    # The stub run carries only benchmark_id, so the run-derived env/forkserver
    # hooks must be bypassed: the module-level tokenizer-env publisher (which
    # now takes the run) is monkeypatched at store_build's call site, and the
    # forkserver pre-start / drain are instance-shadowed. The sidecar path is
    # preset because the shadowed drain never writes one and build() hard-fails
    # a sidecar-less build.
    monkeypatch.setattr(
        store_build, "publish_graph_loader_tokenizer_env", lambda run: None
    )
    builder._prestart_loader_forkserver = lambda: None
    builder._build_graph_store_streaming = _fake_streaming
    builder._sidecar_path = tmp_path / "graph_meta.sidecar"

    result = await builder.build(Path(_HF_WEKA_ID))

    assert called["path"] == Path(_HF_WEKA_ID)
    assert called["fmt"] == "weka_trace"
    assert result.facet.trace_ids == ["trace_a"]
    assert result.sidecar_path == tmp_path / "graph_meta.sidecar"


@pytest.mark.asyncio
async def test_build_prestarts_forkserver_before_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The forkserver pre-start must run BEFORE the offloaded store build.

    ``_eagerly_start_forkserver`` dup2-swaps process-wide stdio; if the helper
    were first started lazily inside the ``asyncio.to_thread`` build, the swap
    would race concurrent event-loop logging. The env publish must come first
    so the helper snapshots the run's trust/revision triple.
    """
    builder = _bare_builder()
    order: list[str] = []

    async def _fake_streaming(path: Path, base_path: Path, fmt: str | None) -> tuple:
        order.append("store-build")
        return {"trace_a": {"n0": 0}}, ParsedGraph(
            graph=GraphRecord(), traces=[TraceRecord(id="trace_a", tags=["x"])]
        )

    monkeypatch.setattr(
        store_build,
        "publish_graph_loader_tokenizer_env",
        lambda run: order.append("publish-env"),
    )
    builder._prestart_loader_forkserver = lambda: order.append("prestart-forkserver")
    builder._build_graph_store_streaming = _fake_streaming
    builder._sidecar_path = tmp_path / "graph_meta.sidecar"

    local_trace = tmp_path / "trace.json"
    local_trace.write_text("{}")
    await builder.build(local_trace)

    assert order == ["publish-env", "prestart-forkserver", "store-build"]


@pytest.mark.asyncio
async def test_configure_graph_workload_returns_builder_facet(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``DatasetManager._configure_graph_workload`` hands back the builder's facet.

    The DatasetManager builds nothing itself: the wrapper must run
    the endpoint gate, delegate to ``GraphStoreBuilder.build``, and return the
    result's facet verbatim (component-integration callers consume
    ``.trace_ids`` off it).
    """
    sentinel_facet = object()
    built: dict[str, object] = {}

    async def _fake_build(self: store_build.GraphStoreBuilder, graph_path: Path):
        built["run"] = self.run
        built["path"] = graph_path
        return SimpleNamespace(
            facet=sentinel_facet,
            sidecar_path=tmp_path / "sidecar",
            base_path=tmp_path,
        )

    monkeypatch.setattr(store_build.GraphStoreBuilder, "build", _fake_build)

    gated: list[object] = []
    from aiperf.dataset.graph import workload_detect

    monkeypatch.setattr(workload_detect, "validate_graph_endpoint_type", gated.append)

    dm = dataset_manager.DatasetManager.__new__(dataset_manager.DatasetManager)
    dm.run = SimpleNamespace(benchmark_id="bid")

    facet = await dm._configure_graph_workload(tmp_path / "trace.json")

    assert facet is sentinel_facet
    assert built["run"] is dm.run
    assert built["path"] == tmp_path / "trace.json"
    assert gated == [dm.run], "the endpoint gate must run before the build"


def test_store_build_has_no_unsupported_combo_error() -> None:
    """Tombstone: the old unsupported-combo error must never come back.

    An ABSENCE can only be asserted against the source text; the positive
    routing halves are covered behaviorally above. Checked against BOTH homes
    the build logic has lived in.
    """
    src = inspect.getsource(dataset_manager.DatasetManager._configure_graph_workload)
    assert "WekaUnifiedStoreUnsupportedError" not in src
    assert "WekaUnifiedStoreUnsupportedError" not in inspect.getsource(store_build)
