# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified-store wiring through the GraphStoreBuilder build path."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from aiperf.common.environment import Environment
from aiperf.config.resolution.plan import GraphWorkloadRef, ResolvedConfig
from aiperf.dataset import dataset_manager
from aiperf.dataset.graph import store_build
from aiperf.dataset.graph.models import GraphRecord, ParsedGraph, TraceRecord
from aiperf.dataset.graph.segment_trie import store_builder

# Recognized weka HF dataset id (marker + nonexistent org/name path); mirrors
# the id the real-ingest suite pins.
_HF_WEKA_ID = "semianalysisai/cc-traces-weka-062126"


def _bare_builder(
    graph_ref: GraphWorkloadRef | None = None,
) -> store_build.GraphStoreBuilder:
    """A GraphStoreBuilder on a stub run for direct method-wiring tests."""
    resolved = ResolvedConfig(
        graph_workload=graph_ref, graph_workload_resolved=graph_ref is not None
    )
    return store_build.GraphStoreBuilder(
        run=SimpleNamespace(benchmark_id="bid", resolved=resolved)
    )


def _stub_build_prerequisites(
    builder: store_build.GraphStoreBuilder,
    monkeypatch: pytest.MonkeyPatch,
    drain,
    sidecar_path: Path,
    on_publish_env=lambda run: None,
) -> None:
    """Neutralize the run-derived hooks ``build()`` needs so a stub run carrying only ``benchmark_id`` can drive it."""
    # The tokenizer-env publisher takes the run, so it is patched at store_build's
    # call site; the forkserver pre-start and drain are instance-shadowed. The
    # sidecar path is preset because a shadowed drain never writes one and
    # build() hard-fails a sidecar-less build.
    monkeypatch.setattr(
        store_build, "publish_graph_loader_tokenizer_env", on_publish_env
    )
    builder._prestart_loader_forkserver = lambda: None
    builder._build_graph_store_streaming = drain
    builder._sidecar_path = sidecar_path


@pytest.mark.asyncio
async def test_build_interned_unified_store_calls_interned_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The interned unified build routes through ``build_unified_trie_store_interned``."""
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
    """An HF weka dataset id reaches the ONE streaming store build as weka_trace."""
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

    _stub_build_prerequisites(
        builder, monkeypatch, _fake_streaming, tmp_path / "graph_meta.sidecar"
    )

    result = await builder.build(Path(_HF_WEKA_ID))

    assert called["path"] == Path(_HF_WEKA_ID)
    assert called["fmt"] == "weka_trace"
    assert result.facet.trace_ids == ["trace_a"]
    assert result.sidecar_path == tmp_path / "graph_meta.sidecar"


@pytest.mark.asyncio
async def test_build_prestarts_forkserver_before_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The forkserver pre-start must run BEFORE the offloaded store build."""
    builder = _bare_builder()
    order: list[str] = []

    async def _fake_streaming(path: Path, base_path: Path, fmt: str | None) -> tuple:
        order.append("store-build")
        return {"trace_a": {"n0": 0}}, ParsedGraph(
            graph=GraphRecord(), traces=[TraceRecord(id="trace_a", tags=["x"])]
        )

    _stub_build_prerequisites(
        builder,
        monkeypatch,
        _fake_streaming,
        tmp_path / "graph_meta.sidecar",
        on_publish_env=lambda run: order.append("publish-env"),
    )
    builder._prestart_loader_forkserver = lambda: order.append("prestart-forkserver")

    local_trace = tmp_path / "trace.json"
    local_trace.write_text("{}")
    await builder.build(local_trace)

    assert order == ["publish-env", "prestart-forkserver", "store-build"]


@pytest.mark.asyncio
async def test_configure_graph_workload_returns_builder_facet(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``DatasetManager._configure_graph_workload`` hands back the builder's facet."""
    sentinel_facet = object()
    built: dict[str, object] = {}

    async def _fake_build(self: store_build.GraphStoreBuilder, graph_path: Path):
        order.append("store-build")
        built["run"] = self.run
        built["path"] = graph_path
        return SimpleNamespace(
            facet=sentinel_facet,
            sidecar_path=tmp_path / "sidecar",
            base_path=tmp_path,
        )

    monkeypatch.setattr(store_build.GraphStoreBuilder, "build", _fake_build)
    # _build_graph_store claims (mkdir + lock) and sweeps under the real store
    # root before delegating, so without this the test would create dirs in the
    # developer's temp dir and sweep whatever else is living there.
    monkeypatch.setattr(Environment.DATASET, "MMAP_BASE_PATH", tmp_path)

    # One shared sequence, not two independent lists: ordering can only be
    # asserted against a single ordered record. With separate `gated`/`built`
    # lists, swapping the gate and the build in _build_graph_store leaves every
    # assertion green.
    order: list[str] = []
    gated: list[object] = []
    from aiperf.dataset.graph import workload_detect

    def _record_gate(run: object) -> None:
        order.append("endpoint-gate")
        gated.append(run)

    monkeypatch.setattr(workload_detect, "validate_graph_endpoint_type", _record_gate)

    dm = dataset_manager.DatasetManager.__new__(dataset_manager.DatasetManager)
    dm.run = SimpleNamespace(benchmark_id="bid")

    facet = await dm._configure_graph_workload(tmp_path / "trace.json")

    assert facet is sentinel_facet
    assert built["run"] is dm.run
    assert built["path"] == tmp_path / "trace.json"
    assert gated == [dm.run]
    assert order == ["endpoint-gate", "store-build"], (
        "the endpoint gate must run BEFORE the build, so an unsupported endpoint "
        "fails fast instead of after a corpus-sized store has been written"
    )


def test_store_build_has_no_unsupported_combo_error() -> None:
    """Source-text guard, not a behavioral test: an ABSENCE can only be asserted against the source, and the positive routing halves are covered behaviorally above."""
    # Mechanical guard. Checked against BOTH homes the build logic has lived in.
    src = inspect.getsource(dataset_manager.DatasetManager._configure_graph_workload)
    assert "WekaUnifiedStoreUnsupportedError" not in src
    assert "WekaUnifiedStoreUnsupportedError" not in inspect.getsource(store_build)
