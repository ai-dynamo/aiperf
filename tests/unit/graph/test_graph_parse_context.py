# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GraphParseContext: sentinel semantics + per-adapter ctx forwarding rules."""

from __future__ import annotations

import copy
import inspect
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from msgspec import UNSET
from pytest import param

import aiperf.dataset._mp_context as mp_context_mod
from aiperf.dataset.graph import parser as parser_mod
from aiperf.dataset.graph.adapters.dynamo import trace as dynamo_trace
from aiperf.dataset.graph.models import GraphRecord, ParsedGraph
from aiperf.dataset.graph.parse_context import (
    GraphParseContext,
    publish_ctx_tokenizer_env,
)

_SpyCalls = list[tuple[tuple[Any, ...], dict[str, Any]]]
_CtxFactory = Callable[[], GraphParseContext | None]

# DynamoTraceAdapter.parse forwarded kwargs with no ctx fields set.
# ``release_replay=True`` is the production store-build opt-in the adapter
# entry always sets (freeing recorded replay hash lists), set at the adapter's
# own entry rather than threaded through the ctx.
_DYNAMO_ENV_DEFAULT_KWARGS: dict[str, Any] = {
    "release_replay": True,
}


def _spy_entry(monkeypatch: pytest.MonkeyPatch, module: Any, name: str) -> _SpyCalls:
    """Replace ``module.name`` with a spy recording (args, kwargs) per call."""
    calls: _SpyCalls = []

    def spy(*args: Any, **kwargs: Any) -> ParsedGraph:
        calls.append((args, kwargs))
        return ParsedGraph(graph=GraphRecord(), traces=[])

    monkeypatch.setattr(module, name, spy)
    return calls


def _spy_publish(monkeypatch: pytest.MonkeyPatch) -> _SpyCalls:
    """Spy on configure_loader_tokenizer_env (no env mutation)."""
    return _spy_entry(monkeypatch, mp_context_mod, "configure_loader_tokenizer_env")


def _full_ctx() -> GraphParseContext:
    """Every forwardable field set, with trust/revision deliberately left unset."""
    # Publishing is covered by its own tests below, so leaving trust/revision
    # unset keeps these forwarding compares env-free.
    return GraphParseContext(
        content_root_seed=1234,
        content_tokenizer="tok",
        prompt_corpus="sonnet",
        max_osl=77,
        idle_gap_cap_seconds=5.0,
        default_model="fallback-model",
        run_streaming=False,
        endpoint_extra=[("k", "v")],
    )


# --- msgspec UNSET sentinel (used by the graph channel store and reducers) -------------------------------------------------


@pytest.mark.parametrize(
    "clone",
    [
        param(lambda obj: pickle.loads(pickle.dumps(obj)), id="pickle-round-trip"),
        param(copy.deepcopy, id="deepcopy"),
    ],
)  # fmt: skip
def test_unset_survives_cloning_as_a_singleton(
    clone: Callable[[Any], Any],
) -> None:
    """UNSET stays identity-comparable across process/copy boundaries.

    ``aiperf.graph.channel_store`` and ``aiperf.graph.reducers`` compare with
    ``is UNSET``, so a clone that produced a new object would silently read as
    a written value.
    """
    assert clone(UNSET) is UNSET


def test_ctx_default_idle_gap_is_none() -> None:
    """An unspecified idle gap cap is None, meaning no per-trace compression."""
    assert GraphParseContext().idle_gap_cap_seconds is None


# --- dynamo forwarding ----------------------------------------------------------


@pytest.mark.parametrize(
    ("make_ctx", "expected_kwargs"),
    [
        param(lambda: None, {}, id="ctx-none-uses-protocol-defaults"),
        param(lambda: GraphParseContext(), {}, id="empty-ctx-forwards-nothing"),
        param(
            _full_ctx,
                {
                    "content_root_seed": 1234,
                    "content_tokenizer": "tok",
                    "prompt_corpus": "sonnet",
                    "max_osl": 77,
                    "streaming": False,
                    "idle_gap_cap_seconds": 5.0,
                },
            id="full-ctx-forwards-content-knobs",
        ),
        # from_dynamo_trace defaults prompt_corpus="coding"; a partial ctx must
        # not forward prompt_corpus=None over it.
        param(
            lambda: GraphParseContext(content_root_seed=42),
            {"content_root_seed": 42},
            id="partial-ctx-keeps-corpus-default",
        ),
        # ``None`` means no per-trace compression on BOTH the ctx and the
        # entry, so omitting it yields the same replay as forwarding it. There
        # is no third "use a built-in default" state to protect any more.
        param(
            lambda: GraphParseContext(idle_gap_cap_seconds=None),
            {},
            id="none-idle-gap-omitted-entry-default-matches",
        ),
    ],
)  # fmt: skip
def test_dynamo_parse_forwards_only_set_ctx_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    make_ctx: _CtxFactory,
    expected_kwargs: dict[str, Any],
) -> None:
    """DynamoTraceAdapter.parse forwards set ctx fields and omits unset ones."""
    calls = _spy_entry(monkeypatch, dynamo_trace, "from_dynamo_trace")
    path = tmp_path / "t.jsonl"

    dynamo_trace.DynamoTraceAdapter.parse(path, ctx=make_ctx())

    assert calls == [((path,), {**_DYNAMO_ENV_DEFAULT_KWARGS, **expected_kwargs})]


# --- trust/revision publish -----------------------------------------------------


@pytest.mark.parametrize(
    ("ctx", "expected_kwargs"),
    [
        param(
            GraphParseContext(
                tokenizer_trust_remote_code=True, tokenizer_revision="abc123"
            ),
            {"trust_remote_code": True, "revision": "abc123"},
            id="trust-true-with-pinned-revision",
        ),
        # revision=None passes verbatim; configure_loader_tokenizer_env's own
        # None -> "main" fallback handles it.
        param(
            GraphParseContext(tokenizer_trust_remote_code=False),
            {"trust_remote_code": False, "revision": None},
            id="trust-false-none-revision-verbatim",
        ),
    ],
)  # fmt: skip
def test_publish_ctx_tokenizer_env_publishes_when_trust_is_set(
    monkeypatch: pytest.MonkeyPatch,
    ctx: GraphParseContext,
    expected_kwargs: dict[str, Any],
) -> None:
    """Setting trust publishes both loader knobs, revision passed through untouched."""
    calls = _spy_publish(monkeypatch)

    publish_ctx_tokenizer_env(ctx)

    assert calls == [((), expected_kwargs)]


@pytest.mark.parametrize(
    "ctx",
    [
        param(GraphParseContext(tokenizer_revision="abc123"), id="revision-only-ctx"),
        param(None, id="no-ctx"),
    ],
)  # fmt: skip
def test_publish_ctx_tokenizer_env_skips_when_trust_unset(
    monkeypatch: pytest.MonkeyPatch, ctx: GraphParseContext | None
) -> None:
    """With trust unset, nothing is published -- the ambient loader env is left alone."""
    calls = _spy_publish(monkeypatch)

    publish_ctx_tokenizer_env(ctx)

    assert calls == []


@pytest.mark.parametrize(
    ("make_ctx", "expected_calls"),
    [
        param(
            lambda: GraphParseContext(
                tokenizer_trust_remote_code=False, tokenizer_revision="pin"
            ),
            [((), {"trust_remote_code": False, "revision": "pin"})],
            id="ctx-with-trust-publishes",
        ),
        param(lambda: None, [], id="ctx-none-does-not-publish"),
    ],
)  # fmt: skip
def test_dynamo_parse_publishes_loader_env_from_ctx(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    make_ctx: _CtxFactory,
    expected_calls: _SpyCalls,
) -> None:
    """The adapter entry itself runs the publish step for the ctx it was handed."""
    publish_calls = _spy_publish(monkeypatch)
    _spy_entry(monkeypatch, dynamo_trace, "from_dynamo_trace")

    dynamo_trace.DynamoTraceAdapter.parse(tmp_path / "t.jsonl", ctx=make_ctx())

    assert publish_calls == expected_calls


# --- parser plumbing ------------------------------------------------------------


@pytest.mark.parametrize(
    ("make_ctx", "expected_kwargs"),
    [
        param(
            lambda: GraphParseContext(content_root_seed=99),
            {"content_root_seed": 99},
            id="ctx-routed-to-adapter",
        ),
        param(lambda: None, {}, id="ctx-none-routes-protocol-defaults"),
    ],
)  # fmt: skip
def test_parse_graph_routes_ctx_to_the_selected_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    make_ctx: _CtxFactory,
    expected_kwargs: dict[str, Any],
) -> None:
    """parse_graph hands the ctx straight to the format's adapter without rewriting it."""
    calls = _spy_entry(monkeypatch, dynamo_trace, "from_dynamo_trace")
    path = tmp_path / "t.jsonl"

    parser_mod.parse_graph(path, format="dynamo_trace", ctx=make_ctx())

    assert calls == [((path,), {**_DYNAMO_ENV_DEFAULT_KWARGS, **expected_kwargs})]


def test_parse_graph_signature_drops_content_root_seed_for_ctx() -> None:
    """parse_graph takes ctx only -- the old per-knob kwarg and its shim are gone."""
    params = inspect.signature(parser_mod.parse_graph).parameters
    assert "ctx" in params
    assert "content_root_seed" not in params
    assert not hasattr(parser_mod, "_accepts_content_root_seed")
