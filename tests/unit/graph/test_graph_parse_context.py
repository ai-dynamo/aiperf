# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GraphParseContext: sentinel semantics + per-adapter ctx forwarding rules.

Pins the adversarially-reviewed contract of the adapter protocol's ``ctx``
parameter:

* forward-only-when-set — a partial ctx never clobbers an adapter entry's
  non-None defaults (dynamo ``prompt_corpus="coding"``, dag
  ``run_streaming=True``) with ``None``;
* tri-state ``idle_gap_cap_seconds`` — explicit ``None`` (user's
  ``synthesis.idle_gap_cap_seconds: null``) forwards VERBATIM (disable
  warping) while ``UNSET`` forwards nothing (adapter default);
* ``ctx=None`` is byte-equal to today's protocol-default entries (literal
  kwarg-dict compares);
* trust/revision publish to the loader-preload env IFF
  ``tokenizer_trust_remote_code`` is set, revision passed verbatim.
"""

from __future__ import annotations

import copy
import inspect
import pickle
from pathlib import Path
from typing import Any

import pytest

import aiperf.dataset._mp_context as mp_context_mod
from aiperf.dataset.graph import parser as parser_mod
from aiperf.dataset.graph.adapters import native as native_mod
from aiperf.dataset.graph.adapters.dag_jsonl import trace as dag_trace
from aiperf.dataset.graph.adapters.dynamo import trace as dynamo_trace
from aiperf.dataset.graph.adapters.weka import trace as weka_trace
from aiperf.dataset.graph.models import GraphRecord, ParsedGraph
from aiperf.dataset.graph.parse_context import (
    UNSET,
    GraphParseContext,
    publish_ctx_tokenizer_env,
)

_SpyCalls = list[tuple[tuple[Any, ...], dict[str, Any]]]

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
    """Every forwardable field set (trust/revision left unset: publish is
    covered by its own tests so forwarding compares stay env-free)."""
    return GraphParseContext(
        content_root_seed=1234,
        content_tokenizer="tok",
        prompt_corpus="sonnet",
        max_osl=77,
        idle_gap_cap_seconds=5.0,
        default_model="fallback-model",
        run_streaming=False,
        delay_cap_seconds=2.5,
        endpoint_extra=[("k", "v")],
    )


# --- UNSET sentinel semantics -------------------------------------------------


def test_unset_pickle_round_trip_preserves_identity() -> None:
    assert pickle.loads(pickle.dumps(UNSET)) is UNSET


def test_unset_deepcopy_preserves_identity() -> None:
    assert copy.deepcopy(UNSET) is UNSET


def test_ctx_default_idle_gap_is_unset_sentinel() -> None:
    assert GraphParseContext().idle_gap_cap_seconds is UNSET


# --- weka forwarding ------------------------------------------------------------


def test_weka_parse_ctx_none_matches_protocol_default_entry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _spy_entry(monkeypatch, weka_trace, "from_weka_trace")
    p = tmp_path / "t.json"
    weka_trace.WekaTraceAdapter.parse(p, ctx=None)
    assert calls == [((p,), {})]


def test_weka_parse_full_ctx_forwards_weka_fields_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _spy_entry(monkeypatch, weka_trace, "from_weka_trace")
    p = tmp_path / "t.json"
    weka_trace.WekaTraceAdapter.parse(p, ctx=_full_ctx())
    assert calls == [
        (
            (p,),
            {
                "content_root_seed": 1234,
                "content_tokenizer": "tok",
                "prompt_corpus": "sonnet",
                "max_osl": 77,
                "idle_gap_cap_seconds": 5.0,
            },
        )
    ]


def test_weka_parse_partial_ctx_forwards_only_set_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _spy_entry(monkeypatch, weka_trace, "from_weka_trace")
    weka_trace.WekaTraceAdapter.parse(
        tmp_path / "t.json", ctx=GraphParseContext(content_root_seed=42)
    )
    assert calls[0][1] == {"content_root_seed": 42}


def test_weka_parse_explicit_none_idle_gap_forwards_none_verbatim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Explicit None = the user's `synthesis.idle_gap_cap_seconds: null` =
    # warping DISABLED. Must forward, never collapse into the _USE_DEFAULT default.
    calls = _spy_entry(monkeypatch, weka_trace, "from_weka_trace")
    weka_trace.WekaTraceAdapter.parse(
        tmp_path / "t.json", ctx=GraphParseContext(idle_gap_cap_seconds=None)
    )
    assert calls[0][1] == {"idle_gap_cap_seconds": None}


def test_weka_parse_unset_idle_gap_forwards_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _spy_entry(monkeypatch, weka_trace, "from_weka_trace")
    weka_trace.WekaTraceAdapter.parse(tmp_path / "t.json", ctx=GraphParseContext())
    assert "idle_gap_cap_seconds" not in calls[0][1]
    assert calls[0][1] == {}


# --- dynamo forwarding ----------------------------------------------------------


def test_dynamo_parse_ctx_none_matches_protocol_default_entry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _spy_entry(monkeypatch, dynamo_trace, "from_dynamo_trace")
    p = tmp_path / "t.jsonl"
    dynamo_trace.DynamoTraceAdapter.parse(p, ctx=None)
    assert calls == [((p,), dict(_DYNAMO_ENV_DEFAULT_KWARGS))]


def test_dynamo_parse_full_ctx_forwards_content_knobs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _spy_entry(monkeypatch, dynamo_trace, "from_dynamo_trace")
    dynamo_trace.DynamoTraceAdapter.parse(tmp_path / "t.jsonl", ctx=_full_ctx())
    assert calls[0][1] == {
        **_DYNAMO_ENV_DEFAULT_KWARGS,
        "content_root_seed": 1234,
        "content_tokenizer": "tok",
        "prompt_corpus": "sonnet",
        "idle_gap_cap_seconds": 5.0,
    }


def test_dynamo_parse_partial_ctx_does_not_clobber_corpus_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # from_dynamo_trace defaults prompt_corpus="coding"; a partial ctx must not
    # forward prompt_corpus=None over it.
    calls = _spy_entry(monkeypatch, dynamo_trace, "from_dynamo_trace")
    dynamo_trace.DynamoTraceAdapter.parse(
        tmp_path / "t.jsonl", ctx=GraphParseContext(content_root_seed=42)
    )
    assert "prompt_corpus" not in calls[0][1]
    assert calls[0][1] == {**_DYNAMO_ENV_DEFAULT_KWARGS, "content_root_seed": 42}


def test_dynamo_parse_explicit_none_idle_gap_forwards_none_verbatim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Same tri-state rule as weka: explicit None = the user's
    # `synthesis.idle_gap_cap_seconds: null` = warping DISABLED. Must forward,
    # never collapse into the shared 60s entry default.
    calls = _spy_entry(monkeypatch, dynamo_trace, "from_dynamo_trace")
    dynamo_trace.DynamoTraceAdapter.parse(
        tmp_path / "t.jsonl", ctx=GraphParseContext(idle_gap_cap_seconds=None)
    )
    assert calls[0][1] == {**_DYNAMO_ENV_DEFAULT_KWARGS, "idle_gap_cap_seconds": None}


def test_dynamo_parse_unset_idle_gap_forwards_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _spy_entry(monkeypatch, dynamo_trace, "from_dynamo_trace")
    dynamo_trace.DynamoTraceAdapter.parse(tmp_path / "t.jsonl", ctx=GraphParseContext())
    assert "idle_gap_cap_seconds" not in calls[0][1]
    assert calls[0][1] == dict(_DYNAMO_ENV_DEFAULT_KWARGS)


# --- dag_jsonl forwarding -------------------------------------------------------


def test_dag_parse_ctx_none_matches_protocol_default_entry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _spy_entry(monkeypatch, dag_trace, "from_dag_jsonl")
    p = tmp_path / "t.jsonl"
    dag_trace.DagJsonlGraphAdapter.parse(p, ctx=None)
    assert calls == [((str(p),), {})]


def test_dag_parse_full_ctx_forwards_dispatch_knobs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _spy_entry(monkeypatch, dag_trace, "from_dag_jsonl")
    p = tmp_path / "t.jsonl"
    dag_trace.DagJsonlGraphAdapter.parse(p, ctx=_full_ctx())
    assert calls == [
        (
            (str(p),),
            {
                "default_model": "fallback-model",
                "run_streaming": False,
                "delay_cap_seconds": 2.5,
                "endpoint_extra": [("k", "v")],
            },
        )
    ]


def test_dag_parse_partial_ctx_does_not_clobber_streaming_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # from_dag_jsonl defaults run_streaming=True; a partial ctx must not
    # forward run_streaming=None over it.
    calls = _spy_entry(monkeypatch, dag_trace, "from_dag_jsonl")
    dag_trace.DagJsonlGraphAdapter.parse(
        tmp_path / "t.jsonl", ctx=GraphParseContext(content_root_seed=42)
    )
    assert calls[0][1] == {}


# --- native ---------------------------------------------------------------------


def test_native_parse_accepts_and_ignores_ctx(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _spy_entry(monkeypatch, parser_mod, "parse_native")
    p = tmp_path / "t.yaml"
    native_mod.NativeGraphAdapter.parse(p, ctx=_full_ctx())
    native_mod.NativeGraphAdapter.parse(p, ctx=None)
    assert calls == [((p,), {}), ((p,), {})]


# --- trust/revision publish -----------------------------------------------------


def test_publish_ctx_tokenizer_env_trust_set_publishes_revision_verbatim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _spy_publish(monkeypatch)
    publish_ctx_tokenizer_env(
        GraphParseContext(tokenizer_trust_remote_code=True, tokenizer_revision="abc123")
    )
    assert calls == [((), {"trust_remote_code": True, "revision": "abc123"})]


def test_publish_ctx_tokenizer_env_trust_false_none_revision_passes_verbatim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # revision=None passes verbatim; configure_loader_tokenizer_env's own
    # None -> "main" fallback handles it.
    calls = _spy_publish(monkeypatch)
    publish_ctx_tokenizer_env(GraphParseContext(tokenizer_trust_remote_code=False))
    assert calls == [((), {"trust_remote_code": False, "revision": None})]


def test_publish_ctx_tokenizer_env_trust_unset_skips_publish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _spy_publish(monkeypatch)
    publish_ctx_tokenizer_env(GraphParseContext(tokenizer_revision="abc123"))
    publish_ctx_tokenizer_env(None)
    assert calls == []


def test_weka_parse_ctx_trust_set_publishes_loader_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    publish_calls = _spy_publish(monkeypatch)
    _spy_entry(monkeypatch, weka_trace, "from_weka_trace")
    weka_trace.WekaTraceAdapter.parse(
        tmp_path / "t.json",
        ctx=GraphParseContext(
            tokenizer_trust_remote_code=True, tokenizer_revision="pin"
        ),
    )
    assert publish_calls == [((), {"trust_remote_code": True, "revision": "pin"})]


def test_dynamo_parse_ctx_trust_set_publishes_loader_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    publish_calls = _spy_publish(monkeypatch)
    _spy_entry(monkeypatch, dynamo_trace, "from_dynamo_trace")
    dynamo_trace.DynamoTraceAdapter.parse(
        tmp_path / "t.jsonl",
        ctx=GraphParseContext(
            tokenizer_trust_remote_code=False, tokenizer_revision="pin"
        ),
    )
    assert publish_calls == [((), {"trust_remote_code": False, "revision": "pin"})]


def test_weka_parse_ctx_none_does_not_publish(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    publish_calls = _spy_publish(monkeypatch)
    _spy_entry(monkeypatch, weka_trace, "from_weka_trace")
    weka_trace.WekaTraceAdapter.parse(tmp_path / "t.json", ctx=None)
    assert publish_calls == []


# --- parser plumbing ------------------------------------------------------------


def test_parse_graph_routes_ctx_to_adapter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _spy_entry(monkeypatch, weka_trace, "from_weka_trace")
    p = tmp_path / "t.json"
    parser_mod.parse_graph(
        p, format="weka_trace", ctx=GraphParseContext(content_root_seed=99)
    )
    assert calls[0][1] == {"content_root_seed": 99}


def test_parse_graph_ctx_none_routes_protocol_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _spy_entry(monkeypatch, weka_trace, "from_weka_trace")
    p = tmp_path / "t.json"
    parser_mod.parse_graph(p, format="weka_trace")
    assert calls == [((p,), {})]


def test_parse_graph_signature_drops_content_root_seed_for_ctx() -> None:
    params = inspect.signature(parser_mod.parse_graph).parameters
    assert "ctx" in params
    assert "content_root_seed" not in params
    assert not hasattr(parser_mod, "_accepts_content_root_seed")
