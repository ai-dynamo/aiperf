# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The loader mp-context must branch on the ``IS_LINUX`` constant (DM6) and publish the run's real tokenizer trust/revision triple rather than silently preloading with ``(False, "main")`` (DM4)."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from pytest import param

import aiperf.dataset._mp_context as mp_context
import aiperf.dataset._tokenizer_preload as tokenizer_preload
from aiperf.common.constants import IS_WINDOWS
from aiperf.common.tokenizer import Tokenizer
from aiperf.config.tokenizer import TokenizerConfig
from aiperf.dataset.graph.adapters.shared.content import CorpusContentSynthesizer
from aiperf.dataset.graph.workload_detect import publish_graph_loader_tokenizer_env

_ENV_VARS = (
    mp_context._ENV_PRELOAD_NAME,
    mp_context._ENV_PRELOAD_TRUST,
    mp_context._ENV_PRELOAD_REVISION,
)


class _StubGenerator:
    """Stand-in for ``CodingContentGenerator`` so only the tokenizer-load call shape is exercised."""

    def __init__(self, config: object, tokenizer: object) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self._hash_id_corpus_rng = None


def _record_tokenizer_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict]:
    """Patch ``Tokenizer.from_pretrained`` to record its call kwargs and return the shared record list."""
    calls: list[dict] = []

    def recording_from_pretrained(name: str, **kwargs) -> SimpleNamespace:
        calls.append({"name": name, **kwargs})
        return SimpleNamespace()

    # One patch covers both call sites: content.py's module-level ``Tokenizer``
    # and _tokenizer_preload's local import are the same class object.
    monkeypatch.setattr(Tokenizer, "from_pretrained", recording_from_pretrained)
    return calls


def _force_preload_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``CorpusContentSynthesizer._build_generator`` take the worker-role on-demand fallback path."""
    monkeypatch.setattr(
        "aiperf.dataset._tokenizer_preload.get_preloaded", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "aiperf.dataset.generator.coding_content.CodingContentGenerator",
        _StubGenerator,
    )


@pytest.fixture
def clean_loader_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Fresh context cache + no preload env vars, spawn context (no forkserver)."""
    monkeypatch.setattr(mp_context, "_loader_ctx", None)
    monkeypatch.setattr(mp_context, "_loader_ctx_key", None)
    monkeypatch.setattr(mp_context, "IS_LINUX", False)
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_get_loader_mp_context_uses_is_linux_constant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ``IS_LINUX`` False the loader context must be a spawn context with no forkserver helper -- a ``platform.system()`` branch would not be patchable and violates the constants contract."""
    monkeypatch.setattr(mp_context, "_loader_ctx", None)
    monkeypatch.setattr(mp_context, "_loader_ctx_key", None)
    monkeypatch.setattr(mp_context, "IS_LINUX", False)

    ctx = mp_context.get_loader_mp_context()

    assert ctx.get_start_method() == "spawn"


@pytest.mark.parametrize(
    "configured,get_kwargs,expected",
    [
        param(
            None,
            {},
            ("some/tok", "false", "main"),
            id="unconfigured_defaults_to_untrusted_main",
        ),
        param(
            {"trust_remote_code": True, "revision": "deadbeef"},
            {},
            ("some/tok", "true", "deadbeef"),
            id="configured_trust_and_revision_survive_name_only_call",
        ),
        param(
            {"trust_remote_code": True, "revision": "r1"},
            {"trust_remote_code": False, "revision": "r2"},
            ("some/tok", "false", "r2"),
            id="explicit_args_override_configured_env",
        ),
    ],
)  # fmt: skip
def test_get_loader_mp_context_resolves_preload_env_triple(
    clean_loader_env: pytest.MonkeyPatch,
    configured: dict | None,
    get_kwargs: dict,
    expected: tuple[str, str, str],
) -> None:
    """The name-only call is the only production call shape (via ``_loader_pool_context``), so it must default the triple when unconfigured, preserve whatever ``configure_loader_tokenizer_env`` published, and still yield to explicit arguments."""
    if configured is not None:
        mp_context.configure_loader_tokenizer_env(**configured)

    mp_context.get_loader_mp_context(preload_tokenizer="some/tok", **get_kwargs)

    assert (
        os.environ[mp_context._ENV_PRELOAD_NAME],
        os.environ[mp_context._ENV_PRELOAD_TRUST],
        os.environ[mp_context._ENV_PRELOAD_REVISION],
    ) == expected


@pytest.mark.skipif(IS_WINDOWS, reason="forkserver context does not exist on Windows")
def test_get_loader_mp_context_second_call_skips_forkserver_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cached context must not re-enter the stdio dup2 swap: the builder pre-starts the forkserver at a known-quiet point precisely so the pool's later call from inside the offloaded build is a no-op instead of racing live event-loop logging."""
    starts: list[None] = []
    monkeypatch.setattr(mp_context, "_loader_ctx", None)
    monkeypatch.setattr(mp_context, "_loader_ctx_key", None)
    monkeypatch.setattr(mp_context, "IS_LINUX", True)
    monkeypatch.setattr(
        mp_context, "_eagerly_start_forkserver", lambda: starts.append(None)
    )

    first = mp_context.get_loader_mp_context()
    second = mp_context.get_loader_mp_context(preload_tokenizer="some/tok")

    assert first is second
    assert len(starts) == 1


def test_parse_graph_workload_publishes_env_triple(
    clean_loader_env: pytest.MonkeyPatch,
) -> None:
    """parse_graph_workload is the one seam every graph parse goes through, so it must publish the run's tokenizer trust/revision triple itself: a direct caller (tooling, tests) has no DatasetManager configure step to do it, and the forkserver helper snapshots the env once at spawn."""
    monkeypatch = clean_loader_env
    from aiperf.config.flags.cli_config import CLIConfig
    from aiperf.dataset.graph.workload_detect import parse_graph_workload
    from tests.unit.conftest import make_run_from_cli

    graph_min = (
        Path(__file__).parent
        / "graph"
        / "adapters"
        / "fixtures"
        / "dynamo_nested"
        / "nested_2_level.jsonl.gz"
    )
    run = make_run_from_cli(
        CLIConfig(
            model_names=["test-model"],
            input_file=str(graph_min),
            tokenizer_name="builtin",
            trust_remote_code=True,
            tokenizer_revision="pinned-rev",
        )
    )

    # A direct caller with no DatasetManager configure step (tooling / tests).
    parse_graph_workload(run, graph_min)

    assert os.environ[mp_context._ENV_PRELOAD_TRUST] == "true"
    assert os.environ[mp_context._ENV_PRELOAD_REVISION] == "pinned-rev"

    # ...and the synthesizer layer loads with exactly that triple: the
    # worker-role fallback in ``content._build_generator`` on a preload miss.
    calls = _record_tokenizer_loads(monkeypatch)
    _force_preload_miss(monkeypatch)
    CorpusContentSynthesizer._build_generator("gpt2", "coding")

    assert calls, "fallback tokenizer load must fire"
    assert calls[0]["trust_remote_code"] is True
    assert calls[0]["revision"] == "pinned-rev"


def test_store_builder_publishes_loader_tokenizer_env_from_run_config(
    clean_loader_env: pytest.MonkeyPatch,
) -> None:
    """Real TokenizerConfig on purpose: a mock would hide attribute-path drift."""
    tokenizer_config = TokenizerConfig(trust_remote_code=True, revision="pinned-rev")
    run = SimpleNamespace(cfg=SimpleNamespace(tokenizer=tokenizer_config))

    publish_graph_loader_tokenizer_env(run)
    mp_context.get_loader_mp_context(preload_tokenizer="some/tok")

    assert os.environ[mp_context._ENV_PRELOAD_TRUST] == "true"
    assert os.environ[mp_context._ENV_PRELOAD_REVISION] == "pinned-rev"


def test_preload_and_fallback_resolve_the_same_tokenizer(
    clean_loader_env: pytest.MonkeyPatch,
) -> None:
    """A preload hit and an on-demand fallback miss must load with the SAME ``(name, trust_remote_code, revision, resolve_alias)`` arguments, or an aliased name silently synthesizes with a different tokenizer depending on whether the forkserver preload fired."""
    monkeypatch = clean_loader_env
    calls = _record_tokenizer_loads(monkeypatch)
    monkeypatch.setattr(tokenizer_preload, "_LOADED", {})
    monkeypatch.setenv(mp_context._ENV_PRELOAD_NAME, "gpt2")
    monkeypatch.setenv(mp_context._ENV_PRELOAD_TRUST, "true")
    monkeypatch.setenv(mp_context._ENV_PRELOAD_REVISION, "pinned-rev")

    # Forkserver-helper role: the preload load.
    tokenizer_preload._preload()

    # Worker role on a preload MISS: the on-demand fallback in
    # content._build_generator.
    _force_preload_miss(monkeypatch)
    CorpusContentSynthesizer._build_generator("gpt2", "coding")

    assert len(calls) == 2, f"expected preload + fallback loads, got {calls}"
    preload_call, fallback_call = calls
    assert preload_call["name"] == fallback_call["name"] == "gpt2"
    for key in ("trust_remote_code", "revision", "resolve_alias"):
        assert preload_call[key] == fallback_call[key], (
            f"{key} diverges between preload ({preload_call[key]!r}) and "
            f"fallback ({fallback_call[key]!r})"
        )
    assert preload_call["trust_remote_code"] is True
    assert preload_call["revision"] == "pinned-rev"
    assert preload_call["resolve_alias"] is True


@pytest.mark.parametrize(
    "module_name",
    [
        param("aiperf.dataset.loader.parallel_convert", id="synthetic"),
        param("aiperf.dataset.loader.weka_parallel_convert", id="weka"),
    ],
)  # fmt: skip
def test_convert_worker_fallback_resolves_aliases_like_the_preload(
    module_name: str,
) -> None:
    """Every worker-init preload-miss fallback must load with ``resolve_alias=True``.

    These pool workers are the third role in the same triangle as
    ``test_preload_and_fallback_resolve_the_same_tokenizer``: on a preload hit
    they get the alias-resolved tokenizer from the forkserver heap, so a
    ``resolve_alias=False`` fallback would tokenize the corpus with a DIFFERENT
    tokenizer depending only on whether the preload fired.
    """
    import ast
    import importlib
    import inspect

    module = importlib.import_module(module_name)
    tree = ast.parse(inspect.getsource(module))
    found = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "from_pretrained"):
            continue
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        assert "resolve_alias" in kwargs, f"{module_name}: fallback omits resolve_alias"
        assert isinstance(kwargs["resolve_alias"], ast.Constant)
        assert kwargs["resolve_alias"].value is True, (
            f"{module_name}: worker fallback uses resolve_alias=False, diverging "
            "from the forkserver preload"
        )
        found += 1
    assert found, f"{module_name}: no Tokenizer.from_pretrained fallback found"
