# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Loader mp-context platform branch (DM6) + preload env triple (DM4).

Platform-conditional code must branch on ``IS_LINUX`` from
``aiperf.common.constants`` -- never on ``platform.system()`` inline. The
constant is also what makes the branch patchable here.

The preload env triple ``(name, trust_remote_code, revision)`` is how the run
config reaches the forkserver helper AND every pool worker; DM4 was the loader
silently preloading/synthesizing with ``(False, "main")`` for runs pinned to a
non-``main`` tokenizer revision, and resolving aliases differently on a preload
hit vs an on-demand fallback.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

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


@pytest.fixture
def clean_loader_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Fresh context cache + no preload env vars, spawn context (no forkserver)."""
    monkeypatch.setattr(mp_context, "_loader_ctx", None)
    monkeypatch.setattr(mp_context, "IS_LINUX", False)
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_get_loader_mp_context_uses_is_linux_constant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With IS_LINUX False the loader context must be a spawn context (no
    forkserver helper). Patching the module constant only works because the
    module imports IS_LINUX; the inline platform.system() call it replaced was
    not patchable and violated the constants contract."""
    monkeypatch.setattr(mp_context, "_loader_ctx", None)
    monkeypatch.setattr(mp_context, "IS_LINUX", False)

    ctx = mp_context.get_loader_mp_context()

    assert ctx.get_start_method() == "spawn"


def test_get_loader_mp_context_defaults_env_triple_when_unconfigured(
    clean_loader_env: pytest.MonkeyPatch,
) -> None:
    mp_context.get_loader_mp_context(preload_tokenizer="some/tok")

    assert os.environ[mp_context._ENV_PRELOAD_NAME] == "some/tok"
    assert os.environ[mp_context._ENV_PRELOAD_TRUST] == "false"
    assert os.environ[mp_context._ENV_PRELOAD_REVISION] == "main"


def test_get_loader_mp_context_respects_configured_trust_and_revision(
    clean_loader_env: pytest.MonkeyPatch,
) -> None:
    """The name-only call (the only production call shape, via
    ``_loader_pool_context``) must NOT clobber the run-config trust/revision
    published by ``configure_loader_tokenizer_env``."""
    mp_context.configure_loader_tokenizer_env(
        trust_remote_code=True, revision="deadbeef"
    )
    mp_context.get_loader_mp_context(preload_tokenizer="some/tok")

    assert os.environ[mp_context._ENV_PRELOAD_NAME] == "some/tok"
    assert os.environ[mp_context._ENV_PRELOAD_TRUST] == "true"
    assert os.environ[mp_context._ENV_PRELOAD_REVISION] == "deadbeef"


def test_get_loader_mp_context_explicit_args_override_configured_env(
    clean_loader_env: pytest.MonkeyPatch,
) -> None:
    mp_context.configure_loader_tokenizer_env(trust_remote_code=True, revision="r1")
    mp_context.get_loader_mp_context(
        preload_tokenizer="some/tok", trust_remote_code=False, revision="r2"
    )

    assert os.environ[mp_context._ENV_PRELOAD_TRUST] == "false"
    assert os.environ[mp_context._ENV_PRELOAD_REVISION] == "r2"


@pytest.mark.skipif(IS_WINDOWS, reason="forkserver context does not exist on Windows")
def test_get_loader_mp_context_second_call_skips_forkserver_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cached context must not re-enter the stdio dup2 swap.

    The GraphStoreBuilder pre-starts the forkserver on the event loop (a
    known-quiet point) precisely so the pool's later call from inside the
    offloaded ``asyncio.to_thread`` build is a cached no-op; a second
    ``_eagerly_start_forkserver`` there would race the process-wide dup2
    stdio swap against live event-loop logging.
    """
    starts: list[None] = []
    monkeypatch.setattr(mp_context, "_loader_ctx", None)
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
    """parse_graph_workload is the one seam every graph parse goes through, so
    it must publish the run's tokenizer trust/revision triple itself: a direct
    caller (tooling, tests) has no DatasetManager configure step to do it, and
    the forkserver helper snapshots the env once at spawn.

    Real resolved config on purpose (a mock would hide wrong-path reads); only
    the HF tokenizer load itself is mocked.
    """
    monkeypatch = clean_loader_env
    from aiperf.config.flags.cli_config import CLIConfig
    from aiperf.dataset.graph.workload_detect import parse_graph_workload
    from tests.unit.conftest import make_run_from_cli

    weka_min = Path(__file__).parent.parent / "graph" / "fixtures" / "weka_min.json"
    run = make_run_from_cli(
        CLIConfig(
            model_names=["test-model"],
            input_file=str(weka_min),
            tokenizer_name="builtin",
            trust_remote_code=True,
            tokenizer_revision="pinned-rev",
        )
    )

    # A direct caller with no DatasetManager configure step (tooling / tests).
    parse_graph_workload(run, weka_min)

    assert os.environ[mp_context._ENV_PRELOAD_TRUST] == "true"
    assert os.environ[mp_context._ENV_PRELOAD_REVISION] == "pinned-rev"

    # ...and the synthesizer layer loads with exactly that triple: the
    # worker-role fallback in ``content._build_generator`` on a preload miss.
    calls: list[dict] = []

    def recording_from_pretrained(name: str, **kwargs) -> SimpleNamespace:
        calls.append({"name": name, **kwargs})
        return SimpleNamespace()

    class _StubGenerator:
        def __init__(self, config, tokenizer) -> None:
            self.config = config
            self.tokenizer = tokenizer
            self._hash_id_corpus_rng = None

    monkeypatch.setattr(Tokenizer, "from_pretrained", recording_from_pretrained)
    monkeypatch.setattr(
        "aiperf.dataset._tokenizer_preload.get_preloaded", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "aiperf.dataset.generator.coding_content.CodingContentGenerator",
        _StubGenerator,
    )
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
    """A preload hit and an on-demand fallback miss must load with the SAME
    ``(name, trust_remote_code, revision, resolve_alias)`` arguments, or an
    aliased name silently synthesizes with a different tokenizer depending on
    whether the forkserver preload fired."""
    monkeypatch = clean_loader_env
    calls: list[dict] = []

    def recording_from_pretrained(name: str, **kwargs) -> SimpleNamespace:
        calls.append({"name": name, **kwargs})
        return SimpleNamespace()

    # One patch covers both call sites: content.py's module-level ``Tokenizer``
    # and _tokenizer_preload's local import are the same class object.
    monkeypatch.setattr(Tokenizer, "from_pretrained", recording_from_pretrained)
    monkeypatch.setattr(tokenizer_preload, "_LOADED", {})
    monkeypatch.setenv(mp_context._ENV_PRELOAD_NAME, "gpt2")
    monkeypatch.setenv(mp_context._ENV_PRELOAD_TRUST, "true")
    monkeypatch.setenv(mp_context._ENV_PRELOAD_REVISION, "pinned-rev")

    # Forkserver-helper role: the preload load.
    tokenizer_preload._preload()

    # Worker role on a preload MISS: the on-demand fallback in
    # content._build_generator. Stub out the corpus generator so only the
    # tokenizer-load call shape is exercised.
    class _StubGenerator:
        def __init__(self, config, tokenizer) -> None:
            self.config = config
            self.tokenizer = tokenizer
            self._hash_id_corpus_rng = None

    monkeypatch.setattr(
        "aiperf.dataset.generator.coding_content.CodingContentGenerator",
        _StubGenerator,
    )
    monkeypatch.setattr(
        "aiperf.dataset._tokenizer_preload.get_preloaded", lambda *a, **k: None
    )
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
