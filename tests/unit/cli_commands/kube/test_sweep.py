# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for `aiperf kube sweep` CR-builder helper."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import param

from aiperf.cli_commands.kube import sweep as sweep_cmd
from aiperf.config.v1 import UserConfig


def _kube_options_mock() -> MagicMock:
    """Stub that satisfies the attribute access in `_build_sweep_cr_dict`."""
    m = MagicMock()
    m.image = "x:latest"
    m.namespace = "ns"
    m.name = None
    m.kubeconfig = None
    m.kube_context = None
    deployment = MagicMock()
    deployment.model_dump = MagicMock(return_value={})
    m.to_deployment_config = MagicMock(return_value=deployment)
    return m


def test_build_sweep_cr_dict_emits_aiperfsweep_kind(tmp_path: Path) -> None:
    """YAML with sweep:+multi_run: produces an AIPerfSweep CR with hoisted spec."""
    config_file = tmp_path / "sweep.yaml"
    config_file.write_text(
        """
models: [Qwen/Qwen3-0.6B]
endpoint:
  urls: [http://localhost:8000/v1/chat/completions]
  type: chat
  streaming: true
datasets:
  - {name: main, type: synthetic}
phases:
  - name: profiling
    type: concurrency
    duration: 5
    concurrency: 1
sweep:
  type: grid
  variables:
    random_seed: [1, 2]
"""
    )
    cr = sweep_cmd._build_sweep_cr_dict(
        config_file=config_file,
        kube_options=_kube_options_mock(),
        multi_run_trials=2,
        cooldown_seconds=10,
        convergence_metric=None,
        convergence_min_runs=3,
        convergence_max_runs=10,
        convergence_threshold=0.05,
    )
    assert cr["kind"] == "AIPerfSweep"
    assert cr["apiVersion"] == "aiperf.nvidia.com/v1alpha1"
    assert "sweep" in cr["spec"]
    assert cr["spec"]["multiRun"]["trials"] == 2
    assert cr["spec"]["multiRun"]["cooldownSeconds"] == 10
    # benchmark in template should NOT have sweep:
    assert "sweep" not in cr["spec"]["template"]["spec"]["benchmark"]


def test_build_sweep_cr_dict_with_convergence(tmp_path: Path) -> None:
    """--convergence-metric populates spec.convergence with min/max/threshold."""
    config_file = tmp_path / "conf.yaml"
    config_file.write_text(
        """
models: [m]
endpoint:
  urls: [http://x]
  type: chat
datasets:
  - {name: main, type: synthetic}
phases:
  - {name: profiling, type: concurrency, duration: 1, concurrency: 1}
multi_run:
  cooldown_seconds: 30
"""
    )
    cr = sweep_cmd._build_sweep_cr_dict(
        config_file=config_file,
        kube_options=_kube_options_mock(),
        multi_run_trials=None,
        cooldown_seconds=0,
        convergence_metric="ttft_p99",
        convergence_min_runs=3,
        convergence_max_runs=7,
        convergence_threshold=0.05,
    )
    assert cr["spec"]["convergence"]["metric"] == "ttft_p99"
    assert cr["spec"]["convergence"]["maxRuns"] == 7
    # multi_run from yaml is preserved (cooldown_seconds present in original)
    assert "multiRun" in cr["spec"]


def test_build_sweep_cr_dict_requires_config_file() -> None:
    """No --config <file> raises a helpful ValueError."""
    with pytest.raises(ValueError, match="--config <file>"):
        sweep_cmd._build_sweep_cr_dict(
            config_file=None,
            kube_options=_kube_options_mock(),
            multi_run_trials=None,
            cooldown_seconds=0,
            convergence_metric=None,
            convergence_min_runs=3,
            convergence_max_runs=10,
            convergence_threshold=0.05,
        )


def test_build_sweep_cr_dict_default_name_from_config_stem(tmp_path: Path) -> None:
    """When no --name given, derive ``<stem>-sweep`` from the config file stem."""
    config_file = tmp_path / "concurrency_grid.yaml"
    config_file.write_text(
        """
models: [m]
endpoint: {urls: [http://x], type: chat}
datasets: [{name: main, type: synthetic}]
phases:
  - {name: profiling, type: concurrency, duration: 1, concurrency: 1}
sweep:
  type: grid
  variables: {random_seed: [1, 2]}
"""
    )
    cr = sweep_cmd._build_sweep_cr_dict(
        config_file=config_file,
        kube_options=_kube_options_mock(),
        multi_run_trials=None,
        cooldown_seconds=0,
        convergence_metric=None,
        convergence_min_runs=3,
        convergence_max_runs=10,
        convergence_threshold=0.05,
    )
    assert cr["metadata"]["name"] == "concurrency-grid-sweep"


def test_build_sweep_cr_dict_no_axis_raises_validation_error(tmp_path: Path) -> None:
    """Plain config (no sweep:/multi_run:/convergence:) is rejected by AIPerfSweepSpec.

    AIPerfSweep requires at least one axis; for a single benchmark, the user
    should use `aiperf kube profile` via AIPerfJob instead.
    """
    from pydantic import ValidationError

    config_file = tmp_path / "plain.yaml"
    config_file.write_text(
        """
models: [m]
endpoint: {urls: [http://x], type: chat}
datasets: [{name: main, type: synthetic}]
phases:
  - {name: profiling, type: concurrency, duration: 1, concurrency: 1}
"""
    )
    with pytest.raises(ValidationError, match="at least one of"):
        sweep_cmd._build_sweep_cr_dict(
            config_file=config_file,
            kube_options=_kube_options_mock(),
            multi_run_trials=None,
            cooldown_seconds=0,
            convergence_metric=None,
            convergence_min_runs=3,
            convergence_max_runs=10,
            convergence_threshold=0.05,
        )


def test_name_from_config_file_sanitizes_and_truncates() -> None:
    """Stem is lowercased, sanitized to [a-z0-9-], capped at 30 chars + '-sweep'."""
    out = sweep_cmd._name_from_config_file(Path("My_Crazy.Config.YAML"))
    assert out.endswith("-sweep")
    assert all(c.islower() or c.isdigit() or c == "-" for c in out)


def test_camelcase_multiRun_key_also_hoisted(tmp_path: Path) -> None:
    """Top-level ``multiRun:`` (camelCase) is honored as a fallback for ``multi_run:``."""
    config_file = tmp_path / "cc.yaml"
    config_file.write_text(
        """
models: [m]
endpoint: {urls: [http://x], type: chat}
datasets: [{name: main, type: synthetic}]
phases:
  - {name: profiling, type: concurrency, duration: 1, concurrency: 1}
multiRun:
  trials: 4
"""
    )
    cr = sweep_cmd._build_sweep_cr_dict(
        config_file=config_file,
        kube_options=_kube_options_mock(),
        multi_run_trials=None,
        cooldown_seconds=0,
        convergence_metric=None,
        convergence_min_runs=3,
        convergence_max_runs=10,
        convergence_threshold=0.05,
    )
    assert cr["spec"]["multiRun"]["trials"] == 4
    assert "multiRun" not in cr["spec"]["template"]["spec"]["benchmark"]


def test_build_sweep_cr_dict_expands_singular_shorthand(tmp_path: Path) -> None:
    """Bare AIPerfConfig YAML with `model:`/`dataset:`/`phases:` shorthand survives.

    Regression: prior to the fix, ``_build_sweep_cr_dict`` dumped the raw YAML
    straight into ``template.spec.benchmark`` without running it through
    AIPerfConfig validation, so ``model: foo`` (singular) reached
    ``AIPerfSweepSpec.model_validate`` unexpanded and tripped a
    ``benchmark.models field required`` error. The shorthand must promote
    just like ``aiperf kube profile -f`` does.
    """
    config_file = tmp_path / "shorthand.yaml"
    config_file.write_text(
        """
model: shorthand-model
endpoint:
  url: http://mock:8000
  type: chat
  streaming: true
dataset:
  type: synthetic
  prompts: {isl: 256, osl: 50}
phases:
  type: concurrency
  concurrency: 4
  requests: 30
sweep:
  type: grid
  variables:
    random_seed: [1, 2]
"""
    )
    cr = sweep_cmd._build_sweep_cr_dict(
        config_file=config_file,
        kube_options=_kube_options_mock(),
        multi_run_trials=None,
        cooldown_seconds=0,
        convergence_metric=None,
        convergence_min_runs=3,
        convergence_max_runs=10,
        convergence_threshold=0.05,
    )
    bench = cr["spec"]["template"]["spec"]["benchmark"]
    assert [m["name"] for m in bench["models"]["items"]] == ["shorthand-model"]
    # `dataset:` (singular) -> `datasets: [{name: default, ...}]`
    assert any(d.get("name") == "default" for d in bench["datasets"])
    # `phases:` (flat) -> `phases: [{name: default, ...}]`
    assert any(p.get("name") == "default" for p in bench["phases"])


def test_build_sweep_cr_dict_unwraps_aiperfjob_cr(tmp_path: Path) -> None:
    """Passing an AIPerfJob CR YAML extracts ``spec.benchmark`` rather than
    nesting the entire CR under ``template.spec.benchmark``.

    Regression: ``aiperf kube init`` produces an AIPerfJob CR; users who fed
    that file straight to ``aiperf kube sweep -f`` saw a stack of validation
    errors (``benchmark.models field required`` etc.) because the apiVersion /
    kind / metadata wrapper was being treated as benchmark fields.
    """
    config_file = tmp_path / "job-cr.yaml"
    config_file.write_text(
        """
apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfJob
metadata:
  name: my-bench
spec:
  benchmark:
    model: cr-model
    endpoint:
      url: http://mock:8000
      type: chat
      streaming: true
    dataset:
      type: synthetic
      prompts: {isl: 256, osl: 50}
    phases:
      type: concurrency
      concurrency: 8
      requests: 30
    sweep:
      type: grid
      variables:
        random_seed: [1, 2]
"""
    )
    cr = sweep_cmd._build_sweep_cr_dict(
        config_file=config_file,
        kube_options=_kube_options_mock(),
        multi_run_trials=None,
        cooldown_seconds=0,
        convergence_metric=None,
        convergence_min_runs=3,
        convergence_max_runs=10,
        convergence_threshold=0.05,
    )
    bench = cr["spec"]["template"]["spec"]["benchmark"]
    # CR wrapper keys are NOT inside benchmark
    for stray in ("apiVersion", "kind", "metadata", "spec"):
        assert stray not in bench, f"{stray!r} leaked into benchmark"
    assert [m["name"] for m in bench["models"]["items"]] == ["cr-model"]
    # sweep block hoisted out of benchmark to spec.sweep
    assert "sweep" not in bench
    # `type: grid` is the default and gets stripped by exclude_defaults; the
    # variables map is the load-bearing carrier that survives the dump.
    assert cr["spec"]["sweep"]["variables"] == {"random_seed": [1, 2]}


def test_build_sweep_cr_dict_rejects_aiperfsweep_cr(tmp_path: Path) -> None:
    """An AIPerfSweep CR is already a sweep; reject with a pointer to kubectl."""
    config_file = tmp_path / "sweep-cr.yaml"
    config_file.write_text(
        """
apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfSweep
metadata:
  name: existing
spec: {}
"""
    )
    with pytest.raises(ValueError, match="already an AIPerfSweep CR"):
        sweep_cmd._build_sweep_cr_dict(
            config_file=config_file,
            kube_options=_kube_options_mock(),
            multi_run_trials=None,
            cooldown_seconds=0,
            convergence_metric=None,
            convergence_min_runs=3,
            convergence_max_runs=10,
            convergence_threshold=0.05,
        )


# ---------------------------------------------------------------------------
# CLI override merge / canonicalization adversarial tests
# ---------------------------------------------------------------------------


def _bare_config(tmp_path: Path) -> Path:
    """Minimal long-form YAML; no recipe / sweep / multi_run preset."""
    p = tmp_path / "bare.yaml"
    p.write_text(
        """
models: [m]
endpoint: {urls: [http://x], type: chat, streaming: true}
datasets: [{name: main, type: synthetic, prompts: {isl: 64, osl: 32}}]
phases:
  - {name: profiling, type: concurrency, requests: 30, concurrency: 8}
"""
    )
    return p


def test_build_sweep_cr_dict_user_config_recipe_lands_adaptive_search(
    tmp_path: Path,
) -> None:
    """``--search-recipe max-throughput-ttft-sla --ttft-sla-ms 200`` paired
    with --config must populate spec.multiRun.adaptiveSearch (the bug fix
    everything in this round was for) -- and the round-trip canonicalization
    must surface the camelCase alias the K8s apiserver requires.
    """
    user = UserConfig.model_validate(
        {
            "endpoint": {"streaming": True},
            "loadgen": {
                "search_recipe": "max-throughput-ttft-sla",
                "ttft_sla_ms": 200.0,
            },
        }
    )
    cr = sweep_cmd._build_sweep_cr_dict(
        config_file=_bare_config(tmp_path),
        kube_options=_kube_options_mock(),
        user_config=user,
        multi_run_trials=None,
        cooldown_seconds=0,
        convergence_metric=None,
        convergence_min_runs=3,
        convergence_max_runs=10,
        convergence_threshold=0.05,
    )
    mr = cr["spec"]["multiRun"]
    # camelCase alias survives the by_alias round-trip; the apiserver rejects
    # snake_case `adaptive_search` even though the local validator accepts it.
    assert "adaptiveSearch" in mr or "adaptive_search" in mr
    ad = mr.get("adaptiveSearch") or mr.get("adaptive_search")
    assert ad["recipeName"] == "max-throughput-ttft-sla" or (
        ad.get("recipe_name") == "max-throughput-ttft-sla"
    )
    sla = ad.get("slaFilters") or ad.get("sla_filters") or []
    assert len(sla) == 1
    assert sla[0].get("threshold") == 200.0


def test_build_sweep_cr_dict_grid_recipe_lifts_to_spec_sweep(tmp_path: Path) -> None:
    """Grid recipes (concurrency-ramp, prefill-ttft-curve, decode-itl-curve)
    emit a sweep block that must hoist to ``spec.sweep`` -- not stay buried
    in ``spec.template.spec.benchmark``."""
    user = UserConfig.model_validate(
        {
            "endpoint": {"streaming": True},
            "loadgen": {"search_recipe": "concurrency-ramp"},
        }
    )
    cr = sweep_cmd._build_sweep_cr_dict(
        config_file=_bare_config(tmp_path),
        kube_options=_kube_options_mock(),
        user_config=user,
        multi_run_trials=None,
        cooldown_seconds=0,
        convergence_metric=None,
        convergence_min_runs=3,
        convergence_max_runs=10,
        convergence_threshold=0.05,
    )
    assert "sweep" in cr["spec"], "recipe-driven sweep must hoist to spec.sweep"
    assert "benchmark.phases.profiling.concurrency" in cr["spec"]["sweep"]["variables"]
    bench = cr["spec"]["template"]["spec"]["benchmark"]
    assert "sweep" not in bench, "sweep must NOT be embedded in benchmark"


def test_build_sweep_cr_dict_filters_grid_only_multirun_keys(tmp_path: Path) -> None:
    """Grid recipes' ``post_process`` and ``sla_filters`` are in-process-only
    (the K8s MultiRunConfig is `extra=forbid` and has no controller-side
    consumer for them yet). They must be stripped before bubbling up to
    ``spec.multiRun``, otherwise the apiserver rejects the CR."""
    user = UserConfig.model_validate(
        {
            "endpoint": {"streaming": True},
            "loadgen": {
                "search_recipe": "concurrency-ramp",
                "degradation_threshold": 0.20,
            },
        }
    )
    cr = sweep_cmd._build_sweep_cr_dict(
        config_file=_bare_config(tmp_path),
        kube_options=_kube_options_mock(),
        user_config=user,
        multi_run_trials=None,
        cooldown_seconds=0,
        convergence_metric=None,
        convergence_min_runs=3,
        convergence_max_runs=10,
        convergence_threshold=0.05,
    )
    mr = cr["spec"].get("multiRun", {}) or {}
    # post_process / sla_filters must NOT be at the multi_run level.
    assert "postProcess" not in mr
    assert "post_process" not in mr
    assert "slaFilters" not in mr
    assert "sla_filters" not in mr


def test_build_sweep_cr_dict_yaml_sweep_wins_over_recipe(tmp_path: Path) -> None:
    """When the YAML already declares a sweep block AND the user passes a
    grid recipe, the YAML wins (recipe overrides only fill in absent
    blocks). This keeps the user's hand-written sweep stable and avoids
    silently swapping their variables for the recipe's defaults."""
    config_file = tmp_path / "yaml-sweep.yaml"
    config_file.write_text(
        """
models: [m]
endpoint: {urls: [http://x], type: chat, streaming: true}
datasets: [{name: main, type: synthetic, prompts: {isl: 64, osl: 32}}]
phases:
  - {name: profiling, type: concurrency, requests: 30, concurrency: 8}
sweep:
  type: grid
  variables:
    random_seed: [1, 2, 3]
"""
    )
    user = UserConfig.model_validate(
        {
            "endpoint": {"streaming": True},
            "loadgen": {"search_recipe": "concurrency-ramp"},
        }
    )
    cr = sweep_cmd._build_sweep_cr_dict(
        config_file=config_file,
        kube_options=_kube_options_mock(),
        user_config=user,
        multi_run_trials=None,
        cooldown_seconds=0,
        convergence_metric=None,
        convergence_min_runs=3,
        convergence_max_runs=10,
        convergence_threshold=0.05,
    )
    # User's YAML sweep variables survive; recipe's concurrency variables
    # don't clobber them.
    assert cr["spec"]["sweep"]["variables"] == {"random_seed": [1, 2, 3]}


def test_build_sweep_cr_dict_no_user_config_keeps_yaml_intact(
    tmp_path: Path,
) -> None:
    """When user_config is None (the legacy direct-call path used by some
    older callers), the merge must be a no-op -- otherwise we'd start
    silently emitting empty endpoint/models/etc blocks that clobber YAML."""
    config_file = tmp_path / "yaml-only.yaml"
    config_file.write_text(
        """
models: [yaml-only-model]
endpoint: {urls: [http://yaml-only], type: chat}
datasets: [{name: main, type: synthetic, prompts: {isl: 64, osl: 32}}]
phases:
  - {name: profiling, type: concurrency, requests: 30, concurrency: 8}
sweep:
  type: grid
  variables: {random_seed: [1, 2]}
"""
    )
    cr = sweep_cmd._build_sweep_cr_dict(
        config_file=config_file,
        kube_options=_kube_options_mock(),
        user_config=None,
        multi_run_trials=None,
        cooldown_seconds=0,
        convergence_metric=None,
        convergence_min_runs=3,
        convergence_max_runs=10,
        convergence_threshold=0.05,
    )
    bench = cr["spec"]["template"]["spec"]["benchmark"]
    assert [m["name"] for m in bench["models"]["items"]] == ["yaml-only-model"]
    assert cr["spec"]["sweep"]["variables"] == {"random_seed": [1, 2]}


# =============================================================================
# Adversarial regression-locks for the just-landed bug-fixes.
# =============================================================================


_VALID_SWEEP_YAML = """
    benchmark:
      models: [m]
      endpoint: {urls: [http://x], type: chat}
      datasets:
        - {name: main, type: synthetic}
      phases:
        - {name: profiling, type: concurrency, duration: 1, concurrency: 1}
    sweep:
      type: grid
      variables: {benchmark.random_seed: [1, 2]}
"""


def _write_valid_sweep(tmp_path: Path, name: str = "sweep.yaml") -> Path:
    p = tmp_path / name
    p.write_text(_VALID_SWEEP_YAML)
    return p


def _build_kwargs() -> dict:
    """Default kwargs for ``_build_sweep_cr_dict`` so callers only override what matters."""
    return {
        "multi_run_trials": None,
        "cooldown_seconds": 0,
        "convergence_metric": None,
        "convergence_min_runs": 3,
        "convergence_max_runs": 10,
        "convergence_threshold": 0.05,
    }


# -----------------------------------------------------------------------------
# A) apiVersion regression-lock (commit 9f85d9eaf)
# -----------------------------------------------------------------------------


def test_build_sweep_cr_dict_apiversion_is_v1alpha1(tmp_path: Path) -> None:
    """Regression lock: apiVersion must be ``aiperf.nvidia.com/v1alpha1``, not ``v1``.

    A prior bug emitted ``apiVersion: aiperf.nvidia.com/v1`` which the apiserver
    rejected because the CRD is registered at v1alpha1 only.
    """
    cr = sweep_cmd._build_sweep_cr_dict(
        config_file=_write_valid_sweep(tmp_path),
        kube_options=_kube_options_mock(),
        **_build_kwargs(),
    )
    assert cr["apiVersion"] == "aiperf.nvidia.com/v1alpha1"
    assert cr["apiVersion"] != "aiperf.nvidia.com/v1"


async def test_submit_sweep_creates_with_v1alpha1_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression lock: the create-call must pass ``version="v1alpha1"``."""
    from contextlib import asynccontextmanager

    create_mock = AsyncMock()
    api_client = MagicMock()

    @asynccontextmanager
    async def fake_k8s_client(**_kwargs):
        yield api_client

    monkeypatch.setattr(
        "aiperf.kubernetes.client.k8s_client", fake_k8s_client, raising=True
    )
    monkeypatch.setattr(
        "kubernetes_asyncio.client.CustomObjectsApi",
        lambda _api: MagicMock(create_namespaced_custom_object=create_mock),
    )

    cr_dict = sweep_cmd._build_sweep_cr_dict(
        config_file=_write_valid_sweep(tmp_path),
        kube_options=_kube_options_mock(),
        **_build_kwargs(),
    )
    await sweep_cmd._submit_sweep(
        cr_dict=cr_dict, kube_options=_kube_options_mock(), namespace="ns"
    )

    create_mock.assert_awaited_once()
    kwargs = create_mock.await_args.kwargs
    assert kwargs["version"] == "v1alpha1"
    assert kwargs["group"] == "aiperf.nvidia.com"
    assert kwargs["plural"] == "aiperfsweeps"
    assert kwargs["namespace"] == "ns"


# -----------------------------------------------------------------------------
# B) JSON serialization regression-lock — model_dump must use mode="json"
# -----------------------------------------------------------------------------


def test_build_sweep_cr_dict_uses_model_dump_mode_json(tmp_path: Path) -> None:
    """Regression lock: deployment.model_dump(mode='json', ...) must be used.

    Without ``mode='json'`` Pydantic returns native Python types (e.g.
    ``datetime``, ``Path``, ``Enum``) which orjson chokes on at serialization
    time.
    """

    class FakeDeployment:
        def model_dump(self, **kwargs):
            if kwargs.get("mode") != "json":
                raise TypeError(
                    f"model_dump must be called with mode='json' — got {kwargs!r}"
                )
            assert kwargs.get("by_alias") is True
            assert kwargs.get("exclude_defaults") is True
            return {}

    kube_options = _kube_options_mock()
    kube_options.to_deployment_config = MagicMock(return_value=FakeDeployment())

    cr = sweep_cmd._build_sweep_cr_dict(
        config_file=_write_valid_sweep(tmp_path),
        kube_options=kube_options,
        **_build_kwargs(),
    )
    assert cr["kind"] == "AIPerfSweep"


# -----------------------------------------------------------------------------
# C) Console mandate — dry-run must not use builtins.print
# -----------------------------------------------------------------------------


async def test_sweep_dry_run_uses_kube_console_not_builtin_print(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression lock: dry-run must use ``kube_console.console.print``, not
    builtin print (the kube CLI forbids builtin print per CLAUDE.md)."""
    import builtins

    from aiperf.config.v1 import ServiceConfig, UserConfig
    from aiperf.kubernetes import console as kube_console

    config_file = _write_valid_sweep(tmp_path)
    user_config = UserConfig.model_validate(
        {"endpoint": {"model_names": ["m"], "urls": ["http://x"]}}
    )
    user_config.config_file = config_file

    def _exploding_print(*_a, **_kw):
        raise AssertionError("builtins.print must not be used by the sweep CLI")

    monkeypatch.setattr(builtins, "print", _exploding_print)

    captured: list[str] = []

    def _capture(*args, **_kw):
        captured.append(" ".join(str(a) for a in args))

    monkeypatch.setattr(kube_console.console, "print", _capture)

    await sweep_cmd.sweep(
        user_config=user_config,
        service_config=ServiceConfig(),
        kube_options=_kube_options_mock(),
        dry_run=True,
    )

    assert len(captured) == 1
    assert "AIPerfSweep" in captured[0]
    assert "v1alpha1" in captured[0]


# -----------------------------------------------------------------------------
# D) AlreadyExists translation — 409 -> RuntimeError mentioning --name
# -----------------------------------------------------------------------------


async def test_submit_sweep_translates_409_into_user_facing_runtime_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ApiException(status=409) must surface a RuntimeError whose message
    mentions 'already exists' and tells the user to use ``--name``."""
    from contextlib import asynccontextmanager

    from kubernetes_asyncio.client import ApiException

    api_client = MagicMock()

    @asynccontextmanager
    async def fake_k8s_client(**_kwargs):
        yield api_client

    monkeypatch.setattr(
        "aiperf.kubernetes.client.k8s_client", fake_k8s_client, raising=True
    )
    monkeypatch.setattr(
        "kubernetes_asyncio.client.CustomObjectsApi",
        lambda _api: MagicMock(
            create_namespaced_custom_object=AsyncMock(
                side_effect=ApiException(status=409, reason="AlreadyExists")
            )
        ),
    )

    cr_dict = sweep_cmd._build_sweep_cr_dict(
        config_file=_write_valid_sweep(tmp_path),
        kube_options=_kube_options_mock(),
        **_build_kwargs(),
    )
    with pytest.raises(RuntimeError, match=r"(?s)already exists.*--name"):
        await sweep_cmd._submit_sweep(
            cr_dict=cr_dict, kube_options=_kube_options_mock(), namespace="ns"
        )


async def test_submit_sweep_does_not_translate_non_409_api_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-409 ApiException must propagate untouched — only 409 is translated."""
    from contextlib import asynccontextmanager

    from kubernetes_asyncio.client import ApiException

    api_client = MagicMock()

    @asynccontextmanager
    async def fake_k8s_client(**_kwargs):
        yield api_client

    monkeypatch.setattr(
        "aiperf.kubernetes.client.k8s_client", fake_k8s_client, raising=True
    )
    monkeypatch.setattr(
        "kubernetes_asyncio.client.CustomObjectsApi",
        lambda _api: MagicMock(
            create_namespaced_custom_object=AsyncMock(
                side_effect=ApiException(status=500, reason="Internal")
            )
        ),
    )

    cr_dict = sweep_cmd._build_sweep_cr_dict(
        config_file=_write_valid_sweep(tmp_path),
        kube_options=_kube_options_mock(),
        **_build_kwargs(),
    )
    with pytest.raises(ApiException):
        await sweep_cmd._submit_sweep(
            cr_dict=cr_dict, kube_options=_kube_options_mock(), namespace="ns"
        )


# -----------------------------------------------------------------------------
# E) Pre-submission validation gates (rule-4 broadening)
# -----------------------------------------------------------------------------


def test_build_sweep_cr_dict_template_spec_sweep_raises_validation_error() -> None:
    """A sweep block placed under template.spec must be rejected by
    AIPerfSweepSpec at submit time — rule-4 broadening from commit
    ``3f772c0de``-era fix."""
    from pydantic import ValidationError

    from aiperf.kubernetes.sweep_models import AIPerfSweepSpec

    bad_spec = {
        "multiRun": {"trials": 3},
        "template": {
            "spec": {
                "benchmark": {},
                "sweep": {"type": "grid", "variables": {"random_seed": [1, 2]}},
            }
        },
    }
    with pytest.raises(ValidationError, match=r"template\.spec\.sweep"):
        AIPerfSweepSpec.model_validate(bad_spec)


def test_build_sweep_cr_dict_convergence_under_benchmark_raises() -> None:
    """``convergence`` under template.spec.benchmark must be rejected (rule 4
    broadened beyond just ``sweep``/``multiRun``)."""
    from pydantic import ValidationError

    from aiperf.kubernetes.sweep_models import AIPerfSweepSpec

    bad_spec = {
        "multiRun": {"cooldownSeconds": 5},
        "template": {
            "spec": {
                "benchmark": {
                    "convergence": {"metric": "ttft_p99"},
                }
            }
        },
    }
    with pytest.raises(
        ValidationError, match=r"template\.spec\.benchmark\.convergence"
    ):
        AIPerfSweepSpec.model_validate(bad_spec)


# -----------------------------------------------------------------------------
# F) ConvergenceConfig — min_runs > max_runs
# -----------------------------------------------------------------------------


def test_convergence_config_min_runs_gt_max_runs_raises_validation_error() -> None:
    """``min_runs`` must be <= ``max_runs``."""
    from pydantic import ValidationError

    from aiperf.kubernetes.sweep_models import ConvergenceConfig

    with pytest.raises(ValidationError, match=r"must be <="):
        ConvergenceConfig(metric="ttft_p99", min_runs=10, max_runs=5)


def test_build_sweep_cr_dict_min_runs_gt_max_runs_surfaces_validation_error(
    tmp_path: Path,
) -> None:
    """CLI flag combination ``--min-runs 10 --max-runs 5`` must be rejected."""
    from pydantic import ValidationError

    config_file = _write_valid_sweep(tmp_path)
    with pytest.raises(ValidationError, match=r"must be <="):
        sweep_cmd._build_sweep_cr_dict(
            config_file=config_file,
            kube_options=_kube_options_mock(),
            multi_run_trials=None,
            cooldown_seconds=0,
            convergence_metric="ttft_p99",
            convergence_min_runs=10,
            convergence_max_runs=5,
            convergence_threshold=0.05,
        )


# -----------------------------------------------------------------------------
# I) DNS-label safety for _name_from_config_file
# -----------------------------------------------------------------------------


def test_name_from_config_file_underscores_only_does_not_start_with_hyphen() -> None:
    """``_name_from_config_file(Path('___'))`` must NOT return a name starting
    with '-' — DNS-1123 labels must start with [a-z0-9]."""
    out = sweep_cmd._name_from_config_file(Path("___"))
    assert not out.startswith("-")
    assert re.fullmatch(r"[a-z0-9][a-z0-9-]*", out)
    assert out.endswith("-sweep")


@pytest.mark.parametrize(
    "stem",
    [
        param("simple", id="lowercase-plain"),
        param("My.Mixed.Case", id="mixed-case-with-dots"),
        param("_leading_underscore", id="leading-underscore"),
        param("a" * 200, id="very-long"),
        param("___", id="all-underscores-sanitizes-to-empty"),
        param("...", id="all-dots-sanitizes-to-empty"),
    ],
)  # fmt: skip
def test_name_from_config_file_produces_valid_dns1123_label(stem: str) -> None:
    """Every output is a valid DNS-1123 label — ``^[a-z0-9][a-z0-9-]*$`` and
    ends with ``-sweep``."""
    out = sweep_cmd._name_from_config_file(Path(f"{stem}.yaml"))
    assert out.endswith("-sweep")
    assert re.fullmatch(r"[a-z0-9][a-z0-9-]*", out), (
        f"name {out!r} from stem {stem!r} is not a valid DNS-1123 label"
    )
    assert len(out) <= 63


# DNS-1123 hardening regression-locks (third-pass fix).
# A strict DNS-1123 *label* (single component, no dots) follows
# ``^[a-z0-9]([-a-z0-9]*[a-z0-9])?$`` and is at most 63 chars. Consecutive
# hyphens are technically legal, but we collapse them so the user-visible
# name doesn't surface ugly ``--`` runs.
_DNS_1123_LABEL_STRICT = r"[a-z0-9]([-a-z0-9]*[a-z0-9])?"


@pytest.mark.parametrize(
    "stem, must_not_contain",
    [
        param("__a__b__", "--", id="collapse-runs-of-special-chars"),
        param("a" * 29 + "-", "--", id="trailing-hyphen-after-truncation"),
        param("---abc---", "--", id="leading-and-trailing-hyphens"),
        param("a..b..c", "--", id="dots-do-not-double-collapse"),
        param("foo!@#bar", "--", id="adjacent-specials-collapse-once"),
    ],
)  # fmt: skip
def test_name_from_config_file_collapses_consecutive_hyphens(
    stem: str, must_not_contain: str
) -> None:
    """No ``--`` runs survive in the output — sanitization collapses runs."""
    out = sweep_cmd._name_from_config_file(Path(f"{stem}.yaml"))
    assert must_not_contain not in out, (
        f"name {out!r} from stem {stem!r} contains {must_not_contain!r}"
    )


@pytest.mark.parametrize(
    "stem",
    [
        param("a" * 30, id="cap-fits-exactly"),
        param("a" * 31, id="cap-truncates-by-one"),
        param("a" * 200, id="far-over-cap"),
        param("a" * 30 + "-bar", id="cap-falls-on-hyphen-after-prefix"),
        param("ab-" * 20, id="cap-falls-mid-hyphen-run"),
        param("a-b-c-d-e-f-g-h-i-j-k-l-m-n-o-p", id="cap-bisects-trailing-hyphen"),
    ],
)  # fmt: skip
def test_name_from_config_file_strict_dns1123_after_truncation(stem: str) -> None:
    """Output matches strict DNS-1123 label form even after the 30-char cap."""
    out = sweep_cmd._name_from_config_file(Path(f"{stem}.yaml"))
    assert re.fullmatch(_DNS_1123_LABEL_STRICT, out), (
        f"name {out!r} from stem {stem!r} fails strict DNS-1123 (start/end alnum, no consecutive '-')"
    )
    assert out.endswith("-sweep")
    assert len(out) <= 63


def test_name_from_config_file_all_digits_stem_is_valid() -> None:
    """All-digit stems produce a valid label (DNS-1123 labels may start with a digit)."""
    out = sweep_cmd._name_from_config_file(Path("123456.yaml"))
    assert out == "123456-sweep"
    assert re.fullmatch(_DNS_1123_LABEL_STRICT, out)


def test_name_from_config_file_empty_stem_falls_back_to_aiperf() -> None:
    """A stem that sanitizes to empty (all special chars) falls back to ``aiperf``."""
    out = sweep_cmd._name_from_config_file(Path("___.yaml"))
    assert out == "aiperf-sweep"


def test_name_from_config_file_unicode_stem_sanitizes_safely() -> None:
    """Non-ASCII chars (emoji, accented) are replaced with ``-`` and collapsed."""
    out = sweep_cmd._name_from_config_file(Path("héllo🚀world.yaml"))
    assert re.fullmatch(_DNS_1123_LABEL_STRICT, out)
    assert out.endswith("-sweep")


# =============================================================================
# Adversarial regression-locks for second-pass fixes (commit 793260d7b).
# `--trials` must OVERRIDE YAML multi_run.trials (was setdefault, so YAML won),
# and `_submit_sweep` must call `kube_console.save_last_benchmark` exactly once.
# =============================================================================


def test_build_sweep_cr_dict_trials_cli_overrides_yaml_trials(tmp_path: Path) -> None:
    """YAML has multi_run.trials=3; CLI passes --trials 10 → CLI wins (10).

    Previously used setdefault, so YAML silently won. The fix flips to direct
    assignment, matching the documented help text.
    """
    config_file = tmp_path / "sweep.yaml"
    config_file.write_text(
        """
models: [m]
endpoint: {urls: [http://x], type: chat}
datasets:
  - {name: main, type: synthetic}
phases:
  - {name: profiling, type: concurrency, duration: 1, concurrency: 1}
multi_run:
  trials: 3
sweep:
  type: grid
  variables: {random_seed: [1, 2]}
"""
    )
    cr = sweep_cmd._build_sweep_cr_dict(
        config_file=config_file,
        kube_options=_kube_options_mock(),
        multi_run_trials=10,
        cooldown_seconds=0,
        convergence_metric=None,
        convergence_min_runs=3,
        convergence_max_runs=10,
        convergence_threshold=0.05,
    )
    assert cr["spec"]["multiRun"]["trials"] == 10, (
        "CLI --trials must override YAML multi_run.trials (was 3, --trials 10)"
    )


def test_build_sweep_cr_dict_trials_cli_when_yaml_has_no_multirun(
    tmp_path: Path,
) -> None:
    """No multi_run in YAML; --trials 5 → multiRun.trials=5 in CR."""
    config_file = tmp_path / "sweep.yaml"
    config_file.write_text(
        """
models: [m]
endpoint: {urls: [http://x], type: chat}
datasets:
  - {name: main, type: synthetic}
phases:
  - {name: profiling, type: concurrency, duration: 1, concurrency: 1}
sweep:
  type: grid
  variables: {random_seed: [1, 2]}
"""
    )
    cr = sweep_cmd._build_sweep_cr_dict(
        config_file=config_file,
        kube_options=_kube_options_mock(),
        multi_run_trials=5,
        cooldown_seconds=0,
        convergence_metric=None,
        convergence_min_runs=3,
        convergence_max_runs=10,
        convergence_threshold=0.05,
    )
    assert cr["spec"]["multiRun"]["trials"] == 5


async def test_submit_sweep_calls_save_last_benchmark_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_submit_sweep` must call `kube_console.save_last_benchmark` once with
    (cr_name, namespace) for parity with `aiperf kube profile`."""
    from contextlib import asynccontextmanager

    api_client = MagicMock()

    @asynccontextmanager
    async def fake_k8s_client(**_kwargs):
        yield api_client

    monkeypatch.setattr(
        "aiperf.kubernetes.client.k8s_client", fake_k8s_client, raising=True
    )
    monkeypatch.setattr(
        "kubernetes_asyncio.client.CustomObjectsApi",
        lambda _api: MagicMock(create_namespaced_custom_object=AsyncMock()),
    )

    save_mock = MagicMock()
    monkeypatch.setattr(
        "aiperf.kubernetes.console.save_last_benchmark", save_mock, raising=True
    )

    cr_dict = sweep_cmd._build_sweep_cr_dict(
        config_file=_write_valid_sweep(tmp_path),
        kube_options=_kube_options_mock(),
        **_build_kwargs(),
    )
    cr_name = cr_dict["metadata"]["name"]

    await sweep_cmd._submit_sweep(
        cr_dict=cr_dict, kube_options=_kube_options_mock(), namespace="ns"
    )

    save_mock.assert_called_once()
    pos = save_mock.call_args.args
    # First two positional args are (cr_name, namespace).
    assert pos[0] == cr_name
    assert pos[1] == "ns"


def test_build_sweep_cr_dict_renders_jinja_in_benchmark(tmp_path: Path) -> None:
    """`_build_sweep_cr_dict` must render `{{ ... }}` literals before stuffing
    raw YAML into `spec.template.spec.benchmark`. Without this, unresolved
    Jinja literals trip AIPerfSweepSpec.model_validate's int_parsing on
    `phases[].concurrency`, mirroring the bug fixed in `aiperf kube profile -f`.
    """
    config_file = tmp_path / "sweep.yaml"
    config_file.write_text(
        """
variables:
  base_concurrency: 30
  multiplier: 4
benchmark:
  models: [Qwen/Qwen3-0.6B]
  endpoint:
    urls: [http://localhost:8000/v1/chat/completions]
    type: chat
    streaming: true
  datasets:
    - {name: main, type: synthetic}
  phases:
    - name: profiling
      type: concurrency
      duration: 5
      concurrency: "{{ base_concurrency * multiplier }}"
sweep:
  type: grid
  variables:
    benchmark.phases.profiling.duration: [5, 10]
"""
    )
    cr = sweep_cmd._build_sweep_cr_dict(
        config_file=config_file,
        kube_options=_kube_options_mock(),
        multi_run_trials=2,
        cooldown_seconds=10,
        convergence_metric=None,
        convergence_min_runs=3,
        convergence_max_runs=10,
        convergence_threshold=0.05,
    )

    benchmark = cr["spec"]["template"]["spec"]["benchmark"]
    import yaml

    rendered = yaml.safe_dump(benchmark)
    assert "{{" not in rendered, (
        f"Jinja literals leaked into spec.template.spec.benchmark:\n{rendered}"
    )
    assert "}}" not in rendered

    phase_concurrencies = [p["concurrency"] for p in benchmark["phases"]]
    assert all(isinstance(c, int) for c in phase_concurrencies), (
        f"phases[].concurrency must be int after Jinja render, got: {phase_concurrencies}"
    )
    assert phase_concurrencies == [120]  # 30 * 4
