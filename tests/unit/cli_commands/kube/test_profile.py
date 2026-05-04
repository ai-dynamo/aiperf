# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for `aiperf kube profile` flag wiring.

Covers the `--skip-endpoint-check` flag: verifies it forwards into
`deploy_via_operator`/`deploy_direct` kwargs and, in operator mode, lands
on the submitted CR spec as `skipEndpointCheck=True` so the operator's
`_check_endpoint_reachable` handler can honor it.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from aiperf.cli_commands.kube.profile import profile
from aiperf.cli_commands.kube.profile_deploy import deploy_via_operator
from aiperf.config.v1 import ServiceConfig, UserConfig
from aiperf.operator.models import AIPerfJobSpec


def _user_config() -> UserConfig:
    """Minimal UserConfig that satisfies kube-CLI argument shape."""
    return UserConfig.model_validate(
        {
            "endpoint": {"model_names": ["m"], "urls": ["http://x"]},
        }
    )


class _StubKubeOptions:
    """Minimal stand-in for KubeOptions used by the profile command."""

    def __init__(self) -> None:
        self.name: str | None = None
        self.namespace: str | None = None
        self.kubeconfig: str | None = None
        self.kube_context: str | None = None
        self.image: str | None = "aiperf:latest"
        self.workers = 1


@pytest.mark.asyncio
async def test_profile_forwards_skip_endpoint_check_to_deploy_via_operator() -> None:
    """--skip-endpoint-check must arrive as a kwarg on deploy_via_operator."""
    kube_options = _StubKubeOptions()
    user_config = _user_config()
    service_config = ServiceConfig()
    fake_spec = {"benchmark": {"endpoint": {"url": "http://x"}}}
    fake_config = object()
    captured: dict[str, Any] = {}

    async def _capture_via_operator(*args: Any, **kwargs: Any) -> None:
        captured["args"] = args
        captured["kwargs"] = kwargs

    with (
        patch(
            "aiperf.cli_commands.kube.profile._resolve_spec_and_name",
            return_value=(fake_spec, fake_config, "bench-1"),
        ),
        patch("aiperf.cli_commands.kube.profile._print_memory_estimate"),
        patch(
            "aiperf.cli_commands.kube.profile_deploy.operator_available",
            new=AsyncMock(return_value=True),
        ),
        patch(
            "aiperf.cli_commands.kube.profile_deploy.deploy_via_operator",
            new=_capture_via_operator,
        ),
    ):
        await profile(
            user_config=user_config,
            service_config=service_config,
            kube_options=kube_options,
            skip_endpoint_check=True,
            dry_run=True,  # short-circuit operator_available so no cluster probe
        )

    assert captured["kwargs"].get("skip_endpoint_check") is True


@pytest.mark.asyncio
async def test_profile_forwards_skip_endpoint_check_to_deploy_direct() -> None:
    """--skip-endpoint-check must arrive as a kwarg on deploy_direct too."""
    kube_options = _StubKubeOptions()
    user_config = _user_config()
    service_config = ServiceConfig()
    fake_spec: dict[str, Any] = {}
    fake_config = object()
    captured: dict[str, Any] = {}

    async def _capture_direct(*args: Any, **kwargs: Any) -> None:
        captured["args"] = args
        captured["kwargs"] = kwargs

    with (
        patch(
            "aiperf.cli_commands.kube.profile._resolve_spec_and_name",
            return_value=(fake_spec, fake_config, "bench-1"),
        ),
        patch("aiperf.cli_commands.kube.profile._print_memory_estimate"),
        patch(
            "aiperf.cli_commands.kube.profile_deploy_direct.deploy_direct",
            new=_capture_direct,
        ),
    ):
        await profile(
            user_config=user_config,
            service_config=service_config,
            kube_options=kube_options,
            skip_endpoint_check=True,
            dry_run=True,
            no_operator=True,  # force direct path
        )

    assert captured["kwargs"].get("skip_endpoint_check") is True


@pytest.mark.asyncio
async def test_deploy_via_operator_injects_skip_endpoint_check_into_cr() -> None:
    """When skip_endpoint_check=True, the submitted CR spec carries skipEndpointCheck=True."""
    kube_options = _StubKubeOptions()
    spec: dict[str, Any] = {"benchmark": {"endpoint": {"url": "http://x"}}}
    config = type(
        "C",
        (),
        {
            "endpoint": type("E", (), {"urls": []})(),
            "get_model_names": lambda self: ["m"],
        },
    )()

    captured_cr: dict[str, Any] = {}

    def _capture_print(*args: Any, **kwargs: Any) -> None:
        captured_cr["printed"] = args[0] if args else kwargs.get("data")

    with patch("aiperf.kubernetes.console.console") as mock_console:
        mock_console.print.side_effect = _capture_print
        await deploy_via_operator(
            spec,
            kube_options,
            config,
            "bench-1",
            "aiperf",
            dry_run=True,  # take the json-print branch, no cluster
            detach=False,
            no_wait=False,
            attach_port=0,
            skip_endpoint_check=True,
        )

    assert spec.get("skipEndpointCheck") is True


def _minimal_benchmark(url: str = "http://x") -> dict:
    """Build a minimal valid AIPerfConfig dict for AIPerfJobSpec.benchmark."""
    return {
        "models": ["test-model"],
        "endpoint": {"url": url},
        "datasets": [
            {
                "name": "default",
                "type": "synthetic",
                "entries": 1,
                "prompts": {"isl": 8, "osl": 8},
            }
        ],
        "phases": [
            {
                "name": "default",
                "type": "concurrency",
                "requests": 1,
                "concurrency": 1,
            }
        ],
    }


def test_aiperfjobspec_reads_skip_endpoint_check_from_crd() -> None:
    """AIPerfJobSpec.model_validate must honor skipEndpointCheck from the raw CR."""
    crd_spec = {
        "image": "aiperf:latest",
        "skipEndpointCheck": True,
        "benchmark": _minimal_benchmark(),
    }
    validated = AIPerfJobSpec.model_validate(crd_spec)
    assert validated.skip_endpoint_check is True


def test_aiperfjobspec_skip_endpoint_check_defaults_false() -> None:
    """Absent skipEndpointCheck defaults to False (preserves prior behaviour)."""
    crd_spec = {
        "image": "aiperf:latest",
        "benchmark": _minimal_benchmark(),
    }
    validated = AIPerfJobSpec.model_validate(crd_spec)
    assert validated.skip_endpoint_check is False


# =============================================================================
# Adversarial regression-locks for the second-pass fixes (commit 793260d7b).
# `_check_config_file_for_sweep_keys` must redirect AIPerfSweep CRs to
# `aiperf kube sweep` and still enforce `_check_no_sweep_keys` for plain
# YAMLs that contain a top-level `sweep:`.
# =============================================================================


def test_check_config_file_for_sweep_keys_aiperfsweep_yaml_exits(tmp_path) -> None:
    """A YAML with `kind: AIPerfSweep` must redirect via raise_startup_error_and_exit."""
    from aiperf import cli_utils
    from aiperf.cli_commands.kube import profile as profile_mod

    config_file = tmp_path / "sweep.yaml"
    config_file.write_text(
        "apiVersion: aiperf.nvidia.com/v1alpha1\n"
        "kind: AIPerfSweep\n"
        "metadata: {name: x}\n"
        "spec:\n"
        "  multiRun: {trials: 2}\n"
        "  template:\n"
        "    spec:\n"
        "      benchmark: {endpoint: {urls: [http://x], type: chat}}\n"
    )

    captured: dict = {}

    def _fake_exit(message, **kwargs):
        captured["message"] = message
        captured["kwargs"] = kwargs
        raise SystemExit(1)

    with (
        patch.object(cli_utils, "raise_startup_error_and_exit", _fake_exit),
        pytest.raises(SystemExit),
    ):
        profile_mod._check_config_file_for_sweep_keys(config_file)

    assert "AIPerfSweep" in str(captured["message"])
    assert "aiperf kube sweep" in str(captured["message"])


def test_check_config_file_for_sweep_keys_aiperfjob_yaml_does_not_exit(
    tmp_path,
) -> None:
    """A YAML with `kind: AIPerfJob` is the CR path — no redirect, no exit."""
    from aiperf import cli_utils
    from aiperf.cli_commands.kube import profile as profile_mod

    config_file = tmp_path / "job.yaml"
    config_file.write_text(
        "apiVersion: aiperf.nvidia.com/v1alpha1\n"
        "kind: AIPerfJob\n"
        "metadata: {name: x}\n"
        "spec:\n"
        "  benchmark: {endpoint: {urls: [http://x], type: chat}}\n"
    )

    def _exploding_exit(*args, **kwargs):
        raise AssertionError("must not redirect for AIPerfJob CR YAML")

    with patch.object(cli_utils, "raise_startup_error_and_exit", _exploding_exit):
        # Must return cleanly.
        profile_mod._check_config_file_for_sweep_keys(config_file)


def test_check_config_file_for_sweep_keys_plain_yaml_with_sweep_key_exits(
    tmp_path,
) -> None:
    """Plain config (not a CR) with a top-level `sweep:` triggers the original
    `_check_no_sweep_keys` redirect — compat lock for that pre-existing path."""
    from aiperf import cli_utils
    from aiperf.cli_commands.kube import profile as profile_mod

    config_file = tmp_path / "plain.yaml"
    config_file.write_text(
        "models: [m]\n"
        "endpoint: {urls: [http://x], type: chat}\n"
        "datasets: [{name: main, type: synthetic}]\n"
        "phases: [{name: profiling, type: concurrency, duration: 1, concurrency: 1}]\n"
        "sweep:\n"
        "  type: grid\n"
        "  variables: {random_seed: [1, 2]}\n"
    )

    captured: dict = {}

    def _fake_exit(message, **kwargs):
        captured["message"] = message
        captured["kwargs"] = kwargs
        raise SystemExit(1)

    with (
        patch.object(cli_utils, "raise_startup_error_and_exit", _fake_exit),
        pytest.raises(SystemExit),
    ):
        profile_mod._check_config_file_for_sweep_keys(config_file)

    # Original check messages mention 'sweep' and direct user to `aiperf kube sweep`.
    msg = str(captured["message"])
    assert "sweep" in msg
    assert "aiperf kube sweep" in msg


# =============================================================================
# Regression-lock: `_build_cr_spec_and_config` must render Jinja2 templates in
# `spec.benchmark` BEFORE submission, mirroring `aiperf kube show -f`. The bug:
# previously the function validated `extract_benchmark_config(spec)` but kept
# the un-rendered raw spec, so `{{ total_concurrency }}` literals reached the
# operator and failed Pydantic int_parsing.
# =============================================================================


def _jinja_aiperfjob_cr_yaml() -> str:
    """A minimal AIPerfJob CR YAML using Jinja2 templates for concurrency/requests."""
    return (
        "apiVersion: aiperf.nvidia.com/v1alpha1\n"
        "kind: AIPerfJob\n"
        "metadata:\n"
        "  name: jinja-bench\n"
        "spec:\n"
        "  benchmark:\n"
        "    variables:\n"
        "      concurrency_per_gpu: 30\n"
        "      deployment_gpu_count: 4\n"
        '      total_concurrency: "{{ concurrency_per_gpu * deployment_gpu_count }}"\n'
        "      isl: 1024\n"
        "      osl: 1024\n"
        "    models: [test-model]\n"
        "    endpoint:\n"
        "      type: chat\n"
        "      urls: [http://server:8000]\n"
        "    datasets:\n"
        "      - name: main\n"
        "        type: synthetic\n"
        "        prompts:\n"
        '          isl: {mean: "{{ isl }}", stddev: 0}\n'
        '          osl: {mean: "{{ osl }}", stddev: 0}\n'
        "    phases:\n"
        "      - name: warmup\n"
        "        type: concurrency\n"
        '        concurrency: "{{ total_concurrency }}"\n'
        '        requests: "{{ total_concurrency }}"\n'
        "      - name: profiling\n"
        "        type: concurrency\n"
        '        concurrency: "{{ total_concurrency }}"\n'
        '        requests: "{{ total_concurrency * 10 }}"\n'
    )


def test_build_cr_spec_and_config_renders_jinja_in_benchmark(tmp_path) -> None:
    """`_build_cr_spec_and_config` must replace `{{ ... }}` literals with
    rendered scalars. Mirrors `aiperf kube show -f`'s pipeline: extract +
    expand_config_dict, then re-emit the validated AIPerfConfig back into
    `spec.benchmark` so the operator never sees raw templates."""
    import yaml

    from aiperf.cli_commands.kube import profile as profile_mod
    from aiperf.config.kube import KubeOptions

    config_file = tmp_path / "perf.yaml"
    config_file.write_text(_jinja_aiperfjob_cr_yaml())

    raw = yaml.safe_load(config_file.read_text())
    kube_options = KubeOptions(image="aiperf:latest", workers=1)

    spec, config = profile_mod._build_cr_spec_and_config(raw, kube_options)

    rendered_yaml = yaml.safe_dump(spec["benchmark"])
    assert "{{" not in rendered_yaml, (
        f"Jinja literals leaked into submitted spec.benchmark:\n{rendered_yaml}"
    )
    assert "}}" not in rendered_yaml

    # Walk the rendered phases to confirm scalars are real ints.
    phase_concurrencies = [p.get("concurrency") for p in spec["benchmark"]["phases"]]
    assert all(isinstance(c, int) for c in phase_concurrencies), (
        f"phases concurrency must be int after Jinja render, got: {phase_concurrencies}"
    )
    assert phase_concurrencies == [120, 120]  # 30 * 4

    # AIPerfConfig matches: drives memory estimate + connectionsPerWorker.
    assert config.benchmark.phases[0].concurrency == 120


@pytest.mark.asyncio
async def test_profile_dry_run_with_jinja_recipe_emits_resolved_cr(
    tmp_path, capsys
) -> None:
    """End-to-end CLI shape: `aiperf kube profile --dry-run -f <jinja-recipe>`
    prints a CR JSON whose spec.benchmark has no `{{ ... }}` literals."""
    import orjson

    from aiperf.cli_commands.kube.profile import profile
    from aiperf.config.kube import KubeOptions

    config_file = tmp_path / "perf.yaml"
    config_file.write_text(_jinja_aiperfjob_cr_yaml())

    user_config = UserConfig.model_validate({"config_file": str(config_file)})
    service_config = ServiceConfig()
    kube_options = KubeOptions(image="aiperf:latest", workers=1)

    captured: dict[str, Any] = {}

    def _capture_print(*args: Any, **kwargs: Any) -> None:
        captured.setdefault("chunks", []).append(args[0] if args else "")

    with (
        patch("aiperf.cli_commands.kube.profile._print_memory_estimate"),
        patch("aiperf.kubernetes.console.console") as mock_console,
    ):
        mock_console.print.side_effect = _capture_print
        await profile(
            user_config=user_config,
            service_config=service_config,
            kube_options=kube_options,
            dry_run=True,
        )

    output = "\n".join(str(c) for c in captured.get("chunks", []))
    assert "{{" not in output, f"Jinja leaked into dry-run CR output:\n{output}"
    assert "}}" not in output

    # The dry-run path emits orjson; locate the CR JSON chunk and verify shape.
    cr_chunk = next(
        (
            c
            for c in captured["chunks"]
            if isinstance(c, str) and c.lstrip().startswith("{")
        ),
        None,
    )
    assert cr_chunk is not None, "expected a JSON CR dump in console.print output"
    cr = orjson.loads(cr_chunk)
    phases = cr["spec"]["benchmark"]["phases"]
    assert all(isinstance(p["concurrency"], int) for p in phases)
    assert [p["concurrency"] for p in phases] == [120, 120]
