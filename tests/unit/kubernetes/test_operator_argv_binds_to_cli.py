# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract guard: every argv the operator emits must bind against the real CLI.

The operator builds pod `command`/`args` in `aiperf.kubernetes.jobset_helpers`
and `aiperf.kubernetes.jobset_builder`, while the flags themselves are declared
in `aiperf.cli_commands.*`. Nothing else couples the two sides, so a flag can be
renamed or dropped on one side and every static check still passes -- the
mismatch only surfaces as a CrashLoopBackOff on a real cluster. These tests
bind the emitted argv with the actual cyclopts app so the mismatch fails in CI.
"""

import importlib.util
from pathlib import Path

import pytest
from cyclopts import App

from aiperf.cli import app as aiperf_app
from aiperf.kubernetes.jobset_helpers import build_container_args


def _bind(argv: list[str]) -> None:
    """Resolve argv against the real aiperf app, raising on any binding error.

    Uses `App.parse_args` with `exit_on_error=False` so cyclopts raises instead
    of calling `sys.exit`, and stops short of invoking the command body.
    """
    aiperf_app.parse_args(argv, exit_on_error=False, print_error=False)


@pytest.mark.parametrize(
    ("service_type", "health_port", "api_port", "service_id"),
    [
        pytest.param("worker_group_manager", 8080, None, None, id="wgm"),
        pytest.param("system_controller", 8080, None, None, id="controller"),
        pytest.param("api", 8080, 8081, None, id="api-with-api-port"),
        pytest.param("worker", 8080, None, "worker-3", id="worker-with-service-id"),
        pytest.param("record_processor", None, None, None, id="no-optional-flags"),
    ],
)  # fmt: skip
def test_operator_service_argv_binds_to_cli(
    service_type: str,
    health_port: int | None,
    api_port: int | None,
    service_id: str | None,
) -> None:
    """The `aiperf service` argv emitted for a pod must bind without error."""
    _bind(build_container_args(service_type, health_port, api_port, service_id))


def test_operator_event_bus_proxy_argv_binds_to_cli() -> None:
    """The `aiperf proxy` argv emitted for the event-bus sidecar must bind."""
    _bind(
        [
            "proxy",
            "--kind",
            "event_bus",
            "--benchmark-run",
            "/etc/aiperf/run_config.json",
            "--health-port",
            "8085",
        ]
    )


@pytest.mark.parametrize(
    "command_name",
    [
        pytest.param("service", id="service"),
        pytest.param("proxy", id="proxy"),
    ],
)  # fmt: skip
def test_operator_emitted_command_is_registered(command_name: str) -> None:
    """Every `aiperf <cmd>` the operator invokes must exist on the CLI app."""
    assert command_name in aiperf_app, (
        f"operator emits `aiperf {command_name}` but that subcommand is not "
        f"registered in aiperf.cli"
    )
    assert isinstance(aiperf_app[command_name], App)


@pytest.mark.parametrize(
    "module_name",
    [
        pytest.param("aiperf.kubernetes.results_sidecar", id="results-sidecar"),
        pytest.param("aiperf.sweep_controller.main", id="sweep-controller"),
    ],
)  # fmt: skip
def test_operator_emitted_python_module_is_importable(module_name: str) -> None:
    """`python -m <module>` container commands must resolve to real modules."""
    assert importlib.util.find_spec(module_name) is not None


def test_service_bootstraps_from_operator_configmap_and_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Binding is not enough -- the emitted argv must reach service bootstrap.

    Serializes a BenchmarkRun through the exact ConfigMap path the operator
    uses, writes it where the pod mounts it, and drives the CLI with the exact
    argv `build_container_args` emits.
    """
    from aiperf.cli_runner import _make_benchmark_run
    from aiperf.common.environment import Environment
    from aiperf.config.flags import CLIConfig
    from aiperf.config.flags.resolver import resolve_config
    from aiperf.config.loader import build_benchmark_plan
    from aiperf.config.resolution.plan import BenchmarkRun
    from aiperf.kubernetes.resources import ConfigMapSpec
    from aiperf.orchestrator.orchestrator import resolve_run_seed

    # --health-port mutates process-global Environment state; scope it to the test.
    monkeypatch.setattr(
        Environment.SERVICE, "HEALTH_ENABLED", Environment.SERVICE.HEALTH_ENABLED
    )
    monkeypatch.setattr(
        Environment.SERVICE, "HEALTH_PORT", Environment.SERVICE.HEALTH_PORT
    )

    plan = build_benchmark_plan(
        resolve_config(CLIConfig(model_names=["m"], url="http://127.0.0.1:8000"), None)
    )
    run = _make_benchmark_run(
        plan.configs[0],
        random_seed=resolve_run_seed(plan, plan.variations[0]),
    )
    config_map = ConfigMapSpec.from_benchmark_run("cm", "ns", run, "job-1")
    run_file = tmp_path / "run_config.json"
    run_file.write_text(config_map.data["run_config.json"])

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "aiperf.common.bootstrap.bootstrap_and_run_service",
        lambda **kwargs: captured.update(kwargs),
    )

    argv = build_container_args("worker_group_manager", 8080, None, None)
    # The operator mounts the ConfigMap at a fixed path; point at the tmp copy.
    argv[argv.index("--benchmark-run") + 1] = str(run_file)
    # cyclopts always exits the interpreter after a successful command body.
    with pytest.raises(SystemExit) as exc:
        aiperf_app(argv, exit_on_error=False, print_error=False)
    assert exc.value.code == 0

    assert isinstance(captured.get("run"), BenchmarkRun)
