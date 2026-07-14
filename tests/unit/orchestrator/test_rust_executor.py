# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Rust-runner catalog negotiation."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import orjson
import pytest

from aiperf.orchestrator import runner_installation, rust_executor


def _completed(payload: object, *, returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["aiperf-runner", "--capabilities"],
        returncode=returncode,
        stdout=orjson.dumps(payload) + b"\n",
        stderr=b"runner diagnostic" if returncode else b"",
    )


def _catalog(*endpoint_types: str) -> dict[str, object]:
    endpoints = endpoint_types or ("chat",)
    return {
        "schema_version": "1.0",
        "endpoint": {endpoint: {"description": endpoint} for endpoint in endpoints},
        "transport": {
            "http": {"metadata": {"transport_type": "http"}},
            "grpc": {"metadata": {"transport_type": "grpc"}},
        },
        "custom_dataset_loader": {
            "single_turn": {},
            "dag_jsonl": {},
        },
        "public_dataset_loader": {
            "sharegpt": {},
        },
        "synthetic": {
            "synthetic": {},
        },
    }


def test_capabilities_accept_plugins_catalog(monkeypatch) -> None:
    response = _catalog("chat")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: _completed(response))

    assert runner_installation._load_capabilities(Path("runner")) == response


def test_capability_child_receives_only_the_selected_provider_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = _catalog("chat")
    root = tmp_path / "provider-root"
    root.mkdir()
    observed: dict[str, str] = {}

    def run(*_args, **kwargs) -> subprocess.CompletedProcess:
        observed.update(kwargs["env"])
        return _completed(response)

    monkeypatch.setenv("AIPERF_EVALUATOR_PROVIDER_ROOTS", "/attacker/ambient")
    monkeypatch.setattr(subprocess, "run", run)

    assert runner_installation._load_capabilities(Path("runner"), (root,)) == response
    assert observed["AIPERF_EVALUATOR_PROVIDER_ROOTS"] == str(root.resolve())


def test_resolve_accepts_independent_explicit_provider_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "aiperf-runner"
    binary.write_bytes(b"runner")
    nemo_root = tmp_path / "nemo"
    openbench_root = tmp_path / "openbench"
    nemo_root.mkdir()
    openbench_root.mkdir()
    selected: list[tuple[Path, ...]] = []

    monkeypatch.setenv("AIPERF_EVALUATOR_PROVIDER_ROOTS", "/attacker/ambient")
    monkeypatch.setattr(
        runner_installation,
        "_resolve_runner_binary",
        lambda _binary: binary,
    )
    monkeypatch.setattr(
        runner_installation,
        "_load_capabilities",
        lambda _binary, roots: selected.append(roots) or _catalog("chat"),
    )

    installation = runner_installation.RunnerInstallation.resolve(
        binary,
        provider_roots=(nemo_root, openbench_root),
    )

    expected = (nemo_root.resolve(), openbench_root.resolve())
    assert installation.provider_roots == expected
    assert selected == [expected]


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (
            {"schema_version": "1.0", "endpoint": {}, "transport": {"http": {}}},
            "endpoint must contain at least one",
        ),
        (
            {"event": "something_else"},
            "schema_version",
        ),
    ],
)
def test_capabilities_reject_incompatible_runner(
    monkeypatch, payload: dict[str, object], match: str
) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: _completed(payload))

    with pytest.raises((RuntimeError, ValueError), match=match):
        runner_installation._load_capabilities(Path("runner"))


def test_capabilities_surface_process_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=["runner", "--capabilities"],
            returncode=2,
            stdout=b"",
            stderr=b"x-api-key: runner-secret\nrunner diagnostic",
        ),
    )

    with pytest.raises(RuntimeError, match="exit 2") as raised:
        runner_installation._load_capabilities(Path("runner"))
    assert "runner diagnostic" in str(raised.value)
    assert "runner-secret" not in str(raised.value)
    assert "<redacted>" in str(raised.value)


def test_distribution_identity_algorithm_is_pinned(tmp_path: Path) -> None:
    binary = tmp_path / "aiperf-runner"
    binary.write_bytes(b"runner-image\0with-binary-bytes")

    assert runner_installation._runner_distribution_id(binary) == (
        "blake3:3cef527b3dd7185b4ab8590b425b730bd84d695f5c9e1a97302780b7056bf2e9"
    )


def test_protocol_v2_validation_response_is_bound_to_exit() -> None:
    payload = {
        "protocol_version": 2,
        "event": "run_validation",
        "benchmark_id": "validate-me",
        "success": True,
        "completeness": "static",
        "deferred_checks": [
            {
                "code": "dataset_load",
                "path": "run.cfg.datasets",
                "reason": "requires execution preparation",
            }
        ],
    }

    response = runner_installation._parse_validation_response(
        orjson.dumps(payload) + b"\n",
        benchmark_id="validate-me",
        returncode=0,
    )

    assert response["success"] is True
    with pytest.raises(ValueError, match="exit code"):
        runner_installation._parse_validation_response(
            orjson.dumps(payload) + b"\n",
            benchmark_id="validate-me",
            returncode=1,
        )


def test_protocol_v2_validation_failure_requires_typed_errors() -> None:
    payload = {
        "protocol_version": 2,
        "event": "run_validation",
        "benchmark_id": "validate-me",
        "success": False,
        "completeness": "static",
        "errors": [],
    }

    with pytest.raises(ValueError, match="must contain errors"):
        runner_installation._parse_validation_response(
            orjson.dumps(payload) + b"\n",
            benchmark_id="validate-me",
            returncode=1,
        )


def test_installation_verify_distribution_identity_is_noop(tmp_path: Path) -> None:
    binary = tmp_path / "aiperf-runner"
    binary.write_bytes(b"negotiated-runner-image")
    installation = runner_installation.RunnerInstallation(
        binary=binary,
        capabilities=_catalog("chat"),
    )
    binary.write_bytes(b"replacement-runner-image")
    installation.verify_distribution_identity()


def test_runner_installation_accepts_runner_owned_endpoint_ids() -> None:
    installation = runner_installation.RunnerInstallation(
        binary=Path("/opt/aiperf-runner"),
        capabilities=_catalog("chat", "messages", "acme_chat"),
    )

    installation.preflight_endpoint("messages")
    installation.preflight_endpoint("acme_chat")


def test_runner_installation_rejects_unavailable_endpoint_clearly() -> None:
    installation = runner_installation.RunnerInstallation(
        binary=Path("/opt/aiperf-runner"),
        capabilities=_catalog("chat", "messages"),
    )

    with pytest.raises(RuntimeError, match="not compiled.*chat, messages") as raised:
        installation.preflight_endpoint("acme_chat")

    message = str(raised.value)
    assert "AIPERF_RUNNER_BIN" in message
    assert "plugin" not in message.lower()


def test_runner_installation_preflights_every_fixed_plan_endpoint() -> None:
    installation = runner_installation.RunnerInstallation(
        binary=Path("/opt/aiperf-runner"),
        capabilities=_catalog("chat", "messages"),
    )
    plan = SimpleNamespace(
        configs=[
            SimpleNamespace(
                endpoint=SimpleNamespace(type="chat"), endpoint_profiles={}
            ),
            SimpleNamespace(
                endpoint=SimpleNamespace(type="future_compiled_endpoint"),
                endpoint_profiles={},
            ),
        ]
    )

    with pytest.raises(RuntimeError, match="future_compiled_endpoint"):
        installation.preflight_plan(plan)


def test_executor_rejects_unavailable_endpoint_without_reading_config_secrets() -> None:
    installation = runner_installation.RunnerInstallation(
        binary=Path("/opt/aiperf-runner"),
        capabilities=_catalog("chat", "messages"),
    )
    request = {
        "protocol_version": 2,
        "operation": "execute",
        "run": {
            "benchmark_id": "custom-endpoint",
            "cfg": {
                "transport": {"type": "http"},
                "endpoint": {"type": "acme_chat", "api_key": "sk-never-log-me"},
                "endpoint_profiles": {},
                "datasets": [],
            },
        },
    }

    with pytest.raises(RuntimeError, match="acme_chat") as raised:
        installation.preflight_request(request)

    assert "sk-never-log-me" not in str(raised.value)


def test_missing_terminal_surfaces_exit_and_redacted_stderr() -> None:
    with pytest.raises(ValueError) as raised:
        rust_executor._parse_terminal(
            b"",
            SimpleNamespace(benchmark_id="run"),
            protocol_version=2,
            returncode=-6,
            stderr=b"Authorization: Bearer runner-secret\nstack overflow",
        )

    message = str(raised.value)
    assert "child exit code -6" in message
    assert "stack overflow" in message
    assert "runner-secret" not in message
    assert "<redacted>" in message


def test_terminal_failure_redacts_runner_error_and_stderr(tmp_path: Path) -> None:
    completed = subprocess.CompletedProcess(
        args=["runner"],
        returncode=1,
        stdout=b"",
        stderr=b"x-api-key: stderr-secret",
    )

    result = rust_executor._failure(
        completed,
        {"error": "Authorization: Bearer terminal-secret"},
        SimpleNamespace(label="failed", artifact_dir=tmp_path),
    )

    assert result.success is False
    assert "terminal-secret" not in result.error
    assert "stderr-secret" not in result.error
    assert result.error.count("<redacted>") == 2


def test_terminal_does_not_require_distribution_id() -> None:
    terminal = rust_executor._parse_terminal(
        orjson.dumps(
            {
                "protocol_version": 2,
                "event": "run_terminal",
                "benchmark_id": "run",
                "success": True,
                "report_path": "/tmp/native-v2.json",
            }
        )
        + b"\n",
        SimpleNamespace(benchmark_id="run"),
        protocol_version=2,
        returncode=0,
    )
    assert terminal["success"] is True


def test_v2_request_capabilities_read_benchmark_run_cfg() -> None:
    capabilities = _catalog("kserve_v2_infer")
    request = {
        "protocol_version": 2,
        "run": {
            "cfg": {
                "transport": {"type": "grpc"},
                "endpoint": {"type": "kserve_v2_infer"},
                "endpoint_profiles": {},
                "datasets": [{"type": "synthetic", "entries": 1}],
            }
        },
    }

    runner_installation._require_v2_request_capabilities(capabilities, request)


def test_v2_request_capabilities_check_optional_dataset_categories() -> None:
    capabilities = _catalog("chat")
    request = {
        "protocol_version": 2,
        "run": {
            "cfg": {
                "transport": {"type": "http"},
                "endpoint": {"type": "chat"},
                "datasets": [
                    {"type": "file", "format": "unknown_format", "path": "x.jsonl"}
                ],
            }
        },
    }

    with pytest.raises(RuntimeError, match="custom_dataset_loader.unknown_format"):
        runner_installation._require_v2_request_capabilities(capabilities, request)
