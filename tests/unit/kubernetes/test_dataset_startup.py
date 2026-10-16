# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dataset-manager startup: rehydration, preflight, and the termination message."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import orjson
import pytest
from pytest import param

from aiperf.common.endpoint_credentials import AIPERF_INJECTED_DATASET_URLS
from aiperf.common.redact import REDACTED_VALUE, redact_url
from aiperf.config import BenchmarkConfig
from aiperf.config.loader.errors import ConfigurationError
from aiperf.config.resolution.plan import BenchmarkRun
from aiperf.dataset.remote.errors import (
    RemoteErrorKind,
    RemoteSourceError,
    RemoteStage,
)
from aiperf.kubernetes import dataset_startup
from aiperf.kubernetes.constants import Containers
from aiperf.kubernetes.dataset_startup import (
    MAX_TERMINATION_MESSAGE_BYTES,
    POD_PREFIX,
    prepare_dataset_manager_run,
    termination_message,
    write_termination_message,
)
from aiperf.kubernetes.environment import K8sEnvironment
from aiperf.kubernetes.jobset import AIPerfJobSetSpec

RAW = "https://bucket.s3.amazonaws.com/t.jsonl?X-Amz-Signature=topsecret"
REDACTED = RAW.replace("topsecret", REDACTED_VALUE)


def _run(artifact_dir: Path, path: object) -> BenchmarkRun:
    cfg = BenchmarkConfig.model_validate(
        {
            "models": ["m"],
            "endpoint": {"urls": ["http://localhost:8000"], "type": "chat"},
            "datasets": [{"name": "main", "type": "file", "path": path, "format": "mooncake_trace"}],
            "phases": [{"name": "profiling", "type": "concurrency", "requests": 4, "concurrency": 1}],
        }
    )  # fmt: skip
    return BenchmarkRun(benchmark_id="job", cfg=cfg, artifact_dir=artifact_dir)


@pytest.fixture
def termination_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    log = tmp_path / "termination-log"
    monkeypatch.setattr(dataset_startup, "TERMINATION_LOG_PATH", log)
    return log


def test_termination_message_is_one_prefixed_redacted_line() -> None:
    error = RemoteSourceError(
        ref=RAW,
        kind=RemoteErrorKind.AUTH_DENIED,
        stage=RemoteStage.POD,
        what="the server denied access (HTTP 403)\nafter a redirect",
        hint="Generate a new presigned URL",
    )
    line = termination_message(error, secrets=[])
    assert POD_PREFIX == "[dataset-manager pod]"
    assert line.startswith(f"[dataset-manager pod] Cannot use {REDACTED}: ")
    assert "\n" not in line
    assert "topsecret" not in line


def test_termination_message_scrubs_known_secret_values() -> None:
    line = termination_message(
        RuntimeError("download failed for s3cr3t-v4lue"), secrets=["s3cr3t-v4lue"]
    )
    assert (
        line
        == f"[dataset-manager pod] RuntimeError: download failed for {REDACTED_VALUE}"
    )


@pytest.mark.parametrize(
    "secret,text,leaks",
    [
        param(
            "https://h/p?X-Amz-Signature=sig&weird=SECRET1",
            "GET https://h/p?X-Amz-Signature=sig&weird=SECRET1 end",
            ["SECRET1", "h/p"],
            id="unlisted-query-parameter",
        ),
        param("user:p@ss", "failed for user:p@ss now", ["@ss", "user:p"], id="userinfo"),
        param(
            "https://u:p@ss@host/x",
            "bad https://u:p@ss@host/x end",
            ["ss@", "@ss", "host"],
            id="url-with-at-in-password",
        ),
    ],
)  # fmt: skip
def test_termination_message_exact_secret_wins_over_token_redaction(
    secret: str, text: str, leaks: list[str]
) -> None:
    """A held secret is replaced whole before any pattern rewrites its pieces."""
    line = termination_message(RuntimeError(text), secrets=[secret])
    for leak in leaks:
        assert leak not in line
    assert line.endswith(f"{REDACTED_VALUE} {text.rsplit(' ', 1)[1]}")


def test_termination_message_keeps_text_after_a_raw_presigned_url() -> None:
    """Whole-text URL redaction would swallow everything after the signature."""
    line = termination_message(
        RuntimeError(f"GET {RAW} returned HTTP 403. Check the clock"), secrets=[]
    )
    assert line == (
        f"[dataset-manager pod] RuntimeError: GET {REDACTED} returned HTTP 403. "
        "Check the clock"
    )


def test_termination_message_multibyte_truncation_stays_valid_utf8() -> None:
    # The one-byte "x" shifts the 2-byte "é" run so byte 4096 lands inside one.
    # The raw URL comes first so redaction is checked on text that survives the cut.
    message = f"{RAW}\nx" + "é\n" * 3000
    untruncated = f"{POD_PREFIX} {REDACTED} x" + " ".join(["é"] * 3000)
    naive_cut = untruncated.encode("utf-8")[:MAX_TERMINATION_MESSAGE_BYTES]
    with pytest.raises(UnicodeDecodeError):
        naive_cut.decode("utf-8")

    line = termination_message(ConfigurationError(message), secrets=[])

    encoded = line.encode("utf-8")
    assert len(encoded) <= MAX_TERMINATION_MESSAGE_BYTES
    assert encoded.decode("utf-8") == line
    assert line.startswith(f"{POD_PREFIX} {REDACTED} xé é")
    assert untruncated.startswith(line)
    assert "\n" not in line
    assert "topsecret" not in line


def test_write_termination_message_missing_directory_is_ignored(tmp_path: Path) -> None:
    write_termination_message(
        ValueError("no such dataset here"), [], path=tmp_path / "missing" / "log"
    )
    assert not (tmp_path / "missing").exists()


def test_prepare_dataset_manager_run_rehydrates_before_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, termination_log: Path
) -> None:
    monkeypatch.setenv(AIPERF_INJECTED_DATASET_URLS, orjson.dumps([RAW]).decode())
    run = _run(tmp_path, {"ref": REDACTED, "resolved_version": "etag:abc"})
    seen: list[str] = []
    with patch(
        "aiperf.cli_runner._preflight.preflight_datasets_for_run",
        side_effect=lambda r: seen.append(r.cfg.datasets[0].path.ref),
    ):
        prepare_dataset_manager_run(run, endpoint_secrets=[])
    assert seen == [RAW]
    assert not termination_log.exists()


def test_prepare_dataset_manager_run_preflight_failure_writes_termination_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, termination_log: Path
) -> None:
    monkeypatch.setenv(AIPERF_INJECTED_DATASET_URLS, orjson.dumps([RAW]).decode())
    run = _run(tmp_path, REDACTED)
    error = RemoteSourceError(
        ref=RAW,
        kind=RemoteErrorKind.AUTH_DENIED,
        stage=RemoteStage.MATERIALIZE,
        what=f"GET {RAW} returned HTTP 403",
        hint="Generate a new presigned URL",
    )
    with (
        patch(
            "aiperf.cli_runner._preflight.preflight_datasets_for_run", side_effect=error
        ),
        pytest.raises(RemoteSourceError),
    ):
        prepare_dataset_manager_run(run, endpoint_secrets=[])
    message = termination_log.read_text(encoding="utf-8")
    assert message.startswith("[dataset-manager pod] Cannot use ")
    assert "topsecret" not in message


def test_prepare_dataset_manager_run_scrubs_bare_signature_from_injected_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, termination_log: Path
) -> None:
    """S3 echoes the signature alone (``<SignatureProvided>``), out of URL context."""
    monkeypatch.setenv(AIPERF_INJECTED_DATASET_URLS, orjson.dumps([RAW]).decode())
    run = _run(tmp_path, REDACTED)
    error = RuntimeError("<SignatureProvided>topsecret</SignatureProvided>")
    with (
        patch(
            "aiperf.cli_runner._preflight.preflight_datasets_for_run", side_effect=error
        ),
        pytest.raises(RuntimeError),
    ):
        prepare_dataset_manager_run(run, endpoint_secrets=[])
    message = termination_log.read_text(encoding="utf-8")
    assert "topsecret" not in message
    assert REDACTED_VALUE in message


def test_prepare_dataset_manager_run_scrubs_bare_password_from_injected_userinfo_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, termination_log: Path
) -> None:
    raw = "https://user:topsecret@h.example/t.jsonl"
    monkeypatch.setenv(AIPERF_INJECTED_DATASET_URLS, orjson.dumps([raw]).decode())
    run = _run(tmp_path, redact_url(raw))
    error = RuntimeError("login rejected for password topsecret")
    with (
        patch(
            "aiperf.cli_runner._preflight.preflight_datasets_for_run", side_effect=error
        ),
        pytest.raises(RuntimeError),
    ):
        prepare_dataset_manager_run(run, endpoint_secrets=[])
    message = termination_log.read_text(encoding="utf-8")
    assert "topsecret" not in message
    assert REDACTED_VALUE in message


def test_prepare_dataset_manager_run_scrubs_endpoint_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, termination_log: Path
) -> None:
    monkeypatch.delenv(AIPERF_INJECTED_DATASET_URLS, raising=False)
    run = _run(tmp_path, "hf://org/repo/t.jsonl")
    error = RuntimeError("upstream rejected key sk-endpoint-key")
    with (
        patch(
            "aiperf.cli_runner._preflight.preflight_datasets_for_run", side_effect=error
        ),
        pytest.raises(RuntimeError),
    ):
        prepare_dataset_manager_run(run, endpoint_secrets=["sk-endpoint-key"])
    assert "sk-endpoint-key" not in termination_log.read_text(encoding="utf-8")


def test_prepare_dataset_manager_run_missing_injection_writes_termination_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, termination_log: Path
) -> None:
    monkeypatch.delenv(AIPERF_INJECTED_DATASET_URLS, raising=False)
    run = _run(tmp_path, REDACTED)
    with pytest.raises(ValueError, match="has no URL for it"):
        prepare_dataset_manager_run(run, endpoint_secrets=[])
    assert "has no URL for it" in termination_log.read_text(encoding="utf-8")


def test_dataset_manager_container_falls_back_to_logs_for_termination_message() -> None:
    spec = AIPerfJobSetSpec(name="ts", namespace="ns", job_id="ts-1", image="aiperf:ci")
    manifest = spec.to_k8s_manifest()
    job = next(
        j for j in manifest["spec"]["replicatedJobs"] if j["name"] == "controller"
    )
    containers = {
        c["name"]: c for c in job["template"]["spec"]["template"]["spec"]["containers"]
    }
    assert (
        containers[Containers.DATASET_MANAGER]["terminationMessagePolicy"]
        == "FallbackToLogsOnError"
    )
    assert "terminationMessagePolicy" not in containers[Containers.CONTROL_PLANE]


def _controller_containers(**spec_fields: object) -> dict[str, dict]:
    spec = AIPerfJobSetSpec(
        name="ts", namespace="ns", job_id="ts-1", image="aiperf:ci", **spec_fields
    )
    job = next(
        j
        for j in spec.to_k8s_manifest()["spec"]["replicatedJobs"]
        if j["name"] == "controller"
    )
    return {
        c["name"]: c for c in job["template"]["spec"]["template"]["spec"]["containers"]
    }


@pytest.mark.parametrize(
    "budget,period,expected",
    [
        param(3600, 5, 720, id="default-budget"),
        param(601, 5, 121, id="rounds-up"),
        param(60, 5, 30, id="never-below-the-normal-threshold"),
    ],
)  # fmt: skip
def test_dataset_manager_startup_probe_remote_dataset_covers_download_budget(
    monkeypatch: pytest.MonkeyPatch, budget: int, period: int, expected: int
) -> None:
    health = K8sEnvironment.HEALTH
    monkeypatch.setattr(health, "DATASET_DOWNLOAD_STARTUP_BUDGET_SECONDS", budget)
    monkeypatch.setattr(health, "STARTUP_PERIOD_SECONDS", period)
    monkeypatch.setattr(health, "STARTUP_FAILURE_THRESHOLD", 30)
    containers = _controller_containers(remote_dataset=True)
    probe = containers[Containers.DATASET_MANAGER]["startupProbe"]
    assert probe["failureThreshold"] == expected
    assert probe["periodSeconds"] == period
    for name, container in containers.items():
        if name != Containers.DATASET_MANAGER and "startupProbe" in container:
            assert container["startupProbe"]["failureThreshold"] == 30, name


def test_dataset_manager_startup_probe_local_dataset_keeps_default_threshold() -> None:
    containers = _controller_containers()
    assert (
        containers[Containers.DATASET_MANAGER]["startupProbe"]["failureThreshold"]
        == K8sEnvironment.HEALTH.STARTUP_FAILURE_THRESHOLD
    )


@pytest.mark.parametrize(
    "path,expected",
    [
        param("hf://org/repo/t.jsonl", True, id="remote"),
        param("t.jsonl", False, id="local"),
    ],
)  # fmt: skip
def test_kubernetes_deployment_jobset_spec_flags_remote_dataset(
    path: str, expected: bool
) -> None:
    from aiperf.config import AIPerfConfig
    from aiperf.config.deployment import DeploymentConfig
    from aiperf.kubernetes.resources import KubernetesDeployment

    benchmark = _run(Path("/tmp"), path).cfg.model_dump(mode="json", exclude_unset=True)
    deployment = KubernetesDeployment(
        job_id="job",
        namespace="ns",
        config=AIPerfConfig(benchmark=benchmark),
        deployment=DeploymentConfig(),
    )
    assert deployment.get_jobset_spec().remote_dataset is expected


def test_prepare_dataset_manager_run_redaction_failure_keeps_the_original_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, termination_log: Path
) -> None:
    monkeypatch.delenv(AIPERF_INJECTED_DATASET_URLS, raising=False)
    run = _run(tmp_path, "hf://org/repo/t.jsonl")
    original = RuntimeError("download failed")
    with (
        patch(
            "aiperf.cli_runner._preflight.preflight_datasets_for_run",
            side_effect=original,
        ),
        patch.object(
            dataset_startup,
            "termination_message",
            side_effect=ValueError("redactor broke"),
        ),
        pytest.raises(RuntimeError) as exc_info,
    ):
        prepare_dataset_manager_run(run, endpoint_secrets=[])
    assert exc_info.value is original
    message = termination_log.read_text(encoding="utf-8")
    assert message.startswith(f"{POD_PREFIX} RuntimeError")
    assert "download failed" not in message
