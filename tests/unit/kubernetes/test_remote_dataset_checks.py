# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pod-credential checks for remote datasets (shared by the kube CLI and the operator)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from pytest import param

from aiperf.config import BenchmarkConfig
from aiperf.config.dataset import FileDataset
from aiperf.config.dataset.remote_ref import RemoteRef
from aiperf.config.deployment import PodTemplateConfig
from aiperf.config.loader import ConfigurationError
from aiperf.kubernetes.preflight import CheckStatus
from aiperf.kubernetes.remote_datasets import (
    check_remote_datasets,
    evaluate_remote_dataset,
    presigned_expiry,
    report_remote_dataset_checks,
)

NOW = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)
HF_SECRET_ENV = {
    "name": "HF_TOKEN",
    "valueFrom": {"secretKeyRef": {"name": "hf", "key": "token"}},
}
AWS_SECRET_ENV = {
    "name": "AWS_ACCESS_KEY_ID",
    "valueFrom": {"secretKeyRef": {"name": "aws", "key": "id"}},
}


def _amz(signed: datetime, seconds: int) -> str:
    return (
        "https://bucket.s3.amazonaws.com/t.jsonl"
        f"?X-Amz-Date={signed:%Y%m%dT%H%M%SZ}&X-Amz-Expires={seconds}"
        "&X-Amz-Signature=<redacted>"
    )


def _benchmark(path: str) -> BenchmarkConfig:
    return BenchmarkConfig.model_validate(
        {
            "models": ["m"],
            "endpoint": {"urls": ["http://localhost:8000"], "type": "chat"},
            "datasets": [{"name": "main", "type": "file", "path": path, "format": "mooncake_trace"}],
            "phases": [{"name": "profiling", "type": "concurrency", "requests": 4, "concurrency": 1}],
        }
    )  # fmt: skip


@pytest.mark.parametrize(
    "url,expected",
    [
        param(_amz(NOW, 3600), NOW + timedelta(hours=1), id="amz"),
        param("https://storage.googleapis.com/b/t.jsonl?X-Goog-Date=20261015T120000Z&X-Goog-Expires=60&X-Goog-Signature=<redacted>", NOW + timedelta(seconds=60), id="goog"),
        param("https://a.blob.core.windows.net/c/t.jsonl?sv=2022-11-02&se=2026-10-15T13:00:00Z&sig=<redacted>", NOW + timedelta(hours=1), id="azure"),
        param("https://a.blob.core.windows.net/c/t.jsonl?se=2026-10-16&sig=<redacted>", datetime(2026, 10, 16, tzinfo=UTC), id="azure-date-only"),
        param("https://h.example/t.jsonl", None, id="plain"),
    ],
)  # fmt: skip
def test_presigned_expiry_reads_each_provider(
    url: str, expected: datetime | None
) -> None:
    assert presigned_expiry(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        param("https://h/t.jsonl?X-Amz-Date=yesterday&X-Amz-Expires=60&X-Amz-Signature=x", id="bad-date"),
        param("https://h/t.jsonl?X-Amz-Date=20261015T120000Z&X-Amz-Expires=soon&X-Amz-Signature=x", id="bad-lifetime"),
        param("https://h/t.jsonl?X-Amz-Date=20261015T120000Z&X-Amz-Expires=99999999999999999999&X-Amz-Signature=x", id="overflow"),
        param("https://h/t.jsonl?se=not-a-date&sig=x", id="bad-sas"),
    ],
)  # fmt: skip
def test_presigned_expiry_malformed_values_return_none(url: str) -> None:
    assert presigned_expiry(url) is None


@pytest.mark.parametrize(
    "needs_auth,env,expected",
    [
        param(True, [], CheckStatus.FAIL, id="private-no-token"),
        param(True, [HF_SECRET_ENV], CheckStatus.PASS, id="private-secret-token"),
        param(True, [{"name": "HF_TOKEN_PATH", "value": "/secrets/hf/token"}], CheckStatus.PASS, id="private-token-path"),
        param(True, [{"name": "HF_TOKEN", "value": "hf_literal"}], CheckStatus.FAIL, id="literal-token-is-not-secret-backed"),
        param(False, [], CheckStatus.PASS, id="public"),
        param(None, [], CheckStatus.WARN, id="unknown"),
    ],
)  # fmt: skip
def test_evaluate_remote_dataset_hf_matrix(
    needs_auth: bool | None, env: list[dict], expected: CheckStatus
) -> None:
    result = evaluate_remote_dataset(
        RemoteRef(ref="hf://org/repo"), needs_auth, PodTemplateConfig(env=env), now=NOW
    )
    assert result.status is expected
    if expected is CheckStatus.FAIL:
        assert "--env-from-secrets HF_TOKEN=<secret>/<key>" in result.hints[0]


@pytest.mark.parametrize(
    "needs_auth,env,service_account,expected",
    [
        param(True, [], None, CheckStatus.WARN, id="private-unset-sa"),
        param(True, [], "default", CheckStatus.WARN, id="private-default-sa"),
        param(None, [], None, CheckStatus.WARN, id="unknown"),
        param(True, [], "s3-reader", CheckStatus.PASS, id="dedicated-sa"),
        param(True, [AWS_SECRET_ENV], None, CheckStatus.PASS, id="aws-env"),
        param(False, [], None, CheckStatus.PASS, id="public"),
    ],
)  # fmt: skip
def test_evaluate_remote_dataset_s3_matrix(
    needs_auth: bool | None,
    env: list[dict],
    service_account: str | None,
    expected: CheckStatus,
) -> None:
    pod_template = PodTemplateConfig(env=env, service_account_name=service_account)
    result = evaluate_remote_dataset(
        RemoteRef(ref="s3://bucket/t.jsonl"), needs_auth, pod_template, now=NOW
    )
    assert result.status is expected


@pytest.mark.parametrize(
    "url,expected",
    [
        param("https://h.example/t.jsonl", CheckStatus.PASS, id="plain"),
        param(_amz(NOW - timedelta(hours=2), 3600), CheckStatus.FAIL, id="expired"),
        param(_amz(NOW - timedelta(minutes=30), 3600), CheckStatus.WARN, id="under-an-hour"),
        param(_amz(NOW, 7200), CheckStatus.PASS, id="two-hours"),
        param("https://h.example/t.jsonl?sig=<redacted>", CheckStatus.PASS, id="no-stated-expiry"),
    ],
)  # fmt: skip
def test_evaluate_remote_dataset_presigned_expiry(
    url: str, expected: CheckStatus
) -> None:
    result = evaluate_remote_dataset(
        RemoteRef(ref=url), True, PodTemplateConfig(), now=NOW
    )
    assert result.status is expected


@pytest.mark.asyncio
async def test_check_remote_datasets_no_remote_reference_skips() -> None:
    datasets = [FileDataset(name="main", type="file", path="t.jsonl")]
    result = await check_remote_datasets(datasets, PodTemplateConfig())
    assert result.status is CheckStatus.SKIP
    assert result.name == "Remote Datasets"


@pytest.mark.asyncio
async def test_check_remote_datasets_reports_worst_status_and_all_hints() -> None:
    datasets = [
        FileDataset(name="a", type="file", path="hf://org/private"),
        FileDataset(name="b", type="file", path="s3://bucket/t.jsonl"),
    ]
    with patch(
        "aiperf.dataset.remote.anonymous.anonymous_needs_auth",
        AsyncMock(return_value=True),
    ):
        result = await check_remote_datasets(datasets, PodTemplateConfig(), now=NOW)
    assert result.status is CheckStatus.FAIL
    assert "hf://org/private" in result.message
    assert "s3://bucket/t.jsonl" in result.message
    assert len(result.hints) == 2


def test_report_remote_dataset_checks_failure_raises_with_hint() -> None:
    with (
        patch(
            "aiperf.dataset.remote.anonymous.anonymous_needs_auth",
            AsyncMock(return_value=True),
        ),
        pytest.raises(ConfigurationError, match="HF_TOKEN=<secret>/<key>"),
    ):
        report_remote_dataset_checks(_benchmark("hf://org/private"), {"image": "x"})


def test_report_remote_dataset_checks_warning_prints_and_continues() -> None:
    with (
        patch(
            "aiperf.dataset.remote.anonymous.anonymous_needs_auth",
            AsyncMock(return_value=True),
        ),
        patch("aiperf.kubernetes.console.print_warning") as warn,
    ):
        report_remote_dataset_checks(_benchmark("s3://bucket/t.jsonl"), {})
    assert "not publicly readable" in warn.call_args.args[0]


def test_report_remote_dataset_checks_uses_spec_pod_template() -> None:
    spec = {"podTemplate": {"env": [HF_SECRET_ENV]}}
    with (
        patch(
            "aiperf.dataset.remote.anonymous.anonymous_needs_auth",
            AsyncMock(return_value=True),
        ),
        patch("aiperf.kubernetes.console.print_warning") as warn,
    ):
        report_remote_dataset_checks(_benchmark("hf://org/private"), spec)
    warn.assert_not_called()


def test_report_remote_dataset_checks_local_dataset_makes_no_probe() -> None:
    probe = AsyncMock()
    with patch("aiperf.dataset.remote.anonymous.anonymous_needs_auth", probe):
        report_remote_dataset_checks(_benchmark("t.jsonl"), {})
    probe.assert_not_awaited()
