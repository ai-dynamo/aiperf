# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Presigned dataset URLs: Secret transport validation and in-pod rehydration."""

from __future__ import annotations

import os
from pathlib import Path

import orjson
import pytest
from pytest import param

from aiperf.common.endpoint_credentials import (
    AIPERF_INJECTED_DATASET_URLS,
    apply_injected_dataset_urls,
    consume_injected_dataset_urls,
    validate_kubernetes_credential_transport,
)
from aiperf.common.redact import REDACTED_VALUE
from aiperf.config import BenchmarkConfig
from aiperf.config.dataset import FileDataset
from aiperf.config.dataset.remote_ref import carries_credential, remote_dataset_refs
from aiperf.config.endpoint import EndpointConfig
from aiperf.config.resolution.plan import BenchmarkRun

PLAIN = "https://bucket.s3.amazonaws.com/t.jsonl"
RAW = (
    f"{PLAIN}?X-Amz-Date=20261015T000000Z&X-Amz-Expires=3600&X-Amz-Signature=topsecret"
)
REDACTED = RAW.replace("topsecret", REDACTED_VALUE)
SECRET_ENV = [
    {
        "name": AIPERF_INJECTED_DATASET_URLS,
        "valueFrom": {"secretKeyRef": {"name": "dataset-urls", "key": "urls"}},
    }
]


def _dataset(path: object) -> FileDataset:
    return FileDataset(name="main", type="file", path=path)


def _endpoint() -> EndpointConfig:
    return EndpointConfig(urls=["http://localhost:8000"])


def _run(path: object, artifact_dir: Path) -> BenchmarkRun:
    cfg = BenchmarkConfig.model_validate(
        {
            "models": ["m"],
            "endpoint": {"urls": ["http://localhost:8000"], "type": "chat"},
            "datasets": [
                {"name": "main", "type": "file", "path": path, "format": "mooncake_trace"}
            ],
            "phases": [
                {"name": "profiling", "type": "concurrency", "requests": 4, "concurrency": 1}
            ],
        }
    )  # fmt: skip
    return BenchmarkRun(benchmark_id="job", cfg=cfg, artifact_dir=artifact_dir)


@pytest.mark.parametrize(
    "value,expected",
    [
        param(RAW, True, id="raw-presigned"),
        param(REDACTED, True, id="redacted-presigned"),
        param("https://user:pw@h.example/t.jsonl", True, id="userinfo"),
        param(PLAIN, False, id="plain"),
        param("hf://org/repo/t.jsonl", False, id="hf"),
    ],
)  # fmt: skip
def test_carries_credential_detects_raw_and_redacted_credentials(
    value: str, expected: bool
) -> None:
    assert carries_credential(value) is expected


def test_remote_dataset_refs_skips_local_and_synthetic_datasets() -> None:
    datasets = [_dataset("hf://org/repo/t.jsonl"), _dataset("t.jsonl"), object()]
    assert [ref.ref for ref in remote_dataset_refs(datasets)] == [
        "hf://org/repo/t.jsonl"
    ]


@pytest.mark.parametrize(
    "path",
    [
        param(RAW, id="raw"),
        param(REDACTED, id="redacted"),
        param("https://user:pw@h.example/t.jsonl", id="userinfo"),
    ],
)  # fmt: skip
def test_validate_kubernetes_credential_transport_credentialed_dataset_requires_injected_urls(
    path: str,
) -> None:
    with pytest.raises(ValueError, match=AIPERF_INJECTED_DATASET_URLS) as exc_info:
        validate_kubernetes_credential_transport(
            _endpoint(), [], datasets=[_dataset(path)]
        )
    message = str(exc_info.value)
    assert "Secret-backed" in message
    assert "kubectl create secret generic" in message
    assert f"--env-from-secrets {AIPERF_INJECTED_DATASET_URLS}=" in message
    assert "topsecret" not in message
    assert "pw@" not in message


def test_validate_kubernetes_credential_transport_credentialed_dataset_with_secret_env_passes() -> (
    None
):
    validate_kubernetes_credential_transport(
        _endpoint(), SECRET_ENV, datasets=[_dataset(REDACTED)]
    )


def test_validate_kubernetes_credential_transport_literal_env_value_is_rejected() -> (
    None
):
    literal = [
        {"name": AIPERF_INJECTED_DATASET_URLS, "value": orjson.dumps([RAW]).decode()}
    ]
    with pytest.raises(ValueError, match=AIPERF_INJECTED_DATASET_URLS):
        validate_kubernetes_credential_transport(
            _endpoint(), literal, datasets=[_dataset(RAW)]
        )


@pytest.mark.parametrize(
    "path",
    [
        param("hf://org/repo/t.jsonl", id="hf"),
        param("s3://bucket/t.jsonl", id="s3"),
        param(PLAIN, id="plain-https"),
        param("t.jsonl", id="local"),
    ],
)  # fmt: skip
def test_validate_kubernetes_credential_transport_public_dataset_needs_no_secret(
    path: str,
) -> None:
    validate_kubernetes_credential_transport(_endpoint(), [], datasets=[_dataset(path)])


def test_consume_injected_dataset_urls_pops_and_decodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(AIPERF_INJECTED_DATASET_URLS, orjson.dumps([RAW]).decode())
    assert consume_injected_dataset_urls() == [RAW]
    assert os.getenv(AIPERF_INJECTED_DATASET_URLS) is None


def test_consume_injected_dataset_urls_unset_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(AIPERF_INJECTED_DATASET_URLS, raising=False)
    assert consume_injected_dataset_urls() is None


def test_consume_injected_dataset_urls_invalid_json_names_the_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(AIPERF_INJECTED_DATASET_URLS, "not json")
    with pytest.raises(ValueError, match=AIPERF_INJECTED_DATASET_URLS):
        consume_injected_dataset_urls()


def test_apply_injected_dataset_urls_rehydrates_redacted_ref_and_keeps_version(
    tmp_path: Path,
) -> None:
    run = _run({"ref": REDACTED, "resolved_version": "etag:abc"}, tmp_path)
    apply_injected_dataset_urls(run, [RAW])
    path = run.cfg.datasets[0].path
    assert path.ref == RAW
    assert path.resolved_version == "etag:abc"


def test_apply_injected_dataset_urls_plain_cr_url_matches_presigned_injection(
    tmp_path: Path,
) -> None:
    run = _run(PLAIN, tmp_path)
    apply_injected_dataset_urls(run, [RAW])
    assert run.cfg.datasets[0].path.ref == RAW


def test_apply_injected_dataset_urls_missing_url_raises_without_secret(
    tmp_path: Path,
) -> None:
    run = _run(REDACTED, tmp_path)
    with pytest.raises(ValueError, match="has no URL for it") as exc_info:
        apply_injected_dataset_urls(run, None)
    assert "topsecret" not in str(exc_info.value)


def test_apply_injected_dataset_urls_unmatched_url_raises_without_echoing_it(
    tmp_path: Path,
) -> None:
    run = _run("hf://org/repo/t.jsonl", tmp_path)
    with pytest.raises(ValueError, match="matches no dataset") as exc_info:
        apply_injected_dataset_urls(run, [RAW])
    assert "topsecret" not in str(exc_info.value)


@pytest.mark.parametrize(
    "entry",
    [
        param("bare-token-s3cr3t", id="bare-token"),
        param("ftp://user:s3cr3t@h.example/t.jsonl", id="unsupported-scheme"),
    ],
)  # fmt: skip
def test_apply_injected_dataset_urls_non_reference_entry_is_named_by_index_only(
    entry: str, tmp_path: Path
) -> None:
    run = _run(PLAIN, tmp_path)
    with pytest.raises(
        ValueError, match="entry 1 .*is not a dataset reference"
    ) as exc_info:
        apply_injected_dataset_urls(run, [RAW, entry])
    message = str(exc_info.value)
    assert "s3cr3t" not in message
    assert "h.example" not in message
    assert "topsecret" not in message


def test_apply_injected_dataset_urls_two_signings_of_one_dataset_raise_by_index(
    tmp_path: Path,
) -> None:
    resigned = RAW.replace("topsecret", "newsecret")
    run = _run(PLAIN, tmp_path)
    with pytest.raises(
        ValueError, match="entries 0 and 1 .*different URLs for the same dataset"
    ) as exc_info:
        apply_injected_dataset_urls(run, [RAW, resigned])
    message = str(exc_info.value)
    assert "topsecret" not in message
    assert "newsecret" not in message
    assert "bucket.s3" not in message


def test_apply_injected_dataset_urls_repeated_identical_url_is_accepted(
    tmp_path: Path,
) -> None:
    run = _run(PLAIN, tmp_path)
    apply_injected_dataset_urls(run, [RAW, RAW])
    assert run.cfg.datasets[0].path.ref == RAW


def test_apply_injected_dataset_urls_public_ref_without_injection_is_unchanged(
    tmp_path: Path,
) -> None:
    run = _run("hf://org/repo/t.jsonl", tmp_path)
    apply_injected_dataset_urls(run, None)
    assert run.cfg.datasets[0].path.ref == "hf://org/repo/t.jsonl"
