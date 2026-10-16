# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Anonymous needs-auth probe for remote dataset references."""

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from pytest import param

from aiperf.common.environment import Environment
from aiperf.config.dataset.remote_ref import RemoteRef
from aiperf.dataset.remote import anonymous


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "info,expected",
    [
        param(SimpleNamespace(gated=False, private=False), False, id="public"),
        param(SimpleNamespace(gated="auto", private=False), True, id="gated"),
        param(SimpleNamespace(gated=False, private=True), True, id="private"),
    ],
)  # fmt: skip
async def test_anonymous_needs_auth_hf_reads_metadata_without_token(
    info: SimpleNamespace, expected: bool
) -> None:
    with patch("huggingface_hub.HfApi.dataset_info", return_value=info) as dataset_info:
        assert (
            await anonymous.anonymous_needs_auth(RemoteRef(ref="hf://org/repo"))
            is expected
        )
    assert dataset_info.call_args.args == ("org/repo",)
    assert dataset_info.call_args.kwargs["token"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error,expected",
    [
        param(RepositoryNotFoundError("404", response=Mock()), True, id="missing-or-private"),
        param(GatedRepoError("gated", response=Mock()), True, id="gated-error"),
        param(HfHubHTTPError("401", response=Mock(status_code=401)), True, id="http-401"),
        param(RevisionNotFoundError("rev", response=Mock()), None, id="bad-revision"),
        param(HfHubHTTPError("503", response=Mock(status_code=503)), None, id="http-503"),
        param(ConnectionError("reset"), None, id="network"),
    ],
)  # fmt: skip
async def test_anonymous_needs_auth_hf_errors_map_to_needs_auth_or_unknown(
    error: Exception, expected: bool | None
) -> None:
    with patch("huggingface_hub.HfApi.dataset_info", side_effect=error):
        assert (
            await anonymous.anonymous_needs_auth(RemoteRef(ref="hf://org/repo"))
            is expected
        )


@pytest.mark.asyncio
async def test_anonymous_needs_auth_s3_public_object_is_false() -> None:
    client = MagicMock()
    with patch.object(anonymous, "_anonymous_s3_client", return_value=client):
        ref = RemoteRef(ref="s3://bucket/traces/t.jsonl?versionId=v1")
        assert await anonymous.anonymous_needs_auth(ref) is False
    client.head_object.assert_called_once_with(
        Bucket="bucket", Key="traces/t.jsonl", VersionId="v1"
    )


@pytest.mark.asyncio
async def test_anonymous_needs_auth_s3_prefix_heads_its_manifest() -> None:
    client = MagicMock()
    with patch.object(anonymous, "_anonymous_s3_client", return_value=client):
        assert (
            await anonymous.anonymous_needs_auth(RemoteRef(ref="s3://bucket/traces/"))
            is False
        )
    client.head_object.assert_called_once_with(
        Bucket="bucket", Key="traces/aiperf.json"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client_factory",
    [
        param(lambda: MagicMock(head_object=MagicMock(side_effect=RuntimeError("403"))), id="refused"),
        param(MagicMock(side_effect=ImportError("botocore")), id="no-botocore"),
    ],
)  # fmt: skip
async def test_anonymous_needs_auth_s3_any_failure_is_true(client_factory) -> None:
    with patch.object(anonymous, "_anonymous_s3_client", client_factory):
        assert (
            await anonymous.anonymous_needs_auth(RemoteRef(ref="s3://bucket/t.jsonl"))
            is True
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,expected",
    [
        param("https://h.example/t.jsonl", False, id="plain"),
        param("https://h.example/t.jsonl?X-Amz-Signature=<redacted>", True, id="redacted-presigned"),
        param("https://h.example/t.jsonl?sig=abc&se=2026-10-15", True, id="raw-sas"),
        param("https://user:pw@h.example/t.jsonl", True, id="userinfo"),
    ],
)  # fmt: skip
async def test_anonymous_needs_auth_url_needs_auth_only_with_credential(
    url: str, expected: bool
) -> None:
    assert await anonymous.anonymous_needs_auth(RemoteRef(ref=url)) is expected


@pytest.mark.asyncio
async def test_anonymous_needs_auth_offline_is_unknown_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Environment.DATASET, "REMOTE_OFFLINE", True)
    with patch("huggingface_hub.HfApi.dataset_info") as dataset_info:
        assert (
            await anonymous.anonymous_needs_auth(RemoteRef(ref="hf://org/repo")) is None
        )
    dataset_info.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ref,probe_name,expected",
    [
        param("hf://org/repo", "_hf_needs_auth", None, id="hf-unknown"),
        param("s3://bucket/t.jsonl", "_s3_needs_auth", True, id="s3-assumes-auth"),
    ],
)  # fmt: skip
async def test_anonymous_needs_auth_probe_timeout_answers_within_budget(
    monkeypatch: pytest.MonkeyPatch, ref: str, probe_name: str, expected: bool | None
) -> None:
    release = threading.Event()

    def hang(_ref: RemoteRef) -> bool:
        release.wait(5)
        return False

    monkeypatch.setattr(anonymous, "PROBE_TIMEOUT_S", 0.05)
    monkeypatch.setattr(anonymous, probe_name, hang)
    monkeypatch.setattr(anonymous.service, "remote_offline", lambda: False)
    start = time.monotonic()
    try:
        assert await anonymous.anonymous_needs_auth(RemoteRef(ref=ref)) is expected
        assert time.monotonic() - start < 2
    finally:
        release.set()
