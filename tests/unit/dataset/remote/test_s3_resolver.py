# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import errno
import io
import logging
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

import orjson
import pytest
from pytest import param

pytest.importorskip("botocore")

import botocore.session  # noqa: E402
from botocore.exceptions import (  # noqa: E402
    ClientError,
    ConnectionClosedError,
    EndpointConnectionError,
    NoCredentialsError,
    ReadTimeoutError,
)
from botocore.response import StreamingBody  # noqa: E402
from botocore.stub import Stubber  # noqa: E402

from aiperf.common.constants import IS_WINDOWS  # noqa: E402
from aiperf.config.dataset.remote_ref import RemoteRef  # noqa: E402
from aiperf.dataset.remote.cache import entry_dir, remote_cache_dir  # noqa: E402
from aiperf.dataset.remote.errors import (  # noqa: E402
    RemoteErrorKind,
    RemoteSourceError,
    RemoteStage,
)
from aiperf.dataset.remote.index import read_index  # noqa: E402
from aiperf.dataset.remote.models import ProbeResult  # noqa: E402
from aiperf.dataset.remote.s3 import (  # noqa: E402
    S3SourceResolver,
    _download_object,
    _is_safe_relpath,
    _Stopped,
)
from aiperf.dataset.remote.service import cached_source, materialize_probe  # noqa: E402
from aiperf.plugin import plugins  # noqa: E402
from aiperf.plugin.enums import PluginType  # noqa: E402

BODY = b'{"timestamp": 1}\n'
MANIFEST = orjson.dumps(
    {"aiperf_manifest_version": 1, "format": "weka_trace", "path": "traces"}
)


def _client():
    return botocore.session.get_session().create_client(
        "s3", region_name="us-east-1", aws_access_key_id="x", aws_secret_access_key="y"
    )


def _body(data: bytes) -> StreamingBody:
    return StreamingBody(io.BytesIO(data), len(data))


def _missing(stubber: Stubber, method: str, params: dict) -> None:
    stubber.add_client_error(method, service_error_code="404", http_status_code=404, expected_params=params)  # fmt: skip


def _client_error(code: str, status: int, op: str = "GetObject") -> ClientError:
    return ClientError(
        {"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}}, op
    )


def _object_probe(version: str, ref: str = "s3://bucket/t.jsonl") -> ProbeResult:
    key = ref.removeprefix("s3://bucket/").partition("?")[0]
    pinned = RemoteRef(ref=ref).with_version(version)
    return ProbeResult(
        ref=pinned, pinned_ref=pinned.display, data_relpath=key, data_is_dir=False,
        filename=Path(key).name, manifest=None, manifest_sha256=None, needs_auth=True,
    )  # fmt: skip


def _stub_directory_probe(s: Stubber, contents: list[dict]) -> None:
    s.add_response("get_object", {"Body": _body(MANIFEST), "ETag": '"m"'}, {"Bucket": "bucket", "Key": "set/aiperf.json"})  # fmt: skip
    _missing(s, "head_object", {"Bucket": "bucket", "Key": "set/traces"})
    _stub_listing(s, contents)


def _stub_listing(s: Stubber, contents: list[dict]) -> None:
    s.add_response("list_objects_v2", {"Contents": contents, "IsTruncated": False}, {"Bucket": "bucket", "Prefix": "set/traces/"})  # fmt: skip


def test_s3_resolver_is_registered() -> None:
    assert (
        plugins.get_class(PluginType.DATASET_SOURCE_RESOLVER, "s3") is S3SourceResolver
    )


@pytest.mark.asyncio
async def test_versioned_file_probe_prefix_and_download(tmp_path: Path) -> None:
    client = _client()
    with Stubber(client) as s:
        s.add_response("head_object", {"ETag": '"e1"', "VersionId": "v9", "ContentLength": len(BODY)}, {"Bucket": "bucket", "Key": "my traces/t.jsonl"})  # fmt: skip
        _missing(s, "get_object", {"Bucket": "bucket", "Key": "my traces/t.jsonl.aiperf.json"})  # fmt: skip
        s.add_response("get_object", {"Body": _body(BODY[:4]), "ContentRange": f"bytes 0-3/{len(BODY)}"},
                       {"Bucket": "bucket", "Key": "my traces/t.jsonl", "Range": "bytes=0-3", "VersionId": "v9"})  # fmt: skip
        s.add_response("get_object", {"Body": _body(BODY)}, {"Bucket": "bucket", "Key": "my traces/t.jsonl", "VersionId": "v9"})  # fmt: skip
        resolver = S3SourceResolver(client=client)
        probe = await resolver.probe(RemoteRef(ref="s3://bucket/my traces/t.jsonl"))
        assert probe.ref.resolved_version == "v:v9"
        assert probe.pinned_ref == "s3://bucket/my traces/t.jsonl?versionId=v9"
        assert await resolver.read_prefix(probe, 4) == (BODY[:4], False)
        source = await resolver.materialize(probe)
    assert source.local_path.read_bytes() == BODY


@pytest.mark.asyncio
async def test_pinned_ref_quotes_the_version_id() -> None:
    client = MagicMock()
    client.head_object.return_value = {"ETag": '"e"', "VersionId": "a+b/c"}
    client.get_object.side_effect = _client_error("NoSuchKey", 404)
    probe = await S3SourceResolver(client=client).probe(
        RemoteRef(ref="s3://bucket/t.jsonl")
    )
    assert probe.pinned_ref == "s3://bucket/t.jsonl?versionId=a%2Bb%2Fc"
    assert RemoteRef(ref=probe.pinned_ref).parsed.revision == "a+b/c"


@pytest.mark.asyncio
async def test_unversioned_object_downloads_with_if_match() -> None:
    client = _client()
    with Stubber(client) as s:
        s.add_response("head_object", {"ETag": '"e1"'}, {"Bucket": "bucket", "Key": "t.jsonl"})  # fmt: skip
        _missing(s, "get_object", {"Bucket": "bucket", "Key": "t.jsonl.aiperf.json"})
        s.add_response("get_object", {"Body": _body(BODY)}, {"Bucket": "bucket", "Key": "t.jsonl", "IfMatch": '"e1"'})  # fmt: skip
        resolver = S3SourceResolver(client=client)
        probe = await resolver.probe(RemoteRef(ref="s3://bucket/t.jsonl"))
        assert probe.ref.resolved_version == 'etag:"e1"'
        assert probe.pinned_ref == "s3://bucket/t.jsonl"
        source = await resolver.materialize(probe)
    assert source.local_path.read_bytes() == BODY


@pytest.mark.asyncio
async def test_object_replaced_after_pin_is_changed_after_pin() -> None:
    client = _client()
    with Stubber(client) as s:
        s.add_client_error("get_object", service_error_code="PreconditionFailed", http_status_code=412,
                           expected_params={"Bucket": "bucket", "Key": "t.jsonl", "IfMatch": '"e1"'})  # fmt: skip
        with pytest.raises(RemoteSourceError, match="changed after AIPerf pinned") as exc:  # fmt: skip
            await S3SourceResolver(client=client).materialize(_object_probe('etag:"e1"'))  # fmt: skip
    assert exc.value.kind == RemoteErrorKind.NOT_FOUND
    assert list(entry_dir(_object_probe('etag:"e1"').ref).glob("*")) == []


@pytest.mark.asyncio
async def test_pinned_version_deleted_is_changed_after_pin() -> None:
    client = _client()
    probe = _object_probe("v:v1", "s3://bucket/t.jsonl?versionId=v1")
    with Stubber(client) as s:
        s.add_client_error("get_object", service_error_code="NoSuchVersion", http_status_code=404,
                           expected_params={"Bucket": "bucket", "Key": "t.jsonl", "VersionId": "v1"})  # fmt: skip
        with pytest.raises(RemoteSourceError, match="changed after AIPerf pinned"):
            await S3SourceResolver(client=client).materialize(probe)


@pytest.mark.asyncio
async def test_reprobe_of_pinned_version_reads_it_after_a_newer_upload() -> None:
    client = _client()
    pinned = RemoteRef(ref="s3://bucket/t.jsonl").with_version("v:v1")
    with Stubber(client) as s:
        s.add_response("head_object", {"ETag": '"e1"', "VersionId": "v1"}, {"Bucket": "bucket", "Key": "t.jsonl", "VersionId": "v1"})  # fmt: skip
        _missing(s, "get_object", {"Bucket": "bucket", "Key": "t.jsonl.aiperf.json"})
        probe = await S3SourceResolver(client=client).probe(pinned)
    assert probe.ref.resolved_version == "v:v1"
    assert probe.pinned_ref == "s3://bucket/t.jsonl?versionId=v1"


@pytest.mark.asyncio
async def test_reprobe_of_deleted_pinned_version_is_changed_after_pin() -> None:
    client = _client()
    pinned = RemoteRef(ref="s3://bucket/t.jsonl").with_version("v:v1")
    with Stubber(client) as s:
        _missing(s, "head_object", {"Bucket": "bucket", "Key": "t.jsonl", "VersionId": "v1"})  # fmt: skip
        with pytest.raises(RemoteSourceError, match="changed after AIPerf pinned"):
            await S3SourceResolver(client=client).probe(pinned)


@pytest.mark.asyncio
async def test_missing_version_id_is_revision_not_found() -> None:
    client = _client()
    with Stubber(client) as s:
        _missing(s, "head_object", {"Bucket": "bucket", "Key": "t.jsonl", "VersionId": "v1"})  # fmt: skip
        with pytest.raises(RemoteSourceError, match="v1") as exc:
            await S3SourceResolver(client=client).probe(
                RemoteRef(ref="s3://bucket/t.jsonl?versionId=v1")
            )
    assert exc.value.kind == RemoteErrorKind.REVISION_NOT_FOUND


@pytest.mark.asyncio
async def test_sidecar_access_denied_is_treated_as_absent() -> None:
    client = _client()
    with Stubber(client) as s:
        s.add_response("head_object", {"ETag": '"e1"'}, {"Bucket": "bucket", "Key": "t.jsonl"})  # fmt: skip
        s.add_client_error("get_object", service_error_code="AccessDenied", http_status_code=403,
                           expected_params={"Bucket": "bucket", "Key": "t.jsonl.aiperf.json"})  # fmt: skip
        probe = await S3SourceResolver(client=client).probe(
            RemoteRef(ref="s3://bucket/t.jsonl")
        )
    assert probe.manifest is None


@pytest.mark.asyncio
async def test_read_prefix_of_empty_object_is_whole() -> None:
    client = _client()
    with Stubber(client) as s:
        s.add_client_error("get_object", service_error_code="InvalidRange", http_status_code=416,
                           expected_params={"Bucket": "bucket", "Key": "t.jsonl", "Range": "bytes=0-63", "IfMatch": '"e"'})  # fmt: skip
        result = await S3SourceResolver(client=client).read_prefix(
            _object_probe('etag:"e"'), 64
        )
    assert result == (b"", True)


@pytest.mark.asyncio
async def test_prefix_with_manifest_and_directory_data() -> None:
    client = _client()
    with Stubber(client) as s:
        _stub_directory_probe(s, [{"Key": "set/traces/a.json", "ETag": '"a"'}])
        probe = await S3SourceResolver(client=client).probe(
            RemoteRef(ref="s3://bucket/set/")
        )
    assert probe.data_is_dir and probe.data_relpath == "set/traces"
    assert probe.ref.resolved_version.startswith("list:")


@pytest.mark.asyncio
async def test_directory_materialize_pins_each_object_with_if_match() -> None:
    client = _client()
    contents = [
        {"Key": "set/traces/", "ETag": '"dir"'},
        {"Key": "set/traces/a.json", "ETag": '"a"'},
        {"Key": "set/traces/sub/b.json", "ETag": '"b"'},
    ]
    resolver = S3SourceResolver(client=client)
    with Stubber(client) as s:
        _stub_directory_probe(s, contents)
        _stub_listing(s, contents)
        s.add_response("get_object", {"Body": _body(b"A")}, {"Bucket": "bucket", "Key": "set/traces/a.json", "IfMatch": '"a"'})  # fmt: skip
        s.add_response("get_object", {"Body": _body(b"BB")}, {"Bucket": "bucket", "Key": "set/traces/sub/b.json", "IfMatch": '"b"'})  # fmt: skip
        probe = await resolver.probe(RemoteRef(ref="s3://bucket/set/"))
        with patch("aiperf.dataset.remote.service.get_resolver", return_value=resolver):  # fmt: skip
            source = await materialize_probe(probe)
        s.assert_no_pending_responses()
    assert source.local_path.name == "traces"
    assert (source.local_path / "a.json").read_bytes() == b"A"
    assert (source.local_path / "sub" / "b.json").read_bytes() == b"BB"
    assert [f.path for f in source.files] == ["set/traces/a.json", "set/traces/sub/b.json"]  # fmt: skip
    assert cached_source(probe.ref).files == source.files


@pytest.mark.asyncio
async def test_directory_object_replaced_mid_download_writes_no_index() -> None:
    client = _client()
    contents = [
        {"Key": "set/traces/a.json", "ETag": '"a"'},
        {"Key": "set/traces/b.json", "ETag": '"b"'},
    ]
    resolver = S3SourceResolver(client=client)
    with Stubber(client) as s:
        _stub_directory_probe(s, contents)
        _stub_listing(s, contents)
        s.add_response("get_object", {"Body": _body(b"A")}, {"Bucket": "bucket", "Key": "set/traces/a.json", "IfMatch": '"a"'})  # fmt: skip
        s.add_client_error("get_object", service_error_code="PreconditionFailed", http_status_code=412,
                           expected_params={"Bucket": "bucket", "Key": "set/traces/b.json", "IfMatch": '"b"'})  # fmt: skip
        probe = await resolver.probe(RemoteRef(ref="s3://bucket/set/"))
        with (
            patch("aiperf.dataset.remote.service.get_resolver", return_value=resolver),
            pytest.raises(RemoteSourceError, match="changed after AIPerf pinned") as exc,
        ):  # fmt: skip
            await materialize_probe(probe)
    assert exc.value.kind == RemoteErrorKind.NOT_FOUND
    assert read_index(probe.ref) is None


@pytest.mark.asyncio
async def test_directory_listing_changed_after_probe_is_changed_after_pin() -> None:
    client = _client()
    resolver = S3SourceResolver(client=client)
    with Stubber(client) as s:
        _stub_directory_probe(s, [{"Key": "set/traces/a.json", "ETag": '"a"'}])
        _stub_listing(s, [{"Key": "set/traces/a.json", "ETag": '"a2"'}])
        probe = await resolver.probe(RemoteRef(ref="s3://bucket/set/"))
        with pytest.raises(RemoteSourceError, match="changed after AIPerf pinned"):
            await resolver.materialize(probe)
        s.assert_no_pending_responses()


@pytest.mark.asyncio
async def test_missing_object_and_prefix_is_not_found() -> None:
    client = _client()
    with Stubber(client) as s:
        _missing(s, "head_object", {"Bucket": "bucket", "Key": "nope.jsonl"})
        _missing(s, "get_object", {"Bucket": "bucket", "Key": "nope.jsonl/aiperf.json"})
        with pytest.raises(RemoteSourceError) as exc:
            await S3SourceResolver(client=client).probe(
                RemoteRef(ref="s3://bucket/nope.jsonl")
            )
    assert exc.value.kind == RemoteErrorKind.NOT_FOUND


@pytest.mark.parametrize(
    "code,status,kind",
    [
        param("AccessDenied", 403, RemoteErrorKind.AUTH_DENIED, id="access-denied"),
        param("403", 403, RemoteErrorKind.AUTH_DENIED, id="head-403"),
        param("ExpiredToken", 400, RemoteErrorKind.AUTH_MISSING, id="expired-token"),
        param("SlowDown", 503, RemoteErrorKind.TRANSIENT, id="slow-down"),
        param("ThrottlingException", 400, RemoteErrorKind.TRANSIENT, id="throttling"),
        param("RequestTimeout", 400, RemoteErrorKind.TRANSIENT, id="request-timeout"),
        param("InternalError", 500, RemoteErrorKind.TRANSIENT, id="internal-error"),
        param("NoSuchBucket", 404, RemoteErrorKind.NOT_FOUND, id="no-such-bucket"),
        param("400", 400, RemoteErrorKind.NOT_FOUND, id="unmapped-400"),
    ],
)  # fmt: skip
@pytest.mark.asyncio
async def test_client_error_mapping(code: str, status: int, kind: RemoteErrorKind) -> None:  # fmt: skip
    client = _client()
    with Stubber(client) as s:
        s.add_client_error("head_object", service_error_code=code, http_status_code=status)  # fmt: skip
        with pytest.raises(RemoteSourceError) as exc:
            await S3SourceResolver(client=client).probe(
                RemoteRef(ref="s3://bucket/k.jsonl")
            )
    assert exc.value.kind == kind


@pytest.mark.parametrize(
    "error",
    [
        param(EndpointConnectionError(endpoint_url="https://s3.example"), id="endpoint"),
        param(ConnectionClosedError(endpoint_url="https://s3.example"), id="closed"),
        param(ReadTimeoutError(endpoint_url="https://s3.example"), id="read-timeout"),
    ],
)  # fmt: skip
@pytest.mark.asyncio
async def test_connection_errors_are_transient(error: Exception) -> None:
    client = MagicMock()
    client.head_object.side_effect = error
    with pytest.raises(RemoteSourceError) as exc:
        await S3SourceResolver(client=client).probe(RemoteRef(ref="s3://bucket/k.jsonl"))  # fmt: skip
    assert exc.value.kind == RemoteErrorKind.TRANSIENT


@pytest.mark.asyncio
async def test_no_credentials_is_auth_missing() -> None:
    client = MagicMock()
    client.head_object.side_effect = NoCredentialsError()
    with pytest.raises(RemoteSourceError, match="AWS credentials") as exc:
        await S3SourceResolver(client=client).probe(
            RemoteRef(ref="s3://bucket/k.jsonl")
        )
    assert exc.value.kind == RemoteErrorKind.AUTH_MISSING


@pytest.mark.asyncio
async def test_missing_botocore_is_missing_dependency() -> None:
    with (
        patch.dict("sys.modules", {"botocore.session": None}),
        pytest.raises(RemoteSourceError, match=r"aiperf\[aws\]") as exc,
    ):
        await S3SourceResolver().probe(RemoteRef(ref="s3://bucket/k.jsonl"))
    assert exc.value.kind == RemoteErrorKind.MISSING_DEPENDENCY


@pytest.mark.asyncio
async def test_client_is_created_once_off_the_event_loop() -> None:
    client = MagicMock()
    client.head_object.return_value = {"ETag": '"e"'}
    client.get_object.side_effect = _client_error("NoSuchKey", 404)
    created: list[bool] = []
    configs: list = []

    def create_client(service: str, *, config) -> MagicMock:
        assert service == "s3"
        created.append(threading.current_thread() is threading.main_thread())
        configs.append(config)
        return client

    resolver = S3SourceResolver()
    with patch("botocore.session.get_session") as get_session:
        get_session.return_value.create_client.side_effect = create_client
        await resolver.probe(RemoteRef(ref="s3://bucket/a.jsonl"))
        await resolver.probe(RemoteRef(ref="s3://bucket/b.jsonl"))
    assert created == [False]
    # One attempt with short timeouts: the service's transient retry is the only retry layer.
    assert configs[0].connect_timeout == 10
    assert configs[0].read_timeout == 30
    assert configs[0].retries == {"mode": "standard", "max_attempts": 1}


@pytest.mark.asyncio
async def test_local_disk_full_is_not_transient_and_names_cache_dir() -> None:
    client = MagicMock()
    client.get_object.return_value = {"Body": _body(BODY)}
    disk_full = OSError(errno.ENOSPC, "No space left on device")
    with (
        patch("aiperf.dataset.remote.s3.write_atomically", side_effect=disk_full),
        pytest.raises(RemoteSourceError) as exc,
    ):
        await S3SourceResolver(client=client).materialize(_object_probe('etag:"e"'))
    assert exc.value.kind == RemoteErrorKind.NOT_FOUND
    assert str(remote_cache_dir()) in str(exc.value)


@pytest.mark.asyncio
async def test_cancelled_download_stops_the_thread_and_leaves_no_partial() -> None:
    started = threading.Event()

    def chunks(chunk_size: int) -> Iterator[bytes]:
        yield b"x" * 8
        started.set()
        # Bounded so a resolver that ignores cancellation fails instead of hanging.
        for _ in range(2000):
            time.sleep(0.005)
            yield b"x" * 8

    body = MagicMock()
    body.iter_chunks.side_effect = chunks
    client = MagicMock()
    client.get_object.return_value = {"Body": body}
    probe = _object_probe('etag:"e"')
    task = asyncio.create_task(S3SourceResolver(client=client).materialize(probe))
    assert await asyncio.to_thread(started.wait, 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # The thread has already finished: it closed the body and removed its partial file.
    body.close.assert_called_once()
    assert list(entry_dir(probe.ref).glob("*")) == []


@pytest.mark.parametrize(
    "rel,safe_on_posix,safe_on_windows",
    [
        param("a.json", True, True, id="plain"),
        param("sub/a.json", True, True, id="nested"),
        param("../x.json", False, False, id="dot-dot"),
        param("./x.json", False, False, id="dot"),
        param("a//x.json", False, False, id="empty-segment"),
        param("a\\b.json", False, False, id="backslash"),
        param("x/D:y.json", True, False, id="drive-in-later-segment"),
        param("C:/x.json", True, False, id="drive-first-segment"),
        param("x.json:stream", True, False, id="alternate-data-stream"),
        param("conv-2026-10-13T12:00:00.json", True, False, id="timestamped-name"),
    ],
)  # fmt: skip
@pytest.mark.parametrize("is_windows", [False, True])
def test_is_safe_relpath_rejects_colons_and_drives_only_on_windows(
    rel: str, safe_on_posix: bool, safe_on_windows: bool, is_windows: bool
) -> None:
    with patch("aiperf.dataset.remote.s3.IS_WINDOWS", is_windows):
        assert _is_safe_relpath(rel) is (
            safe_on_windows if is_windows else safe_on_posix
        )


@pytest.mark.parametrize(
    "unsafe_key,is_windows",
    [
        param("set/traces/../x.json", False, id="dot-dot"),
        param("set/traces//x.json", False, id="empty-segment"),
        param("set/traces/a\\b.json", False, id="backslash"),
        param("set/traces/x/D:y.json", True, id="windows-drive-in-later-segment"),
        param("set/traces/C:/x.json", True, id="windows-drive-first-segment"),
        param("set/traces/x.json:stream", True, id="windows-alternate-data-stream"),
    ],
)  # fmt: skip
@pytest.mark.asyncio
async def test_directory_key_unsafe_as_local_path_fails_access_naming_the_key(
    unsafe_key: str, is_windows: bool
) -> None:
    client = _client()
    contents = [
        {"Key": "set/traces/a.json", "ETag": '"a"'},
        {"Key": unsafe_key, "ETag": '"u"'},
    ]
    resolver = S3SourceResolver(client=client)
    with (
        Stubber(client) as s,
        patch("aiperf.dataset.remote.s3.IS_WINDOWS", is_windows),
        pytest.raises(RemoteSourceError) as exc,
    ):
        _stub_directory_probe(s, contents)
        await resolver.probe(RemoteRef(ref="s3://bucket/set/"))
    assert exc.value.kind == RemoteErrorKind.MANIFEST_INVALID
    assert exc.value.stage == RemoteStage.ACCESS
    assert repr(unsafe_key) in str(exc.value)


@pytest.mark.skipif(IS_WINDOWS, reason="':' is not valid in a Windows file name")
@pytest.mark.asyncio
async def test_directory_key_with_colon_is_downloaded_off_windows() -> None:
    client = _client()
    key = "set/traces/conv-2026-10-13T12:00:00.json"
    contents = [{"Key": key, "ETag": '"c"'}]
    resolver = S3SourceResolver(client=client)
    with Stubber(client) as s, patch("aiperf.dataset.remote.s3.IS_WINDOWS", False):
        _stub_directory_probe(s, contents)
        _stub_listing(s, contents)
        s.add_response("get_object", {"Body": _body(b"C")}, {"Bucket": "bucket", "Key": key, "IfMatch": '"c"'})  # fmt: skip
        probe = await resolver.probe(RemoteRef(ref="s3://bucket/set/"))
        source = await resolver.materialize(probe)
        s.assert_no_pending_responses()
    assert [f.path for f in source.files] == [key]


@pytest.mark.parametrize(
    "key",
    [
        param("dir/C:trace.jsonl", id="drive-relative"),
        param("dir/trace.jsonl:stream", id="alternate-data-stream"),
    ],
)  # fmt: skip
@pytest.mark.asyncio
async def test_single_object_name_unsafe_on_windows_fails_access_before_download(
    key: str,
) -> None:
    """``entry / "C:trace.jsonl"`` resets to drive C on Windows, outside the cache."""
    client = _client()
    with (
        Stubber(client) as s,
        patch("aiperf.dataset.remote.s3.IS_WINDOWS", True),
        pytest.raises(RemoteSourceError) as exc,
    ):
        s.add_response("head_object", {"ETag": '"e1"'}, {"Bucket": "bucket", "Key": key})  # fmt: skip
        _missing(s, "get_object", {"Bucket": "bucket", "Key": key + ".aiperf.json"})
        await S3SourceResolver(client=client).probe(RemoteRef(ref=f"s3://bucket/{key}"))
    assert exc.value.kind == RemoteErrorKind.MANIFEST_INVALID
    assert exc.value.stage == RemoteStage.ACCESS
    assert repr(key) in str(exc.value)


@pytest.mark.skipif(IS_WINDOWS, reason="':' is not valid in a Windows file name")
@pytest.mark.asyncio
async def test_single_object_with_colon_downloads_off_windows() -> None:
    client = _client()
    key = "dir/C:trace.jsonl"
    with Stubber(client) as s, patch("aiperf.dataset.remote.s3.IS_WINDOWS", False):
        s.add_response("head_object", {"ETag": '"e1"'}, {"Bucket": "bucket", "Key": key})  # fmt: skip
        _missing(s, "get_object", {"Bucket": "bucket", "Key": key + ".aiperf.json"})
        s.add_response("get_object", {"Body": _body(BODY)}, {"Bucket": "bucket", "Key": key, "IfMatch": '"e1"'})  # fmt: skip
        resolver = S3SourceResolver(client=client)
        source = await resolver.materialize(
            await resolver.probe(RemoteRef(ref=f"s3://bucket/{key}"))
        )
    assert source.local_path.name == "C:trace.jsonl"
    assert source.local_path.read_bytes() == BODY


@pytest.mark.asyncio
async def test_single_object_resolving_outside_the_entry_is_rejected() -> None:
    probe = _object_probe('etag:"e"').model_copy(
        update={"filename": "../../escape.jsonl"}
    )
    client = MagicMock()
    with pytest.raises(RemoteSourceError, match="outside") as exc:
        await S3SourceResolver(client=client).materialize(probe)
    assert exc.value.kind == RemoteErrorKind.MANIFEST_INVALID
    client.get_object.assert_not_called()


@pytest.mark.asyncio
async def test_directory_marker_keys_are_ignored() -> None:
    client = _client()
    contents = [
        {"Key": "set/traces/", "ETag": '"d"'},
        {"Key": "set/traces/sub/", "ETag": '"d"'},
        {"Key": "set/traces/a.json", "ETag": '"a"'},
    ]
    resolver = S3SourceResolver(client=client)
    with Stubber(client) as s:
        _stub_directory_probe(s, contents)
        _stub_listing(s, contents)
        s.add_response("get_object", {"Body": _body(b"A")}, {"Bucket": "bucket", "Key": "set/traces/a.json", "IfMatch": '"a"'})  # fmt: skip
        probe = await resolver.probe(RemoteRef(ref="s3://bucket/set/"))
        source = await resolver.materialize(probe)
        s.assert_no_pending_responses()
    assert [f.path for f in source.files] == ["set/traces/a.json"]


@pytest.mark.asyncio
async def test_directory_key_resolving_outside_the_entry_is_rejected() -> None:
    client = _client()
    contents = [{"Key": "set/traces/../../escape.json", "ETag": '"e"'}]
    resolver = S3SourceResolver(client=client)
    with (
        Stubber(client) as s,
        patch("aiperf.dataset.remote.s3._is_safe_relpath", return_value=True),
    ):
        _stub_directory_probe(s, contents)
        _stub_listing(s, contents)
        probe = await resolver.probe(RemoteRef(ref="s3://bucket/set/"))
        with pytest.raises(RemoteSourceError, match="outside") as exc:
            await resolver.materialize(probe)
        s.assert_no_pending_responses()
    assert exc.value.kind == RemoteErrorKind.MANIFEST_INVALID


@pytest.mark.asyncio
async def test_cancel_before_get_object_never_requests_it() -> None:
    # A one-worker default executor, kept busy, holds the download thread back so
    # the cancellation lands before it reaches get_object.
    executor = ThreadPoolExecutor(max_workers=1)
    asyncio.get_running_loop().set_default_executor(executor)
    release = threading.Event()
    executor.submit(release.wait, 5)
    client = MagicMock()
    probe = _object_probe('etag:"e"')
    task = asyncio.create_task(S3SourceResolver(client=client).materialize(probe))
    for _ in range(10):
        await asyncio.sleep(0)
    task.cancel()
    for _ in range(5):
        await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    client.get_object.assert_not_called()
    assert list(entry_dir(probe.ref).glob("*")) == []


def test_stop_after_get_object_reads_no_chunk(tmp_path: Path) -> None:
    stop = threading.Event()
    reads: list[int] = []

    def chunks(chunk_size: int) -> Iterator[bytes]:
        reads.append(chunk_size)
        yield b"x"

    body = MagicMock()
    body.iter_chunks.side_effect = chunks

    def get_object(**params: str) -> dict:
        stop.set()
        return {"Body": body}

    client = MagicMock()
    client.get_object.side_effect = get_object
    with pytest.raises(_Stopped):
        _download_object(client, {"Bucket": "bucket", "Key": "t.jsonl"}, tmp_path / "t.jsonl", stop)  # fmt: skip
    assert reads == []
    body.close.assert_called_once()
    assert list(tmp_path.glob("*")) == []


def _parquet_keys(*names: str) -> list[dict]:
    return [{"Key": f"set/traces/{n}", "ETag": f'"{i}"'} for i, n in enumerate(names)]


@pytest.mark.asyncio
async def test_prefix_manifest_parquet_folder_lists_train_shards(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = _client()
    with Stubber(client) as s, caplog.at_level(logging.INFO):
        _stub_directory_probe(s, _parquet_keys(
            "train-00001-of-00002.parquet", "train-00000-of-00002.parquet",
            "test-00000-of-00001.parquet", "README.md",
        ))  # fmt: skip
        probe = await S3SourceResolver(client=client).probe(
            RemoteRef(ref="s3://bucket/set/")
        )
    assert probe.data_files == [
        "set/traces/train-00000-of-00002.parquet",
        "set/traces/train-00001-of-00002.parquet",
    ]
    assert [
        r.getMessage() for r in caplog.records if "holds splits" in r.getMessage()
    ] == [
        "Remote dataset s3://bucket/set/: 'set/traces' holds splits test, train; using 'train'"
    ]


@pytest.mark.asyncio
async def test_prefix_manifest_json_folder_has_no_shards() -> None:
    client = _client()
    with Stubber(client) as s:
        _stub_directory_probe(s, [{"Key": "set/traces/a.json", "ETag": '"a"'}])
        probe = await S3SourceResolver(client=client).probe(
            RemoteRef(ref="s3://bucket/set/")
        )
    assert probe.data_is_dir and probe.data_files is None


@pytest.mark.asyncio
async def test_prefix_manifest_parquet_folder_without_train_fails() -> None:
    client = _client()
    with Stubber(client) as s:
        _stub_directory_probe(s, _parquet_keys(
            "test-00000-of-00001.parquet", "validation-00000-of-00001.parquet",
        ))  # fmt: skip
        with pytest.raises(RemoteSourceError) as exc_info:
            await S3SourceResolver(client=client).probe(
                RemoteRef(ref="s3://bucket/set/")
            )
    assert exc_info.value.kind == RemoteErrorKind.MANIFEST_INVALID
    assert "test, validation" in str(exc_info.value)


@pytest.mark.asyncio
async def test_prefix_manifest_spark_folder_lists_part_files() -> None:
    client = _client()
    with Stubber(client) as s:
        _stub_directory_probe(s, _parquet_keys(
            "part-00000-abc.snappy.parquet", "part-00001-abc.snappy.parquet",
            "_SUCCESS", "_committed_123", ".part-00000-abc.snappy.parquet.crc",
        ))  # fmt: skip
        probe = await S3SourceResolver(client=client).probe(
            RemoteRef(ref="s3://bucket/set/")
        )
    assert probe.data_files == [
        "set/traces/part-00000-abc.snappy.parquet",
        "set/traces/part-00001-abc.snappy.parquet",
    ]
