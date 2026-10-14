# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import orjson
import pytest
from aiohttp import web
from pytest import param

from aiperf.common.redact import REDACTED_VALUE
from aiperf.config.dataset.remote_ref import RemoteRef
from aiperf.dataset.remote import service
from aiperf.dataset.remote.cache import entry_dir
from aiperf.dataset.remote.errors import RemoteErrorKind, RemoteSourceError
from aiperf.dataset.remote.http import HTTPSourceResolver, _filename_of
from aiperf.dataset.remote.models import UNVERSIONED
from aiperf.dataset.remote.protocols import SourceResolverProtocol
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType

BODY = b'{"timestamp": 1, "input_length": 4, "output_length": 2}\n'


def test_registered_for_https_and_http() -> None:
    for scheme in ("https", "http"):
        cls = plugins.get_class(PluginType.DATASET_SOURCE_RESOLVER, scheme)
        assert cls is HTTPSourceResolver
        assert isinstance(cls(), SourceResolverProtocol)


async def _serve(http_server, tmp_path: Path, files: dict[str, bytes]) -> str:
    root = tmp_path / "srv"
    root.mkdir(exist_ok=True)
    for name, data in files.items():
        (root / name).write_bytes(data)
    app = web.Application()
    app.router.add_static("/", root)
    return await http_server(app)


@pytest.mark.asyncio
async def test_probe_pins_etag_and_reads_sidecar(http_server, tmp_path: Path) -> None:
    sidecar = orjson.dumps({"aiperf_manifest_version": 1, "format": "mooncake_trace"})
    base = await _serve(
        http_server, tmp_path, {"t.jsonl": BODY, "t.jsonl.aiperf.json": sidecar}
    )
    probe = await HTTPSourceResolver().probe(RemoteRef(ref=f"{base}/t.jsonl"))
    assert probe.ref.resolved_version.startswith(("etag:", "lm:"))
    assert probe.filename == "t.jsonl" and not probe.data_is_dir
    assert probe.manifest is not None and probe.manifest.format == "mooncake_trace"


@pytest.mark.asyncio
async def test_presigned_url_skips_sidecar(http_server, tmp_path: Path) -> None:
    base = await _serve(http_server, tmp_path, {"t.jsonl": BODY})
    probe = await HTTPSourceResolver().probe(
        RemoteRef(ref=f"{base}/t.jsonl?X-Amz-Signature=s")
    )
    assert probe.manifest is None
    assert REDACTED_VALUE in probe.pinned_ref
    assert "X-Amz-Signature=s" not in probe.pinned_ref


@pytest.mark.asyncio
async def test_materialize_downloads_into_entry_dir(
    http_server, tmp_path: Path
) -> None:
    base = await _serve(http_server, tmp_path, {"t.jsonl": BODY})
    resolver = HTTPSourceResolver()
    source = await resolver.materialize(
        await resolver.probe(RemoteRef(ref=f"{base}/t.jsonl"))
    )
    assert source.local_path.read_bytes() == BODY
    assert source.local_path.name == "t.jsonl"
    assert source.files[0].size_bytes == len(BODY)


@pytest.mark.asyncio
async def test_changed_after_pin_is_an_error(http_server, tmp_path: Path) -> None:
    base = await _serve(http_server, tmp_path, {"t.jsonl": BODY})
    stale = RemoteRef(ref=f"{base}/t.jsonl").with_version('etag:"stale"')
    with pytest.raises(RemoteSourceError, match="changed"):
        await HTTPSourceResolver().probe(stale)


@pytest.mark.asyncio
async def test_changed_during_materialize_leaves_no_file(
    http_server, tmp_path: Path
) -> None:
    base = await _serve(http_server, tmp_path, {"t.jsonl": BODY})
    resolver = HTTPSourceResolver()
    probe = await resolver.probe(RemoteRef(ref=f"{base}/t.jsonl"))
    stale = probe.model_copy(update={"ref": probe.ref.with_version('etag:"stale"')})
    with pytest.raises(RemoteSourceError, match="changed") as exc:
        await resolver.materialize(stale)
    assert exc.value.kind == RemoteErrorKind.NOT_FOUND
    assert list(entry_dir(stale.ref).iterdir()) == []


@pytest.mark.asyncio
async def test_unversioned_server(http_server) -> None:
    async def bare(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse()
        await resp.prepare(request)
        # A body on a HEAD response would corrupt the kept-alive connection.
        if request.method == "GET":
            await resp.write(BODY)
        return resp

    app = web.Application()
    app.router.add_get("/t.jsonl", bare)
    probe = await HTTPSourceResolver().probe(
        RemoteRef(ref=f"{await http_server(app)}/t.jsonl")
    )
    assert probe.ref.resolved_version == UNVERSIONED


@pytest.mark.asyncio
async def test_last_modified_pin_tolerates_missing_content_length(
    http_server,
) -> None:
    """A full GET may be chunked (no Content-Length) where HEAD reported one."""
    last_modified = "Mon, 12 Oct 2026 00:00:00 GMT"

    async def handler(request: web.Request) -> web.StreamResponse:
        if request.method == "HEAD":
            return web.Response(
                headers={
                    "Last-Modified": last_modified,
                    "Content-Length": str(len(BODY)),
                }
            )
        resp = web.StreamResponse(headers={"Last-Modified": last_modified})
        resp.enable_chunked_encoding()
        await resp.prepare(request)
        await resp.write(BODY)
        return resp

    app = web.Application()
    app.router.add_get("/t.jsonl", handler)
    resolver = HTTPSourceResolver()
    probe = await resolver.probe(RemoteRef(ref=f"{await http_server(app)}/t.jsonl"))
    assert probe.ref.resolved_version == f"lm:{last_modified}|{len(BODY)}"
    source = await resolver.materialize(probe)
    assert source.local_path.read_bytes() == BODY


@pytest.mark.asyncio
async def test_read_prefix(http_server, tmp_path: Path) -> None:
    base = await _serve(http_server, tmp_path, {"t.jsonl": BODY})
    resolver = HTTPSourceResolver()
    probe = await resolver.probe(RemoteRef(ref=f"{base}/t.jsonl"))
    assert await resolver.read_prefix(probe, 4) == (BODY[:4], False)
    assert await resolver.read_prefix(probe, 4096) == (BODY, True)


def _sidecar_app(sidecar: web.Response) -> web.Application:
    """Serve BODY at /t.jsonl (strong ETag) and ``sidecar`` at its sidecar path."""

    async def data(_: web.Request) -> web.Response:
        return web.Response(body=BODY, headers={"ETag": '"v1"'})

    async def side(_: web.Request) -> web.Response:
        return sidecar

    app = web.Application()
    app.router.add_get("/t.jsonl", data)
    app.router.add_get("/t.jsonl.aiperf.json", side)
    return app


@pytest.mark.asyncio
async def test_html_sidecar_is_treated_as_absent(http_server) -> None:
    """SPA catch-alls and SSO landing pages answer any path with 200 HTML."""
    page = web.Response(text="<html>sign in</html>", content_type="text/html")
    url = f"{await http_server(_sidecar_app(page))}/t.jsonl"
    probe = await service.probe_ref(RemoteRef(ref=url))
    assert probe.manifest is None and probe.manifest_sha256 is None
    source = await service.materialize_probe(probe)
    assert source.local_path.read_bytes() == BODY


@pytest.mark.asyncio
async def test_oversize_sidecar_is_invalid(http_server) -> None:
    huge = web.Response(body=b" " * ((1 << 20) + 1), content_type="application/json")
    url = f"{await http_server(_sidecar_app(huge))}/t.jsonl"
    with pytest.raises(RemoteSourceError, match="larger than 1 MiB") as exc:
        await HTTPSourceResolver().probe(RemoteRef(ref=url))
    assert exc.value.kind == RemoteErrorKind.MANIFEST_INVALID


@pytest.mark.asyncio
async def test_sidecar_spanning_many_chunks_is_read_whole(http_server) -> None:
    raw = orjson.dumps({"aiperf_manifest_version": 1, "format": "mooncake_trace"})
    # Leading padding: a reader that stops after the first chunk never sees the JSON.
    padded = b" " * ((1 << 20) - len(raw)) + raw
    sidecar = web.Response(body=padded, content_type="application/json")
    url = f"{await http_server(_sidecar_app(sidecar))}/t.jsonl"
    probe = await HTTPSourceResolver().probe(RemoteRef(ref=url))
    assert probe.manifest is not None and probe.manifest.format == "mooncake_trace"


def _etag_app(head_etag: str, get_etag: str) -> web.Application:
    """HEAD and GET report different ETags, as nginx does when gzip weakens it."""

    async def handler(request: web.Request) -> web.Response:
        etag = head_etag if request.method == "HEAD" else get_etag
        return web.Response(body=BODY, headers={"ETag": etag})

    app = web.Application()
    app.router.add_get("/t.jsonl", handler)
    return app


@pytest.mark.asyncio
async def test_weak_etag_on_get_matches_strong_pin(http_server) -> None:
    base = await http_server(_etag_app('"x"', 'W/"x"'))
    url = f"{base}/t.jsonl"
    resolver = HTTPSourceResolver()
    probe = await resolver.probe(RemoteRef(ref=url))
    assert probe.ref.resolved_version == 'etag:"x"'
    source = await resolver.materialize(probe)
    assert source.local_path.read_bytes() == BODY


@pytest.mark.asyncio
async def test_weak_compare_still_detects_a_different_etag(http_server) -> None:
    base = await http_server(_etag_app('"x"', 'W/"y"'))
    url = f"{base}/t.jsonl"
    resolver = HTTPSourceResolver()
    probe = await resolver.probe(RemoteRef(ref=url))
    with pytest.raises(RemoteSourceError, match="changed"):
        await resolver.materialize(probe)


@pytest.mark.parametrize(
    "url, expected",
    [
        param("https://h/d/t.jsonl", "t.jsonl", id="plain"),
        param("https://h/d/my%20trace.jsonl", "my trace.jsonl", id="percent-decoded"),
        param("https://h/", "download", id="empty"),
        param("https://h/a/%2e%2e", "download", id="dot-dot"),
        param("https://h/a/..", "download", id="literal-dot-dot"),
        param("https://h/d/%5C..%5C..%5Cx.jsonl", "download", id="backslashes"),
        param("https://h/d/a%5Cb.jsonl", "download", id="backslash"),
    ],
)  # fmt: skip
def test_filename_of_never_escapes_the_entry_dir(url: str, expected: str) -> None:
    assert _filename_of(url) == expected
