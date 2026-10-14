# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import gzip
import tracemalloc
import zlib
from pathlib import Path

import pytest
from pytest import param

from aiperf.config.dataset.remote_ref import RemoteRef
from aiperf.dataset.remote import service
from aiperf.dataset.remote.cache import entry_dir
from aiperf.dataset.remote.index import write_index
from aiperf.dataset.remote.models import ProbeResult, RemoteFile, ResolvedSource
from aiperf.dataset.remote.sniff import (
    MAX_FIRST_LINE_BYTES,
    PREFIX_STEPS,
    remote_first_record_has_timestamp,
    sniff_first_record_timestamp,
)
from aiperf.dataset.trace_timestamps import first_record_has_timestamp

TS = b'{"timestamp": 5, "input_length": 3}\n{"timestamp": 6}\n'
NO_TS = b'{"input_length": 3}\n'


@pytest.mark.parametrize(
    "prefix,complete,expected",
    [
        param(TS, True, True, id="plain-timestamp"),
        param(NO_TS, True, False, id="plain-no-timestamp"),
        param(b'\n\n{"timing_events": [{"timestamp": 1}]}\n', True, True, id="timing-events-after-blank-lines"),
        param(TS[:10], False, None, id="first-line-incomplete"),
        param(b'{"timestamp": 5}', True, True, id="no-trailing-newline-whole-file"),
        param(b"not json\n", True, False, id="invalid-json"),
        param(b"[1, 2]\n", True, False, id="not-an-object"),
        param(b"", True, False, id="empty-file"),
        param(b"", False, None, id="nothing-yet"),
        param(b"\x28\xb5\x2f\xfd binary", True, False, id="zstd-matches-local-behavior"),
    ],
)  # fmt: skip
def test_sniff(prefix: bytes, complete: bool, expected: bool | None) -> None:
    assert sniff_first_record_timestamp(prefix, complete=complete) is expected


def test_gzip_prefix_cut_mid_stream() -> None:
    blob = gzip.compress(TS * 2000)
    assert sniff_first_record_timestamp(blob[:200], complete=False) is True


def test_gzip_prefix_too_short_for_first_line() -> None:
    blob = gzip.compress(b'{"pad": "' + b"x" * 500_000 + b'", "timestamp": 1}\n')
    assert sniff_first_record_timestamp(blob[:40], complete=False) is None


def test_gzip_prefix_is_inflated_only_until_the_first_line() -> None:
    blob = gzip.compress(TS + b"\n" * (64 * 1024 * 1024))
    tracemalloc.start()
    try:
        assert sniff_first_record_timestamp(blob, complete=True) is True
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 4 * 1024 * 1024


def _gzip_single_line(nbytes: int) -> bytes:
    """Gzip of one ``nbytes``-long line with no line end, built without holding it inflated."""
    encoder = zlib.compressobj(wbits=31)
    chunk = b"x" * (1024 * 1024)
    parts = [encoder.compress(b'{"pad": "')]
    parts += [encoder.compress(chunk) for _ in range(nbytes // len(chunk))]
    return b"".join([*parts, encoder.flush()])


@pytest.mark.parametrize("complete", [False, True])
def test_gzip_single_line_bomb_is_capped_and_left_undecided(complete: bool) -> None:
    """A first line with no line end stops inflating at the cap; the download decides."""
    blob = _gzip_single_line(128 * 1024 * 1024)
    assert len(blob) < PREFIX_STEPS[-1]
    tracemalloc.start()
    try:
        assert sniff_first_record_timestamp(blob, complete=complete) is None
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 3 * MAX_FIRST_LINE_BYTES < 128 * 1024 * 1024


@pytest.mark.asyncio
async def test_capped_first_line_falls_back_to_the_download(monkeypatch) -> None:
    fake = _FakeResolver(_gzip_single_line(64 * 1024 * 1024))
    monkeypatch.setattr(service, "get_resolver", lambda scheme: fake)
    ref = RemoteRef(ref="https://example.com/t.jsonl.gz").with_version("etag-1")
    source = _cache(ref, fake.data, filename="t.jsonl.gz")

    async def materialize(probe: ProbeResult) -> ResolvedSource:
        return source

    monkeypatch.setattr(service, "materialize_probe", materialize)
    assert await remote_first_record_has_timestamp(_probe(ref, "t.jsonl.gz")) is False
    # The whole object fits in the first step, so no larger prefix is requested.
    assert fake.reads == [PREFIX_STEPS[0]]


# Every payload is checked against the local-file answer, so the remote path reaches
# the decision first_record_has_timestamp would reach on a downloaded copy.
_PARITY_PAYLOADS = [
    param(TS, id="timestamp"),
    param(NO_TS, id="no-timestamp"),
    param(b'{"timestamp": null}\n', id="null-timestamp"),
    param(b'{"timing_events": []}\n', id="empty-timing-events"),
    param(b'{"timing_events": [1]}\n', id="non-dict-timing-event"),
    param(b'{"timestamp": 1}\r\n{"x": 1}\r\n', id="crlf"),
    param(b'{"timestamp": 1}\r{"x": 1}\r', id="bare-cr-line-endings"),
    param(b'not json\r{"timestamp": 1}\r', id="bare-cr-first-line-invalid"),
    param(b' \t\r\n\n{"timestamp": 1}\n', id="whitespace-lines-first"),
    param('  \n{"timestamp": 1}\n'.encode(), id="unicode-whitespace-line-first"),
    param(b'\xef\xbb\xbf{"timestamp": 1}\n', id="utf8-bom"),
    param(b'{"timestamp": 1, "x": "\xff"}\n', id="invalid-utf8-first-line"),
    param('{"timestamp": 1, "x": "é中"}\n'.encode(), id="multibyte-utf8-first-line"),
    param(b"[1, 2]\n", id="array"),
    param(b"   \n\n", id="only-blank-lines"),
    param(b"", id="empty"),
    param(b"\x28\xb5\x2f\xfd binary", id="zstd"),
    param(gzip.compress(TS), id="gzip-timestamp"),
    param(gzip.compress(NO_TS), id="gzip-no-timestamp"),
    param(gzip.compress(b'{"time') + gzip.compress(b'stamp": 1}\n'), id="gzip-record-spans-members"),
    param(gzip.compress(TS) + b"\x00" * 8, id="gzip-zero-padding"),
    param(b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03" + b"\xff" * 20 + b"\x00" * 8, id="gzip-corrupt-deflate"),
    param(b"\x1f\x8b\x09\x00\x00\x00\x00\x00\x00\x00", id="gzip-bad-method"),
]  # fmt: skip


def _local_answer(tmp_path: Path, payload: bytes) -> bool:
    path = tmp_path / "trace.jsonl"
    path.write_bytes(payload)
    return first_record_has_timestamp(path)


@pytest.mark.parametrize(
    "payload",
    [*_PARITY_PAYLOADS, param(gzip.compress(TS)[:-12], id="gzip-truncated")],
)  # fmt: skip
def test_whole_file_sniff_matches_local_first_record_check(
    tmp_path: Path, payload: bytes
) -> None:
    assert sniff_first_record_timestamp(payload, complete=True) is _local_answer(
        tmp_path, payload
    )


@pytest.mark.parametrize("payload", _PARITY_PAYLOADS)
def test_no_prefix_decides_differently_from_the_whole_file(
    tmp_path: Path, payload: bytes
) -> None:
    local = _local_answer(tmp_path, payload)
    for n in range(len(payload) + 1):
        assert sniff_first_record_timestamp(payload[:n], complete=False) in (
            None,
            local,
        )


def _probe(
    ref: RemoteRef, filename: str = "t.jsonl", *, is_dir: bool = False
) -> ProbeResult:
    return ProbeResult(
        ref=ref, pinned_ref=ref.display, data_relpath=None, data_is_dir=is_dir,
        filename=filename, manifest=None, manifest_sha256=None, needs_auth=False,
    )  # fmt: skip


def _cache(ref: RemoteRef, data: bytes, filename: str = "t.jsonl") -> ResolvedSource:
    local = entry_dir(ref) / filename
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(data)
    source = ResolvedSource(
        probe=_probe(ref, filename),
        local_path=local,
        files=[RemoteFile(path=filename, sha256="0" * 64, size_bytes=len(data))],
    )
    write_index(source)
    return source


class _FakeResolver:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.reads: list[int] = []

    async def read_prefix(self, probe: ProbeResult, nbytes: int) -> tuple[bytes, bool]:
        self.reads.append(nbytes)
        return self.data[:nbytes], nbytes >= len(self.data)


def _no_resolver(scheme: str) -> object:
    raise AssertionError(f"resolver for {scheme} must not be constructed")


@pytest.mark.asyncio
async def test_offline_reads_cached_copy_without_any_resolver(monkeypatch) -> None:
    ref = RemoteRef(ref="https://example.com/t.jsonl").with_version("etag-1")
    _cache(ref, TS)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(service, "get_resolver", _no_resolver)
    probe = await service.probe_ref(RemoteRef(ref="https://example.com/t.jsonl"))
    assert await remote_first_record_has_timestamp(probe) is True


@pytest.mark.asyncio
async def test_first_record_is_decided_from_the_smallest_prefix(monkeypatch) -> None:
    fake = _FakeResolver(TS * 100_000)
    monkeypatch.setattr(service, "get_resolver", lambda scheme: fake)
    monkeypatch.setattr(service, "materialize_probe", _no_resolver)
    ref = RemoteRef(ref="https://example.com/t.jsonl").with_version("etag-1")
    assert await remote_first_record_has_timestamp(_probe(ref)) is True
    assert fake.reads == [PREFIX_STEPS[0]]


@pytest.mark.asyncio
async def test_first_record_longer_than_every_prefix_reads_the_download(
    monkeypatch,
) -> None:
    data = b'{"pad": "' + b"x" * PREFIX_STEPS[-1] + b'", "timestamp": 1}\n'
    fake = _FakeResolver(data)
    monkeypatch.setattr(service, "get_resolver", lambda scheme: fake)
    ref = RemoteRef(ref="https://example.com/t.jsonl").with_version("etag-1")
    source = _cache(ref, data)

    async def materialize(probe: ProbeResult) -> ResolvedSource:
        return source

    monkeypatch.setattr(service, "materialize_probe", materialize)
    assert await remote_first_record_has_timestamp(_probe(ref)) is True
    assert fake.reads == list(PREFIX_STEPS)


@pytest.mark.asyncio
async def test_columnar_file_reads_the_download_not_a_prefix(monkeypatch) -> None:
    monkeypatch.setattr(service, "get_resolver", _no_resolver)
    ref = RemoteRef(ref="https://example.com/t.parquet").with_version("etag-1")
    source = _cache(ref, b"PAR1 not really parquet", filename="t.parquet")
    calls: list[ProbeResult] = []

    async def materialize(probe: ProbeResult) -> ResolvedSource:
        calls.append(probe)
        return source

    monkeypatch.setattr(service, "materialize_probe", materialize)
    assert await remote_first_record_has_timestamp(_probe(ref, "t.parquet")) is False
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_directory_has_no_first_record(monkeypatch) -> None:
    monkeypatch.setattr(service, "get_resolver", _no_resolver)
    monkeypatch.setattr(service, "materialize_probe", _no_resolver)
    ref = RemoteRef(ref="hf://org/repo/data").with_version("abc")
    assert (
        await remote_first_record_has_timestamp(_probe(ref, "data", is_dir=True))
        is False
    )
