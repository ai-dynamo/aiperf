# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Accept-Encoding negotiation when serving zstd-compressed artifacts.

Substring matching on the header ignores RFC 9110 quality values, so a client
sending ``gzip, zstd;q=0`` -- an explicit refusal of zstd -- was still handed a
zstd body it had said it could not decode.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import zstandard
from pytest import param
from starlette.datastructures import Headers

from aiperf.operator.routers.results_files_io import _accepts_encoding, _serve_zst_file


class _FakeRequest:
    """Minimal stand-in exposing the only attribute ``_serve_zst_file`` reads."""

    def __init__(self, accept_encoding: str | None) -> None:
        raw = {} if accept_encoding is None else {"accept-encoding": accept_encoding}
        self.headers = Headers(raw)


@pytest.mark.parametrize(
    "header,encoding,expected",
    [
        param("zstd", "zstd", True, id="plain-token"),
        param("gzip, zstd", "zstd", True, id="listed-no-q"),
        param("gzip, zstd;q=0", "zstd", False, id="q-zero-refusal"),
        param("gzip, zstd;q=0", "gzip", True, id="sibling-still-accepted"),
        param("zstd;q=0.001", "zstd", True, id="tiny-q-is-acceptance"),
        param("ZSTD;Q=0", "zstd", False, id="uppercase-q-zero-refusal"),
        param("gzip, zstd ; q = 0", "zstd", False, id="whitespace-around-q"),
        param("*", "zstd", True, id="wildcard-accepts"),
        param("*;q=0", "zstd", False, id="wildcard-refuses"),
        param("*;q=0, zstd", "zstd", True, id="explicit-beats-wildcard-refusal"),
        param("*, zstd;q=0", "zstd", False, id="explicit-refusal-beats-wildcard"),
        param("br", "zstd", False, id="unlisted-no-wildcard"),
        param("", "zstd", False, id="empty-header"),
        param(None, "zstd", False, id="missing-header"),
    ],
)  # fmt: skip
def test_accepts_encoding_quality_values_respected(
    header: str | None, encoding: str, expected: bool
) -> None:
    assert _accepts_encoding(header, encoding) is expected


def _write_zst(tmp_path: Path, payload: bytes) -> Path:
    zst_path = tmp_path / "profile_export_aiperf.json.zst"
    zst_path.write_bytes(zstandard.ZstdCompressor().compress(payload))
    return zst_path


@pytest.mark.parametrize(
    "header,expected_encoding",
    [
        param("zstd", "zstd", id="zstd-accepted"),
        param("gzip, zstd", "zstd", id="zstd-preferred-over-gzip"),
        param("gzip, zstd;q=0", "gzip", id="zstd-refused-falls-back-to-gzip"),
        param("zstd;q=0, gzip;q=0", None, id="both-refused-falls-back-to-identity"),
        param("identity", None, id="identity-only"),
        param(None, None, id="no-header"),
    ],
)  # fmt: skip
def test_serve_zst_file_negotiation_honors_refusals(
    tmp_path: Path, header: str | None, expected_encoding: str | None
) -> None:
    zst_path = _write_zst(tmp_path, b'{"ok": true}')
    response = _serve_zst_file(
        _FakeRequest(header),  # type: ignore[arg-type]
        zst_path,
        "profile_export_aiperf.json",
    )
    assert response.headers.get("content-encoding") == expected_encoding


@pytest.mark.asyncio
async def test_serve_zst_file_zstd_refused_body_is_decodable_gzip(
    tmp_path: Path,
) -> None:
    """The refusal path must actually emit gzip bytes, not relabeled zstd."""
    import zlib

    payload = b'{"metric": "value"}' * 64
    zst_path = _write_zst(tmp_path, payload)
    response = _serve_zst_file(
        _FakeRequest("gzip, zstd;q=0"),  # type: ignore[arg-type]
        zst_path,
        "profile_export_aiperf.json",
    )

    chunks = [chunk async for chunk in response.body_iterator]
    body = b"".join(chunks)
    assert isinstance(body, bytes)
    assert zlib.decompress(body, wbits=31) == payload
