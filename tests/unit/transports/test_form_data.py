# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for AioHttpTransport._build_form_data multipart serialization."""

import base64

import aiohttp
import pytest

from aiperf.transports.aiohttp_transport import AioHttpTransport


class TestBuildFormData:
    """Tests for the static _build_form_data helper."""

    def test_text_fields_preserved(self):
        """Plain text fields are added as string-valued form parts."""
        form = AioHttpTransport._build_form_data(
            {"prompt": "edit", "size": "1024x1024", "n": 2}
        )
        assert isinstance(form, aiohttp.FormData)

        fields = list(form._fields)
        assert {f[0]["name"] for f in fields} == {"prompt", "size", "n"}
        for opts, _headers, value in fields:
            assert "filename" not in opts
            assert isinstance(value, str)

    def test_bool_fields_lowercased(self):
        """Booleans are stringified as `true`/`false` so backends accept them."""
        form = AioHttpTransport._build_form_data(
            {"enable_teacache": True, "stream": False}
        )
        values = {f[0]["name"]: f[2] for f in form._fields}
        assert values == {"enable_teacache": "true", "stream": "false"}

    def test_none_values_skipped(self):
        """Fields with None are dropped (mirrors the JSON path's omission)."""
        form = AioHttpTransport._build_form_data({"prompt": "p", "seed": None})
        names = [f[0]["name"] for f in form._fields]
        assert names == ["prompt"]

    def test_b64_data_dict_decoded_to_file_upload(self):
        """Dict with `b64_data` key is base64-decoded and emitted as a file part."""
        png = b"\x89PNG\r\n\x1a\n\x00"
        b64 = base64.b64encode(png).decode("ascii")
        form = AioHttpTransport._build_form_data(
            {
                "prompt": "edit",
                "image": {
                    "b64_data": b64,
                    "filename": "ref.png",
                    "content_type": "image/png",
                },
            }
        )
        by_name = {f[0]["name"]: (f[0], f[1], f[2]) for f in form._fields}
        assert by_name["prompt"][2] == "edit"
        image_opts, image_headers, image_value = by_name["image"]
        assert image_value == png
        assert image_opts["filename"] == "ref.png"
        assert image_headers["Content-Type"] == "image/png"

    def test_b64_data_dict_defaults_filename_and_content_type(self):
        """Missing filename/content_type fall back to safe defaults."""
        b64 = base64.b64encode(b"\x00\x01\x02").decode("ascii")
        form = AioHttpTransport._build_form_data({"image": {"b64_data": b64}})
        opts, headers, value = form._fields[0]
        assert opts["name"] == "image"
        assert opts["filename"] == "image"
        assert headers["Content-Type"] == "application/octet-stream"
        assert value == b"\x00\x01\x02"

    def test_b64_data_invalid_raises(self):
        """Malformed base64 surfaces as a clear ValueError."""
        with pytest.raises(ValueError, match="not valid base64"):
            AioHttpTransport._build_form_data({"image": {"b64_data": "!!notb64!!"}})

    def test_dict_without_b64_data_falls_back_to_string(self):
        """A non-file-shaped dict is stringified, not treated as a file upload.

        Guards against accidentally turning JSON-shaped extras into file fields.
        """
        form = AioHttpTransport._build_form_data({"meta": {"foo": "bar"}})
        opts, _headers, value = form._fields[0]
        assert opts["name"] == "meta"
        assert "filename" not in opts
        assert isinstance(value, str)

    def test_text_only_payload_still_encodes_as_multipart(self):
        """Text-only FormData must encode as multipart/form-data, not urlencoded.

        Without ``default_to_multipart=True``, aiohttp treats text-only forms as
        ``application/x-www-form-urlencoded``, which breaks the contract for
        endpoints that declare ``requires_form_data: true`` (e.g., image_edit
        with a ``url`` field instead of an inline image).
        """
        form = AioHttpTransport._build_form_data(
            {"prompt": "edit", "url": "https://example.com/source.png"}
        )
        payload = form()
        content_type = payload.headers.get("Content-Type", "")
        assert content_type.startswith("multipart/form-data"), (
            f"text-only FormData must be multipart, got {content_type!r}"
        )
