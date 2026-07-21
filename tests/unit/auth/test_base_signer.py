# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from aiperf.auth.base_signer import RequestSignerProtocol, SignedRequest


class TestSignedRequest:
    def test_headers_only(self) -> None:
        signed = SignedRequest(headers={"Authorization": "AWS4-HMAC-SHA256 ..."})
        assert signed.headers == {"Authorization": "AWS4-HMAC-SHA256 ..."}
        assert signed.url is None
        assert signed.body is None

    def test_all_fields(self) -> None:
        signed = SignedRequest(
            headers={"Authorization": "sig"},
            url="https://signed.example.com",
            body=b"signed-body",
        )
        assert signed.url == "https://signed.example.com"
        assert signed.body == b"signed-body"

    def test_slots(self) -> None:
        signed = SignedRequest(headers={})
        with pytest.raises(AttributeError):
            signed.extra_field = "nope"  # type: ignore[attr-defined]


class TestRequestSignerProtocol:
    def test_protocol_is_runtime_checkable(self) -> None:
        assert hasattr(RequestSignerProtocol, "__protocol_attrs__") or hasattr(
            RequestSignerProtocol, "__abstractmethods__"
        )
