# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from aiperf.auth.base_signer import RequestSignerProtocol, SignedRequest
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType
from tests.unit.transports.conftest import create_model_endpoint_info


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
    def test_registered_signers_satisfy_protocol(self) -> None:
        """Every plugin in the request_signer category must actually satisfy
        the protocol the category declares."""
        entries = plugins.list_entries(PluginType.REQUEST_SIGNER)
        assert entries, "no request_signer plugins registered"
        for entry in entries:
            signer = plugins.get_class(PluginType.REQUEST_SIGNER, entry.name)(
                model_endpoint=create_model_endpoint_info()
            )
            assert isinstance(signer, RequestSignerProtocol), (
                f"{entry.name} does not satisfy RequestSignerProtocol"
            )
