# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.auth.base_signer import SignedRequest
from aiperf.transports.aiohttp_transport import AioHttpTransport
from tests.unit.transports.conftest import create_model_endpoint_info


class TestAioHttpTransportSignerCreation:
    def test_no_signer_when_auth_type_not_set(self) -> None:
        transport = AioHttpTransport(model_endpoint=create_model_endpoint_info())
        assert transport.request_signer is None

    @patch("aiperf.transports.base_transports.plugins.get_class")
    def test_creates_signer_when_auth_type_set(self, mock_get_class: MagicMock) -> None:
        mock_signer_cls = MagicMock()
        mock_signer_instance = MagicMock()
        mock_signer_cls.return_value = mock_signer_instance
        mock_get_class.return_value = mock_signer_cls

        transport = AioHttpTransport(
            model_endpoint=create_model_endpoint_info(
                auth_type="sigv4", aws_region="us-east-1", aws_service="sagemaker"
            )
        )

        assert transport.request_signer is mock_signer_instance
        mock_signer_cls.assert_called_once()

    @patch("aiperf.transports.base_transports.plugins.get_class")
    def test_signer_attached_as_child_lifecycle(
        self, mock_get_class: MagicMock
    ) -> None:
        mock_signer_cls = MagicMock()
        mock_signer_instance = MagicMock()
        mock_signer_cls.return_value = mock_signer_instance
        mock_get_class.return_value = mock_signer_cls

        transport = AioHttpTransport(
            model_endpoint=create_model_endpoint_info(
                auth_type="sigv4", aws_region="us-east-1", aws_service="sagemaker"
            )
        )

        assert mock_signer_instance in transport._children


class TestAioHttpTransportSignIfNeeded:
    @pytest.mark.asyncio
    async def test_no_signer_returns_unchanged(self) -> None:
        transport = AioHttpTransport(model_endpoint=create_model_endpoint_info())
        signed = await transport._sign_if_needed(
            "POST", "https://a.com", {"H": "V"}, b"body"
        )
        assert signed.url == "https://a.com"
        assert signed.headers == {"H": "V"}
        assert signed.body == b"body"

    @pytest.mark.asyncio
    async def test_with_signer_returns_signed_headers(self) -> None:
        transport = AioHttpTransport(model_endpoint=create_model_endpoint_info())
        mock_signer = AsyncMock()
        mock_signer.sign.return_value = SignedRequest(
            headers={"H": "V", "Authorization": "AWS4-HMAC-SHA256 ..."}
        )
        transport.request_signer = mock_signer

        signed = await transport._sign_if_needed(
            "POST", "https://a.com", {"H": "V"}, b"body"
        )

        assert signed.headers["Authorization"] == "AWS4-HMAC-SHA256 ..."
        mock_signer.sign.assert_called_once_with(
            "POST", "https://a.com", {"H": "V"}, b"body"
        )

    @pytest.mark.asyncio
    async def test_with_signer_url_override(self) -> None:
        transport = AioHttpTransport(model_endpoint=create_model_endpoint_info())
        mock_signer = AsyncMock()
        mock_signer.sign.return_value = SignedRequest(
            headers={"H": "V"},
            url="https://signed.com",
        )
        transport.request_signer = mock_signer

        signed = await transport._sign_if_needed(
            "GET", "https://original.com", {"H": "V"}, None
        )

        assert signed.url == "https://signed.com"

    @pytest.mark.asyncio
    async def test_with_signer_body_override(self) -> None:
        transport = AioHttpTransport(model_endpoint=create_model_endpoint_info())
        mock_signer = AsyncMock()
        mock_signer.sign.return_value = SignedRequest(
            headers={},
            body=b"signed-body",
        )
        transport.request_signer = mock_signer

        signed = await transport._sign_if_needed(
            "POST", "https://a.com", {}, b"original"
        )

        assert signed.body == b"signed-body"
