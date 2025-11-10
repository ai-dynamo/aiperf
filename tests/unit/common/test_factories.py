# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytest import param

from aiperf.common.enums import TransportType
from aiperf.common.factories import TransportFactory


class TestTransportFactoryDetectFromUrl:
    """Test suite for TransportFactory.detect_from_url method."""

    @pytest.mark.parametrize(
        "url,expected_transport",
        [
            param("http://api.example.com:8000", TransportType.HTTP, id="http_with_port"),
            param("https://api.example.com:8443", TransportType.HTTP, id="https_with_port"),
            param("http://localhost:8000", TransportType.HTTP, id="http_localhost"),
            param("http://127.0.0.1:8000", TransportType.HTTP, id="http_localhost_ip"),
            param("http://[::1]:8000", TransportType.HTTP, id="http_ipv6"),
            param("http://api.example.com", TransportType.HTTP, id="http_no_port"),
            param("https://api.example.com", TransportType.HTTP, id="https_no_port"),
            param("http://localhost:8000/api/v1/chat", TransportType.HTTP, id="with_path"),
            param("http://api.example.com?model=gpt-4&key=value", TransportType.HTTP, id="with_query"),
            param("http://user:password@api.example.com:8000", TransportType.HTTP, id="with_credentials"),
            param("http://api.example.com#section", TransportType.HTTP, id="with_fragment"),
            param("http://api.example.com/path/with%20spaces", TransportType.HTTP, id="with_encoded_spaces"),
            param("https://api.openai.com/v1/chat/completions", TransportType.HTTP, id="openai_api"),
        ],
    )  # fmt: skip
    def test_http_https_detection(self, url, expected_transport):
        """Test detection of HTTP/HTTPS URLs with various components."""
        result = TransportFactory.detect_from_url(url)
        assert result is not None
        assert result == expected_transport

    @pytest.mark.parametrize(
        "url",
        [
            param("HTTP://api.example.com", id="uppercase_scheme"),
            param("Http://api.example.com", id="mixed_case_scheme"),
            param("hTTp://api.example.com", id="random_case_scheme"),
        ],
    )
    def test_scheme_case_insensitive(self, url):
        """Test that scheme detection is case-insensitive."""
        result = TransportFactory.detect_from_url(url)
        assert result == TransportType.HTTP

    @pytest.mark.parametrize(
        "url",
        [
            param("", id="empty_string"),
            param("http://", id="scheme_only"),
            param("api.example.com:8000", id="no_scheme_with_port"),
            param("api.example.com", id="no_scheme_no_port"),
            param("localhost", id="localhost_no_scheme"),
        ],
    )
    def test_edge_cases_default_to_http_or_none(self, url):
        """Test edge cases return HTTP or None."""
        result = TransportFactory.detect_from_url(url)
        assert result is None or result == TransportType.HTTP

    @pytest.mark.parametrize(
        "url",
        [
            param("unknown://api.example.com", id="unknown_scheme"),
            param("ftp://files.example.com", id="ftp_scheme"),
            param("grpc://localhost:50051", id="grpc_scheme"),
            param("/path/to/file.sock", id="file_path"),
        ],
    )
    def test_unregistered_schemes_return_none(self, url):
        """Test that unregistered schemes return None."""
        result = TransportFactory.detect_from_url(url)
        assert result is None
