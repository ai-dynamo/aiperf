# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport-side tests for per-row endpoint routing.

A turn carrying ``endpoint_type`` must reach that endpoint's registered path on
the same base URL, and must not inherit request shaping (streaming Accept
header, custom path) configured for the run-level endpoint.
"""

from __future__ import annotations

from aiperf.common.models import Turn
from aiperf.transports.aiohttp_transport import AioHttpTransport
from tests.unit.transports.conftest import create_model_endpoint_info
from tests.unit.transports.test_aiohttp_transport import create_request_info


def _request_info(model_endpoint, endpoint_type: str | None):
    """Build a RequestInfo whose dispatch turn optionally overrides the endpoint."""
    request_info = create_request_info(model_endpoint)
    request_info.turns = [Turn(endpoint_type=endpoint_type)]
    return request_info


class TestRoutedUrl:
    def test_override_uses_target_endpoint_path(self):
        """An overridden row hits the override's registered path, same base URL."""
        model_endpoint = create_model_endpoint_info(
            base_url="http://localhost:8000", custom_endpoint=None
        )
        transport = AioHttpTransport(model_endpoint=model_endpoint)

        assert transport.get_url(_request_info(model_endpoint, "embeddings")) == (
            "http://localhost:8000/v1/embeddings"
        )

    def test_unrouted_turn_uses_run_level_path(self):
        """Rows without an override are unaffected."""
        model_endpoint = create_model_endpoint_info(
            base_url="http://localhost:8000", custom_endpoint=None
        )
        transport = AioHttpTransport(model_endpoint=model_endpoint)

        assert transport.get_url(_request_info(model_endpoint, None)) == (
            "http://localhost:8000/v1/chat/completions"
        )

    def test_custom_endpoint_not_inherited_by_override(self):
        """A custom path configured for the run-level endpoint must not leak.

        --custom-endpoint describes where the *configured* endpoint lives; a row
        that explicitly names a different endpoint would otherwise be silently
        sent to the wrong path.
        """
        model_endpoint = create_model_endpoint_info(
            base_url="http://localhost:8000", custom_endpoint="/internal/chat/v2"
        )
        transport = AioHttpTransport(model_endpoint=model_endpoint)

        assert transport.get_url(_request_info(model_endpoint, "embeddings")) == (
            "http://localhost:8000/v1/embeddings"
        )
        assert transport.get_url(_request_info(model_endpoint, None)) == (
            "http://localhost:8000/internal/chat/v2"
        )

    def test_override_preserves_base_path_and_query(self):
        """Path-joining rules (dedup, query preservation) still apply."""
        model_endpoint = create_model_endpoint_info(
            base_url="http://h/v1?key=abc", custom_endpoint=None
        )
        transport = AioHttpTransport(model_endpoint=model_endpoint)

        assert transport.get_url(_request_info(model_endpoint, "embeddings")) == (
            "http://h/v1/embeddings?key=abc"
        )


class TestRoutedHeaders:
    def test_non_streaming_endpoint_in_streaming_run_gets_json_accept(self):
        """Embeddings has no streaming variant; it must not advertise SSE.

        Without this gate a --streaming run would send Accept: text/event-stream
        for a plain JSON response.
        """
        model_endpoint = create_model_endpoint_info(
            base_url="http://localhost:8000", custom_endpoint=None, streaming=True
        )
        transport = AioHttpTransport(model_endpoint=model_endpoint)

        headers = transport.get_transport_headers(
            _request_info(model_endpoint, "embeddings")
        )
        assert headers["Accept"] == "application/json"

    def test_streaming_preserved_for_run_level_endpoint(self):
        """Unrouted rows in a streaming run still ask for SSE."""
        model_endpoint = create_model_endpoint_info(
            base_url="http://localhost:8000", custom_endpoint=None, streaming=True
        )
        transport = AioHttpTransport(model_endpoint=model_endpoint)

        headers = transport.get_transport_headers(_request_info(model_endpoint, None))
        assert headers["Accept"] == "text/event-stream"

    def test_non_streaming_run_unaffected(self):
        model_endpoint = create_model_endpoint_info(
            base_url="http://localhost:8000", custom_endpoint=None, streaming=False
        )
        transport = AioHttpTransport(model_endpoint=model_endpoint)

        for endpoint_type in (None, "embeddings"):
            headers = transport.get_transport_headers(
                _request_info(model_endpoint, endpoint_type)
            )
            assert headers["Accept"] == "application/json"
