# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for transport-field serialization on ModelEndpointInfo.

When a URL without a scheme is passed to aiperf (e.g. ``127.0.0.1:8000``),
``normalize_http_url`` prepends ``http://`` before validation.  The run
completes correctly, but if the auto-detected transport is stored as a plain
``str`` instead of a ``TransportType`` enum value, Pydantic emits a
``PydanticSerializationUnexpectedValue`` warning on ``model_dump()`` because
the field is declared ``TransportType | None``.

These tests pin the fix: after transport auto-detection, ``model_dump()`` must
be warning-free.
"""

from __future__ import annotations

import warnings

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.common.enums import ModelSelectionStrategy
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.config.endpoint import EndpointConfig, EndpointDefaults
from aiperf.plugin.enums import EndpointType, TransportType


def test_endpoint_config_normalizes_schemeless_urls() -> None:
    endpoint = EndpointConfig(urls=["localhost:8000"])
    assert endpoint.urls == ["http://localhost:8000"]


def test_endpoint_config_timeout_uses_endpoint_default() -> None:
    endpoint = EndpointConfig(urls=["http://localhost:8000"])
    assert endpoint.timeout == EndpointDefaults.TIMEOUT


def test_endpoint_config_dynamo_session_control_defaults_disabled() -> None:
    endpoint = EndpointConfig(urls=["http://localhost:8000"])
    assert endpoint.use_dynamo_conv_aware_routing is False
    assert endpoint.use_legacy_dynamo_session_control is False
    assert (
        endpoint.dynamo_session_timeout_seconds
        == EndpointDefaults.DYNAMO_SESSION_TIMEOUT_SECONDS
    )


def test_endpoint_config_legacy_dynamo_without_routing_raises_error() -> None:
    """Legacy session control only selects the wire contract for the routing
    feature, so enabling it alone is incoherent and must be rejected."""
    with pytest.raises(
        ValidationError, match="use-legacy-dynamo-session-control has no effect"
    ):
        EndpointConfig(
            urls=["http://localhost:8000"],
            use_legacy_dynamo_session_control=True,
        )


@pytest.mark.parametrize(
    ("use_routing", "use_legacy"),
    [
        param(True, False, id="modern-routing-only"),
        param(True, True, id="legacy-with-routing"),
        param(False, False, id="both-disabled"),
    ],
)  # fmt: skip
def test_endpoint_config_dynamo_session_control_coherent_combos_accepted(
    use_routing: bool, use_legacy: bool
) -> None:
    endpoint = EndpointConfig(
        urls=["http://localhost:8000"],
        use_dynamo_conv_aware_routing=use_routing,
        use_legacy_dynamo_session_control=use_legacy,
    )
    assert endpoint.use_dynamo_conv_aware_routing is use_routing
    assert endpoint.use_legacy_dynamo_session_control is use_legacy


def test_uuid_and_strip_rejects_non_chat_endpoint() -> None:
    with pytest.raises(ValueError, match="requires endpoint type 'chat'"):
        EndpointConfig(
            urls=["http://localhost:8000"],
            type=EndpointType.COMPLETIONS,
            uuid_and_strip=True,
        )


def _make_model_endpoint(
    base_url: str, transport: TransportType | None = None
) -> ModelEndpointInfo:
    return ModelEndpointInfo(
        models=ModelListInfo(
            models=[ModelInfo(name="test-model")],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
        ),
        endpoint=EndpointInfo(
            type=EndpointType.CHAT,
            base_urls=[base_url],
        ),
        transport=transport,
    )


@pytest.mark.parametrize(
    ("base_url", "transport"),
    [
        param("http://127.0.0.1:8000", TransportType("http"), id="bare-url-auto-transport"),
        param("http://127.0.0.1:8000", TransportType("http"), id="explicit-http-url-auto-transport"),
        param("https://api.example.com", TransportType("http"), id="https-url-auto-transport"),
        param("http://127.0.0.1:8000", None, id="no-transport-none"),
    ],
)  # fmt: skip
def test_model_endpoint_info_model_dump_no_pydantic_serialization_warning(
    base_url: str, transport: TransportType | None
) -> None:
    """model_dump() must not emit PydanticSerializationUnexpectedValue.

    Storing the bare plugin-name string (e.g. 'http') instead of
    TransportType('http') on the transport field triggers the warning because
    Pydantic can't serialize a plain str as the declared TransportType enum.
    """
    endpoint = _make_model_endpoint(base_url, transport)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        endpoint.model_dump()

    pydantic_warnings = [
        w for w in captured if "PydanticSerializationUnexpectedValue" in str(w.message)
    ]
    assert not pydantic_warnings, (
        f"Unexpected Pydantic serialization warnings for url={base_url!r} "
        f"transport={transport!r}: {pydantic_warnings}"
    )


@pytest.mark.parametrize(
    "base_url",
    [
        param("http://127.0.0.1:8000", id="http-ip-port"),
        param("http://localhost:8000", id="http-localhost"),
        param("https://api.example.com", id="https-domain"),
    ],
)  # fmt: skip
def test_model_endpoint_info_bare_string_transport_triggers_warning(
    base_url: str,
) -> None:
    """Assigning a bare str to transport post-validation does trigger the warning.

    This is the regression canary: if this test starts failing (no warning),
    Pydantic changed its coercion behavior and the fix may no longer be needed.
    """
    endpoint = _make_model_endpoint(base_url)
    # Deliberately assign a bare string post-validation (the pre-fix behavior)
    endpoint.transport = "http"  # type: ignore[assignment]

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        endpoint.model_dump()

    pydantic_warnings = [
        w for w in captured if "PydanticSerializationUnexpectedValue" in str(w.message)
    ]
    assert pydantic_warnings, (
        "Expected a PydanticSerializationUnexpectedValue warning when transport is a bare "
        "string, but none was emitted. Pydantic may have changed its behavior."
    )
