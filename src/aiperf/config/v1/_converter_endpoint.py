# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""v1->v2 converter: endpoint + models sections.

Reads from ``UserConfig.endpoint`` (and ``UserConfig.input`` for headers/extras)
to produce the dict shape consumed by ``AIPerfConfig``. Mirrors the field
semantics of the legacy ``aiperf.config._cli_sections.build_endpoint`` /
``build_models`` but sources values from v1 nested DTOs instead of the flat
CLIModel.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiperf.config.v1.user_config import UserConfig


def _url(item: str) -> str:
    return item if "://" in item else f"http://{item}"


def _endpoint_template_from_extra(
    endpoint: dict[str, Any], extra: dict[str, Any]
) -> None:
    payload_template = extra.pop("payload_template", None)
    if payload_template is None:
        return
    path = Path(payload_template)
    body = path.read_text() if path.is_file() else payload_template
    endpoint["template"] = {
        "body": body,
        "response_field": extra.pop("response_field", "text"),
    }


def _endpoint_template_fallback(endpoint: dict[str, Any]) -> None:
    from aiperf.plugin.enums import EndpointType

    if endpoint.get("type") != EndpointType.TEMPLATE or "template" in endpoint:
        return
    extra_raw = endpoint.get("extra")
    if not extra_raw:
        return
    ex = dict(extra_raw) if isinstance(extra_raw, list) else extra_raw
    ts = ex.get("payload_template")
    if ts is None:
        return
    tp = Path(ts)
    endpoint["template"] = {"body": tp.read_text() if tp.is_file() else ts}


# Map (v1 EndpointConfig field name) -> (v2 AIPerfConfig endpoint key).
# Source field names come from src/aiperf/config/v1/_endpoint.py; some differ
# from the legacy CLIModel names (e.g. ``timeout_seconds`` was
# ``request_timeout_seconds`` on CLIModel, ``transport`` was ``transport_type``).
_ENDPOINT_FIELD_MAP: dict[str, str] = {
    "url_selection_strategy": "url_strategy",
    "type": "type",
    "streaming": "streaming",
    "custom_endpoint": "path",
    "api_key": "api_key",
    "timeout_seconds": "timeout",
    "ready_check_timeout": "ready_check_timeout",
    "transport": "transport",
    "use_legacy_max_tokens": "use_legacy_max_tokens",
    "use_server_token_count": "use_server_token_count",
    "connection_reuse_strategy": "connection_reuse",
}


def build_endpoint(user: UserConfig) -> dict[str, Any]:
    """Build the AIPerfConfig ``endpoint`` section from a v1 UserConfig.

    Reads ``user.endpoint`` for endpoint-shaped fields and ``user.input`` for
    ``headers`` / ``extra`` (both live on InputConfig in v1, not EndpointConfig).
    Only fields explicitly set by the user (per nested model's
    ``model_fields_set``) flow through, except ``urls`` which always populates.
    """
    ep = user.endpoint
    inp = user.input
    if ep is None:
        # Defensive: caller should typically pre-populate endpoint, but keep
        # this graceful (yields a minimal dict downstream layers will reject).
        return {"urls": []}

    endpoint: dict[str, Any] = {"urls": [_url(u) for u in ep.urls]}
    ep_set = ep.model_fields_set
    for field, key in _ENDPOINT_FIELD_MAP.items():
        if field in ep_set:
            endpoint[key] = getattr(ep, field)

    if inp is not None:
        inp_set = inp.model_fields_set
        if "headers" in inp_set and inp.headers:
            endpoint["headers"] = dict(inp.headers)
        if "extra" in inp_set and inp.extra:
            extra = dict(inp.extra)
            _endpoint_template_from_extra(endpoint, extra)
            endpoint["extra"] = extra

    _endpoint_template_fallback(endpoint)
    return endpoint


def build_models(user: UserConfig) -> dict[str, Any]:
    """Build the AIPerfConfig ``models`` section from a v1 UserConfig.

    ``model_names`` and ``model_selection_strategy`` both live on
    EndpointConfig in v1 (mirroring origin/main).
    """
    ep = user.endpoint
    if ep is None:
        return {"items": []}
    models: dict[str, Any] = {"items": [{"name": n} for n in ep.model_names]}
    if "model_selection_strategy" in ep.model_fields_set:
        models["strategy"] = ep.model_selection_strategy
    return models
