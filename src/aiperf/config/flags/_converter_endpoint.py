# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLIConfig -> AIPerfConfig converter: endpoint + models sections.

Reads from the flattened endpoint fields on ``CLIConfig`` (with ``headers``
and ``extra_inputs`` living as top-level fields on ``CLIConfig``) to produce
the dict shape consumed by ``AIPerfConfig``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiperf.config.flags._section_fields import ENDPOINT_FIELDS

if TYPE_CHECKING:
    from aiperf.config.flags.cli_config import CLIConfig


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


# Map (CLIConfig endpoint field name) -> (AIPerfConfig endpoint key).
_ENDPOINT_FIELD_MAP: dict[str, str] = {
    "url_selection_strategy": "url_strategy",
    "endpoint_type": "type",
    "streaming": "streaming",
    "custom_endpoint": "path",
    "api_key": "api_key",
    "timeout_seconds": "timeout",
    "wait_for_model_timeout": "wait_for_model_timeout",
    "wait_for_model_mode": "wait_for_model_mode",
    "wait_for_model_interval": "wait_for_model_interval",
    "transport": "transport",
    "use_legacy_max_tokens": "use_legacy_max_tokens",
    "use_server_token_count": "use_server_token_count",
    "connection_reuse_strategy": "connection_reuse",
    "download_video_content": "download_video_content",
    "request_content_type": "request_content_type",
    "session_header": "session_header",
}


def build_endpoint(cli: CLIConfig) -> dict[str, Any]:
    """Build the AIPerfConfig ``endpoint`` section from a CLIConfig.

    Reads flattened endpoint fields directly off ``cli``. ``urls`` always
    populates; other endpoint fields only flow through when explicitly set
    by the user (per ``cli.model_fields_set & ENDPOINT_FIELDS``).
    ``headers`` / ``extra`` live as top-level fields on CLIConfig and
    flow through to the endpoint dict.
    """
    endpoint: dict[str, Any] = {"urls": [_url(u) for u in cli.urls]}
    ep_set = cli.model_fields_set & ENDPOINT_FIELDS
    for field, key in _ENDPOINT_FIELD_MAP.items():
        if field in ep_set:
            endpoint[key] = getattr(cli, field)

    cli_set = cli.model_fields_set
    if "headers" in cli_set and cli.headers:
        endpoint["headers"] = dict(cli.headers)
    if "extra_inputs" in cli_set and cli.extra_inputs:
        extra = dict(cli.extra_inputs)
        _endpoint_template_from_extra(endpoint, extra)
        endpoint["extra"] = extra

    _endpoint_template_fallback(endpoint)
    return endpoint


def build_models(cli: CLIConfig) -> dict[str, Any]:
    """Build the AIPerfConfig ``models`` section from a CLIConfig.

    ``model_names`` and ``model_selection_strategy`` both live on the
    flattened endpoint section of CLIConfig.
    """
    models: dict[str, Any] = {"items": [{"name": n} for n in cli.model_names]}
    if "model_selection_strategy" in cli.model_fields_set:
        models["strategy"] = cli.model_selection_strategy
    return models
