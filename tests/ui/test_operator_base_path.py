# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for operator UI path resolution under a reverse-proxy path prefix."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.ui.node_utils import run_node

_UI_LIB = (
    Path(__file__).resolve().parents[2] / "src" / "aiperf" / "operator" / "ui" / "lib"
)
_BASE_PATH = (_UI_LIB / "base-path.js").as_uri()
_JOB_WS = (_UI_LIB / "job-ws.js").as_uri()


def _resolve(base_uri: str) -> dict[str, str]:
    script = f"""
        globalThis.document = {{ baseURI: {json.dumps(base_uri)} }};
        const {{ appPath, API_BASE }} = await import({_BASE_PATH!r});
        console.log(JSON.stringify({{
          api: API_BASE,
          dashboard: appPath('dashboard/'),
          leadingSlash: appPath('/api/v1/jobs'),
        }}));
    """
    return json.loads(run_node(script))


@pytest.mark.parametrize(
    "base_uri",
    [
        "http://operator.example.test/",
        "http://operator.example.test/index.html",
        "http://operator.example.test/?x=1#/jobs/ns/name",
    ],
)
def test_app_paths_at_origin_root_match_legacy_absolute_paths(base_uri: str) -> None:
    assert _resolve(base_uri) == {
        "api": "/api/v1",
        "dashboard": "/dashboard/",
        "leadingSlash": "/api/v1/jobs",
    }


@pytest.mark.parametrize(
    "base_uri",
    [
        "https://gw.example.test/aiperf/",
        "https://gw.example.test/aiperf/index.html",
        "https://gw.example.test/aiperf/#/jobs/ns/name",
    ],
)
def test_app_paths_keep_reverse_proxy_prefix(base_uri: str) -> None:
    assert _resolve(base_uri) == {
        "api": "/aiperf/api/v1",
        "dashboard": "/aiperf/dashboard/",
        "leadingSlash": "/aiperf/api/v1/jobs",
    }


def test_job_ws_url_keeps_reverse_proxy_prefix() -> None:
    script = f"""
        const sockets = [];
        globalThis.WebSocket = class {{
          constructor(url) {{ this.url = url; sockets.push(this); }}
          send() {{}}
          close() {{}}
        }};
        globalThis.window = {{ location: {{ protocol: 'https:', host: 'gw.example.test' }} }};
        globalThis.document = {{ baseURI: 'https://gw.example.test/aiperf/' }};
        const {{ openJobWs }} = await import({_JOB_WS!r});
        const handle = openJobWs('ns', 'job', () => {{}});
        console.log(sockets[0].url);
        handle.close();
    """

    assert run_node(script) == "wss://gw.example.test/aiperf/api/v1/jobs/ns/job/ws"
