# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for operator UI path resolution under a reverse-proxy path prefix."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests.ui.node_utils import run_node

_UI_ROOT = Path(__file__).resolve().parents[2] / "src" / "aiperf" / "operator" / "ui"
_UI_LIB = _UI_ROOT / "lib"
_BASE_PATH = (_UI_LIB / "base-path.js").as_uri()
_JOB_WS = (_UI_LIB / "job-ws.js").as_uri()
_TOP_NAV_PATH = _UI_ROOT / "components" / "top-nav.js"

# A quote or backtick directly followed by /api/ or /dashboard/ starts a
# root-absolute URL that would drop a reverse-proxy prefix.
_ROOT_ABSOLUTE_URL_RE = re.compile(r"['\"`]/(api|dashboard)/")


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


def test_ui_sources_build_no_root_absolute_api_or_dashboard_urls() -> None:
    """Every API and dashboard URL must go through ``API_BASE`` or ``appPath``."""
    violations = []
    for path in sorted(_UI_ROOT.rglob("*.js")):
        if "vendor" in path.relative_to(_UI_ROOT).parts:
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if line.lstrip().startswith(("//", "*", "/*")):
                continue
            if _ROOT_ABSOLUTE_URL_RE.search(line):
                rel = path.relative_to(_UI_ROOT)
                violations.append(f"{rel}:{lineno}: {line.strip()}")

    assert violations == []


def _plots_link(base_uri: str, features: dict[str, bool] | None) -> str | None:
    source = _TOP_NAV_PATH.read_text()
    helpers = source[
        source.index("const PRIMARY_GROUP") : source.index("function isActive")
    ]
    script = f"""
        globalThis.document = {{ baseURI: {json.dumps(base_uri)} }};
        const {{ appPath }} = await import({_BASE_PATH!r});
        const buildNavGroups = eval({json.dumps(helpers + "buildNavGroups")});
        const items = buildNavGroups({json.dumps(features)}).flatMap((g) => g.items);
        const plots = items.find((item) => item.testId === 'nav-link-plots');
        console.log(JSON.stringify(plots ? plots.path : null));
    """
    return json.loads(run_node(script))


def test_plots_link_is_offered_at_origin_root_when_dashboard_enabled() -> None:
    root = "http://operator.example.test/"
    assert _plots_link(root, {"dashboard_enabled": True}) == "/dashboard/"
    assert _plots_link(root, {"dashboard_enabled": False}) is None
    assert _plots_link(root, None) is None
    assert (
        _plots_link(f"{root}index.html", {"dashboard_enabled": True}) == "/dashboard/"
    )


def test_plots_link_is_hidden_under_reverse_proxy_prefix() -> None:
    # The Dash app only loads from the origin-root /dashboard/ prefix.
    prefixed = "https://gw.example.test/aiperf/"
    assert _plots_link(prefixed, {"dashboard_enabled": True}) is None
