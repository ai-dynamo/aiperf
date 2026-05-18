# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for operator UI router helpers."""

from __future__ import annotations

from pathlib import Path

from tests.unit.ui.node_utils import run_node

ROUTER_HELPERS_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "aiperf"
    / "operator"
    / "ui"
    / "lib"
    / "router-helpers.js"
)


def test_replace_hash_updates_url_without_push_state() -> None:
    script = f"""
        import {{ replaceHash }} from {ROUTER_HELPERS_PATH.as_uri()!r};
        const calls = [];
        const win = {{
          location: {{ hash: '#/jobs/ns/job' }},
          history: {{
            replaceState(state, title, url) {{ calls.push(url); win.location.hash = url; }},
          }},
        }};
        replaceHash(win, '/jobs/ns/job/runs/123');
        console.log(win.location.hash + '|' + calls.join(','));
    """

    assert run_node(script) == "#/jobs/ns/job/runs/123|#/jobs/ns/job/runs/123"
