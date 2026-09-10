# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for operator UI router helpers."""

from __future__ import annotations

from pathlib import Path

from tests.ui.node_utils import run_node

ROUTER_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "aiperf"
    / "operator"
    / "ui"
    / "lib"
    / "router.js"
)


def test_replace_hash_updates_url_without_push_state() -> None:
    script = f"""
        global.window = {{
          location: {{ hash: '#/' }},
          addEventListener() {{}},
        }};
        import {{ readFileSync }} from 'node:fs';
        let source = readFileSync({str(ROUTER_PATH)!r}, 'utf8');
        source = source.replace(
          "import {{ signal }} from '@preact/signals';",
          "const signal = (value) => ({{ value }});",
        );
        const routerModuleUrl = 'data:text/javascript;base64,' + Buffer.from(source).toString('base64');
        const {{ replaceHash }} = await import(routerModuleUrl);
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
