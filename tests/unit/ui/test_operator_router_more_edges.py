# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Additional edge-case tests for operator UI router behavior."""

from __future__ import annotations

import json
from pathlib import Path

from tests.unit.ui.node_utils import run_node

ROUTER_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "aiperf"
    / "operator"
    / "ui"
    / "lib"
    / "router.js"
)
ROUTER_HELPERS_PATH = ROUTER_PATH.with_name("router-helpers.js")


def router_import_script() -> str:
    return f"""
        import {{ readFileSync }} from 'node:fs';
        let source = readFileSync({str(ROUTER_PATH)!r}, 'utf8');
        source = source.replace(
          "import {{ signal }} from '@preact/signals';",
          "const signal = (value) => ({{ value }});",
        );
        source = source.replace(
          "import {{ normalizePath, replaceHash }} from './router-helpers.js';",
          "import {{ normalizePath, replaceHash }} from {ROUTER_HELPERS_PATH.as_uri()!r};",
        );
        const routerModuleUrl = 'data:text/javascript;base64,' + Buffer.from(source).toString('base64');
        const router = await import(routerModuleUrl);
    """


def test_hash_url_encodes_query_values_and_omits_empty_values() -> None:
    script = f"""
        global.window = {{
          location: {{ hash: '#/' }},
          addEventListener() {{}},
        }};
        {router_import_script()}
        const {{ hashUrl }} = router;
        console.log(hashUrl('/jobs/ns/job', {{
          ns: 'team/a',
          name: 'job 1',
          phase: '',
          none: null,
          missing: undefined,
          filter: 'a&b=c',
        }}));
    """

    assert run_node(script) == "#/jobs/ns/job?ns=team%2Fa&name=job%201&filter=a%26b%3Dc"


def test_set_query_preserves_existing_values_and_removes_blank_updates() -> None:
    script = f"""
        global.window = {{
          location: {{ hash: '#/jobs?ns=default&phase=Running&search=old%20job&page=2' }},
          addEventListener() {{}},
        }};
        {router_import_script()}
        const {{ setQuery }} = router;
        setQuery({{ phase: '', search: 'new/job?', page: null, sort: 'created desc' }});
        console.log(window.location.hash);
    """

    assert (
        run_node(script) == "#/jobs?ns=default&search=new%2Fjob%3F&sort=created%20desc"
    )


def test_set_query_noops_without_reordering_when_update_is_unchanged() -> None:
    script = f"""
        global.window = {{
          location: {{ hash: '#/jobs?b=2&a=1' }},
          addEventListener() {{}},
        }};
        {router_import_script()}
        const {{ setQuery }} = router;
        setQuery({{ a: '1' }});
        console.log(window.location.hash);
    """

    assert run_node(script) == "#/jobs?b=2&a=1"


def test_match_and_build_route_round_trip_encoded_slashes() -> None:
    script = f"""
        global.window = {{
          location: {{ hash: '#/' }},
          addEventListener() {{}},
        }};
        {router_import_script()}
        const {{ buildRoute, matchRoute }} = router;
        const path = buildRoute('/jobs/:namespace/:name/runs/:epoch', {{
          namespace: 'prod/team-a',
          name: 'bench/job 1',
          epoch: '2026-05-18T03:00:00Z',
        }});
        console.log(JSON.stringify({{ path, params: matchRoute('/jobs/:namespace/:name/runs/:epoch', path) }}));
    """

    assert json.loads(run_node(script)) == {
        "path": "/jobs/prod%2Fteam-a/bench%2Fjob%201/runs/2026-05-18T03%3A00%3A00Z",
        "params": {
            "namespace": "prod/team-a",
            "name": "bench/job 1",
            "epoch": "2026-05-18T03:00:00Z",
        },
    }


def test_replace_route_normalizes_hash_path_and_updates_signals() -> None:
    script = f"""
        const events = {{}};
        global.window = {{
          location: {{ hash: '#/old?ns=default' }},
          history: {{
            replaceState(state, title, url) {{ window.location.hash = url; }},
          }},
          addEventListener(name, cb) {{ events[name] = cb; }},
        }};
        {router_import_script()}
        const {{ replaceRoute, route, query }} = router;
        replaceRoute('jobs/prod%2Fteam/job%2Fone?tab=artifacts&empty=');
        console.log(JSON.stringify({{ hash: window.location.hash, route: route.value, query: query.value }}));
    """

    assert json.loads(run_node(script)) == {
        "hash": "#/jobs/prod%2Fteam/job%2Fone?tab=artifacts&empty=",
        "route": "/jobs/prod%2Fteam/job%2Fone",
        "query": {"tab": "artifacts", "empty": ""},
    }
