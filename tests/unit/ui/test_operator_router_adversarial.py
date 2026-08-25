# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for operator UI router hash/query handling."""

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


def test_empty_hash_defaults_to_root_with_empty_query() -> None:
    script = f"""
        global.window = {{
          location: {{ hash: '' }},
          addEventListener() {{}},
        }};
        {router_import_script()}
        const {{ route, query }} = router;
        console.log(JSON.stringify({{ route: route.value, query: query.value }}));
    """

    assert json.loads(run_node(script)) == {"route": "/", "query": {}}


def test_query_before_hash_is_ignored_in_favor_of_hash_query() -> None:
    script = f"""
        global.window = {{
          location: {{ hash: '#/jobs?ns=hash&tab=runs', search: '?ns=outer&tab=wrong' }},
          addEventListener() {{}},
        }};
        {router_import_script()}
        const {{ route, query }} = router;
        console.log(JSON.stringify({{ route: route.value, query: query.value }}));
    """

    assert json.loads(run_node(script)) == {
        "route": "/jobs",
        "query": {"ns": "hash", "tab": "runs"},
    }


def test_repeated_query_keys_use_last_value_without_collecting_arrays() -> None:
    script = f"""
        global.window = {{
          location: {{ hash: '#/jobs?phase=Pending&phase=Running&ns=default&phase=Completed' }},
          addEventListener() {{}},
        }};
        {router_import_script()}
        const {{ query }} = router;
        console.log(JSON.stringify(query.value));
    """

    assert json.loads(run_node(script)) == {"phase": "Completed", "ns": "default"}


def test_plus_signs_remain_literal_plus_not_space() -> None:
    script = f"""
        global.window = {{
          location: {{ hash: '#/jobs?search=model+a%2Bb&label=C%2B%2B' }},
          addEventListener() {{}},
        }};
        {router_import_script()}
        const {{ hashUrl, query }} = router;
        console.log(JSON.stringify({{
          query: query.value,
          href: hashUrl('/jobs', {{ search: query.value.search, label: query.value.label }}),
        }}));
    """

    assert json.loads(run_node(script)) == {
        "query": {"search": "model+a+b", "label": "C++"},
        "href": "#/jobs?search=model%2Ba%2Bb&label=C%2B%2B",
    }


def test_unicode_query_values_round_trip_through_set_query() -> None:
    script = f"""
        global.window = {{
          location: {{ hash: '#/jobs?ns=%E7%A0%94%E7%A9%B6&name=%F0%9F%9A%80' }},
          addEventListener() {{}},
        }};
        {router_import_script()}
        const {{ query, setQuery }} = router;
        setQuery({{ owner: 'München/東京' }});
        console.log(JSON.stringify({{ query: query.value, hash: window.location.hash }}));
    """

    assert json.loads(run_node(script)) == {
        "query": {"ns": "研究", "name": "🚀"},
        "hash": "#/jobs?ns=%E7%A0%94%E7%A9%B6&name=%F0%9F%9A%80&owner=M%C3%BCnchen%2F%E6%9D%B1%E4%BA%AC",
    }


def test_encoded_path_traversal_strings_are_route_params_not_segments() -> None:
    script = f"""
        global.window = {{
          location: {{ hash: '#/' }},
          addEventListener() {{}},
        }};
        {router_import_script()}
        const {{ buildRoute, matchRoute }} = router;
        const path = buildRoute('/jobs/:namespace/:name', {{
          namespace: '../kube-system',
          name: '..%2Fnot-a-segment',
        }});
        console.log(JSON.stringify({{ path, params: matchRoute('/jobs/:namespace/:name', path) }}));
    """

    assert json.loads(run_node(script)) == {
        "path": "/jobs/..%2Fkube-system/..%252Fnot-a-segment",
        "params": {"namespace": "../kube-system", "name": "..%2Fnot-a-segment"},
    }


def test_set_query_deletes_all_empty_null_and_undefined_updates() -> None:
    script = f"""
        global.window = {{
          location: {{ hash: '#/jobs?ns=default&phase=Running&search=needle&page=2' }},
          addEventListener() {{}},
        }};
        {router_import_script()}
        const {{ setQuery }} = router;
        setQuery({{ ns: undefined, phase: null, search: '', page: 3 }});
        console.log(window.location.hash);
    """

    assert run_node(script) == "#/jobs?page=3"


def test_malformed_percent_encoding_in_query_does_not_crash_router() -> None:
    script = f"""
        import {{ readFileSync }} from 'node:fs';
        global.window = {{
          location: {{ hash: '#/jobs?search=%E0%A4%A&ok=yes' }},
          addEventListener() {{}},
        }};
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
        try {{
          const router = await import(routerModuleUrl);
          const {{ route, query }} = router;
          console.log(JSON.stringify({{ ok: true, route: route.value, query: query.value }}));
        }} catch (error) {{
          console.log(JSON.stringify({{ ok: false, name: error.name, message: error.message }}));
        }}
    """

    assert json.loads(run_node(script)) == {
        "ok": True,
        "route": "/jobs",
        "query": {"search": "%E0%A4%A", "ok": "yes"},
    }


def test_malformed_percent_encoding_in_route_param_does_not_crash_match() -> None:
    script = f"""
        global.window = {{
          location: {{ hash: '#/' }},
          addEventListener() {{}},
        }};
        {router_import_script()}
        const {{ matchRoute }} = router;
        try {{
          console.log(JSON.stringify({{ ok: true, params: matchRoute('/jobs/:namespace/:name', '/jobs/default/bad%E0%A4%A') }}));
        }} catch (error) {{
          console.log(JSON.stringify({{ ok: false, name: error.name, message: error.message }}));
        }}
    """

    assert json.loads(run_node(script)) == {
        "ok": True,
        "params": {"namespace": "default", "name": "bad%E0%A4%A"},
    }
