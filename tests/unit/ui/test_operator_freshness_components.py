# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source checks for operator UI freshness components."""

from __future__ import annotations

import json
from pathlib import Path

from tests.unit.ui.node_utils import run_node

UI_ROOT = Path(__file__).resolve().parents[3] / "src" / "aiperf" / "operator" / "ui"
COMPONENT_PATH = UI_ROOT / "components" / "freshness.js"
APP_PATH = UI_ROOT / "app.js"
STYLE_PATH = UI_ROOT / "style.css"


def _freshness_component_import_script() -> str:
    return f"""
        import {{ readFileSync }} from 'node:fs';
        const freshnessSources = {{ value: [] }};
        const renderValue = (value) => Array.isArray(value)
          ? value.join('')
          : value == null || value === false
            ? ''
            : String(value);
        globalThis.__freshnessHtml = (strings, ...values) => {{
          if (strings[0] === '<' && typeof values[0] === 'function') {{
            return values[0]({{ source: values[2], compact: values[3] }});
          }}
          return strings.reduce(
            (acc, part, index) => acc + part + renderValue(values[index]),
            '',
          );
        }};
        globalThis.__freshnessSources = freshnessSources;
        let source = readFileSync({str(COMPONENT_PATH)!r}, 'utf8');
        source = source
          .replace("import {{ html }} from 'htm/preact';", 'const html = globalThis.__freshnessHtml;')
          .replace(
            "import {{ freshnessSources }} from '../lib/state.js';",
            'const freshnessSources = globalThis.__freshnessSources;',
          );
        const componentUrl = 'data:text/javascript;base64,' + Buffer.from(source).toString('base64');
        const components = await import(componentUrl);
    """


def test_freshness_components_define_strip_pill_and_stale_banner() -> None:
    source = COMPONENT_PATH.read_text(encoding="utf-8")

    assert "export function FreshnessPill" in source
    assert "export function FreshnessStrip" in source
    assert "export function StaleBanner" in source
    assert 'data-testid="freshness-strip"' in source
    assert 'data-testid="freshness-pill"' in source
    assert 'data-testid="stale-banner"' in source
    assert "last successful update" in source
    assert "Retrying" in source


def test_freshness_strip_returns_null_when_there_are_no_sources() -> None:
    script = f"""
        {_freshness_component_import_script()}
        console.log(JSON.stringify(components.FreshnessStrip()));
    """

    assert json.loads(run_node(script)) is None


def test_freshness_strip_renders_test_id_and_source_label() -> None:
    script = f"""
        {_freshness_component_import_script()}
        freshnessSources.value = [{{ source: 'job-detail', status: 'fresh', lastSuccessAt: Date.now() }}];
        const output = components.FreshnessStrip();
        console.log(JSON.stringify({{
          hasStrip: output.includes('data-testid="freshness-strip"'),
          hasSource: output.includes('Job Detail'),
        }}));
    """

    assert json.loads(run_node(script)) == {"hasStrip": True, "hasSource": True}


def test_freshness_pill_compact_separates_source_and_status() -> None:
    script = f"""
        {_freshness_component_import_script()}
        const now = Date.now();
        const output = components.FreshnessPill({{
          source: {{ source: 'jobs', status: 'fresh', lastSuccessAt: now }},
          compact: true,
        }});
        console.log(JSON.stringify({{
          hasSeparatedLabel: output.includes('Jobs Live') || output.includes('Jobs · Live'),
          hasConcatenatedLabel: output.includes('JobsLive'),
        }}));
    """

    assert json.loads(run_node(script)) == {
        "hasSeparatedLabel": True,
        "hasConcatenatedLabel": False,
    }


def test_stale_banner_returns_null_for_fresh_sources() -> None:
    script = f"""
        {_freshness_component_import_script()}
        console.log(JSON.stringify(components.StaleBanner({{ source: {{ status: 'fresh', source: 'jobs' }} }})));
    """

    assert json.loads(run_node(script)) is None


def test_stale_banner_renders_test_id_and_last_known_data_message() -> None:
    script = f"""
        {_freshness_component_import_script()}
        const output = components.StaleBanner({{
          source: {{ status: 'stale', source: 'jobs', lastSuccessAt: Date.now() - 2000 }},
        }});
        console.log(JSON.stringify({{
          hasBanner: output.includes('data-testid="stale-banner"'),
          hasLastKnownData: output.includes('showing last-known data'),
        }}));
    """

    assert json.loads(run_node(script)) == {"hasBanner": True, "hasLastKnownData": True}


def test_app_shell_renders_freshness_strip_below_alpha_banner() -> None:
    source = APP_PATH.read_text(encoding="utf-8")

    assert "import { FreshnessStrip } from './components/freshness.js';" in source
    assert "<${FreshnessStrip} />" in source
    assert source.index('data-testid="alpha-banner"') < source.index(
        "<${FreshnessStrip} />"
    )


def test_freshness_css_classes_exist() -> None:
    source = STYLE_PATH.read_text(encoding="utf-8")

    for selector in [
        ".freshness-strip",
        ".freshness-pill",
        ".freshness-pill--fresh",
        ".freshness-pill--stale",
        ".freshness-pill--retrying",
        ".freshness-pill--stopped",
        ".stale-banner",
    ]:
        assert selector in source


def test_stale_banner_speaks_up_for_a_source_that_never_loaded() -> None:
    """status='failed' must not claim last-known data is on screen.

    A never-loaded source has nothing behind it, so "showing last-known data"
    would send the reader hunting for stale numbers on an empty page.
    """
    script = f"""
        {_freshness_component_import_script()}
        const output = components.StaleBanner({{
          source: {{
            status: 'failed',
            source: 'jobs',
            lastSuccessAt: null,
            lastError: 'API 503: operator down',
          }},
        }});
        console.log(JSON.stringify({{
          hasBanner: output.includes('data-testid="stale-banner"'),
          hasFailedClass: output.includes('stale-banner--failed'),
          hasCouldNotLoad: output.includes('could not be loaded'),
          claimsLastKnownData: output.includes('showing last-known data'),
          hasError: output.includes('API 503: operator down'),
        }}));
    """

    assert json.loads(run_node(script)) == {
        "hasBanner": True,
        "hasFailedClass": True,
        "hasCouldNotLoad": True,
        "claimsLastKnownData": False,
        "hasError": True,
    }


def test_freshness_pill_labels_a_failed_source_as_failed_not_loading() -> None:
    script = f"""
        {_freshness_component_import_script()}
        const output = components.FreshnessPill({{
          source: {{
            source: 'jobs',
            status: 'failed',
            lastSuccessAt: null,
            lastAttemptAt: Date.now(),
            lastError: 'API 503: operator down',
          }},
          compact: true,
        }});
        console.log(JSON.stringify({{
          hasFailedLabel: output.includes('Failed'),
          hasFailedClass: output.includes('freshness-pill--failed'),
          claimsLoading: output.includes('Loading'),
          titleCarriesError: output.includes('last error: API 503: operator down'),
        }}));
    """

    assert json.loads(run_node(script)) == {
        "hasFailedLabel": True,
        "hasFailedClass": True,
        "claimsLoading": False,
        "titleCarriesError": True,
    }


def test_failed_freshness_states_have_css_rules() -> None:
    source = STYLE_PATH.read_text(encoding="utf-8")

    assert ".freshness-pill--failed" in source
    assert ".stale-banner--failed" in source
