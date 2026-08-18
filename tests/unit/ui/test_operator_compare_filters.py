# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from pathlib import Path

COMPARE_FILTERS_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "aiperf"
    / "operator"
    / "ui"
    / "pages"
    / "compare-filters.js"
)


def _run_node(script: str) -> str:
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)
    return result.stdout.strip()


_JOBS_LITERAL = """[
  { job_id: 'a', namespace: 'default', model: 'meta/llama-3', endpoint: '/v1/chat' },
  { job_id: 'b', namespace: 'default', model: 'meta/llama-3', endpoint: '/v1/chat' },
  { job_id: 'c', namespace: 'staging', model: 'openai/gpt-oss', endpoint: '/v1/chat' },
  { job_id: 'd', namespace: 'staging', model: null,            endpoint: null },
  { job_id: 'e', namespace: 'bench',   model: 'meta/llama-3', endpoint: '/v1/completions' },
]"""


def _filter_script(filters_js: str, search: str = "") -> str:
    return f"""
        import {{ applyJobFilters }} from {COMPARE_FILTERS_PATH.as_uri()!r};
        const jobs = {_JOBS_LITERAL};
        const out = applyJobFilters(jobs, {filters_js});
        console.log(out.map(j => j.job_id).join(','));
    """


def test_apply_filters_no_filters_returns_all() -> None:
    script = _filter_script(
        "{ nsFilter: new Set(), modelFilter: new Set(), endpointFilter: new Set(), search: '' }"
    )
    assert _run_node(script) == "a,b,c,d,e"


def test_apply_filters_single_namespace_narrows() -> None:
    script = _filter_script(
        "{ nsFilter: new Set(['default']), modelFilter: new Set(), endpointFilter: new Set(), search: '' }"
    )
    assert _run_node(script) == "a,b"


def test_apply_filters_multi_value_namespace_is_or_within_dimension() -> None:
    script = _filter_script(
        "{ nsFilter: new Set(['default', 'bench']), modelFilter: new Set(), endpointFilter: new Set(), search: '' }"
    )
    assert _run_node(script) == "a,b,e"


def test_apply_filters_two_dimensions_are_and() -> None:
    script = _filter_script(
        "{ nsFilter: new Set(['staging']), modelFilter: new Set(['openai/gpt-oss']), endpointFilter: new Set(), search: '' }"
    )
    assert _run_node(script) == "c"


def test_apply_filters_none_bucket_keeps_null_model_jobs() -> None:
    script = _filter_script(
        "{ nsFilter: new Set(), modelFilter: new Set(['__none__']), endpointFilter: new Set(), search: '' }"
    )
    assert _run_node(script) == "d"


def test_apply_filters_search_composes_with_chips() -> None:
    script = _filter_script(
        "{ nsFilter: new Set(['default']), modelFilter: new Set(), endpointFilter: new Set(), search: 'b' }"
    )
    assert _run_node(script) == "b"


def test_apply_filters_search_matches_model() -> None:
    script = _filter_script(
        "{ nsFilter: new Set(), modelFilter: new Set(), endpointFilter: new Set(), search: 'llama' }"
    )
    assert _run_node(script) == "a,b,e"


def test_compare_search_endpoint_text_requires_endpoint_chip_or_is_documented_not_searchable() -> (
    None
):
    script = _filter_script(
        "{ nsFilter: new Set(), modelFilter: new Set(), endpointFilter: new Set(), search: 'completions' }"
    )
    assert _run_node(script) == "e"


def test_extract_facets_counts_distinct_values_and_buckets_nulls() -> None:
    script = f"""
        import {{ extractFacets }} from {COMPARE_FILTERS_PATH.as_uri()!r};
        const jobs = {_JOBS_LITERAL};
        const f = extractFacets(jobs);
        const dump = (m) => Array.from(m.entries());
        console.log(JSON.stringify({{
          ns: dump(f.ns),
          model: dump(f.model),
          endpoint: dump(f.endpoint),
        }}));
    """
    out = _run_node(script)
    import json

    parsed = json.loads(out)
    assert dict(parsed["ns"]) == {"default": 2, "staging": 2, "bench": 1}
    assert dict(parsed["model"]) == {
        "meta/llama-3": 3,
        "openai/gpt-oss": 1,
        "__none__": 1,
    }
    assert dict(parsed["endpoint"]) == {
        "/v1/chat": 3,
        "/v1/completions": 1,
        "__none__": 1,
    }
