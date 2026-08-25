# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge tests for the operator Launch page."""

from __future__ import annotations

import json
from pathlib import Path

from tests.unit.ui.node_utils import run_node

UI_DIR = Path(__file__).resolve().parents[3] / "src" / "aiperf" / "operator" / "ui"
LAUNCH_PATH = UI_DIR / "pages" / "launch.js"
API_PATH = UI_DIR / "lib" / "api.js"
YAML_PATH = UI_DIR / "vendor" / "js-yaml.mjs"


def _launch_helper_script(body: str) -> str:
    return f"""
        import fs from 'node:fs';
        import {{ loadAll }} from {json.dumps(YAML_PATH.as_uri())};
        const source = fs.readFileSync({json.dumps(str(LAUNCH_PATH))}, 'utf8');
        const helpers = source
          .slice(0, source.indexOf('export function Launch()'))
          .replace(/^import .*$/gm, '');
        eval(helpers + {json.dumps(chr(10))} + {json.dumps(body)});
    """


def test_launch_default_templates_are_aiperfjob_yaml_with_dated_names() -> None:
    script = _launch_helper_script(
        """
        const RealDate = Date;
        globalThis.Date = class extends RealDate {
          constructor(...args) { super(args.length ? args[0] : '2026-05-18T12:34:56Z'); }
          static now() { return new RealDate('2026-05-18T12:34:56Z').getTime(); }
          static parse(value) { return RealDate.parse(value); }
          static UTC(...args) { return RealDate.UTC(...args); }
        };
        const templates = buildTemplates();
        const peeked = templates.map((template) => ({ id: template.id, ...peekManifest(template.yaml) }));
        console.log(JSON.stringify({ count: templates.length, peeked }));
        """
    )

    out = json.loads(run_node(script))

    assert out == {
        "count": 3,
        "peeked": [
            {
                "id": "llama3-70b-throughput",
                "kind": "AIPerfJob",
                "name": "llama3-70b-throughput-20260518",
                "namespace": "default",
                "parseError": None,
            },
            {
                "id": "mistral-burst",
                "kind": "AIPerfJob",
                "name": "mistral-7b-smoke-20260518",
                "namespace": "default",
                "parseError": None,
            },
            {
                "id": "minimal",
                "kind": "AIPerfJob",
                "name": "my-benchmark",
                "namespace": "default",
                "parseError": None,
            },
        ],
    }


def test_launch_parser_preserves_submit_manifest_shape_scalars_and_urls() -> None:
    script = _launch_helper_script(
        """
        const manifest = parseYaml(`apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfJob
metadata:
  name: edge-case
  namespace: perf-ns
spec:
  benchmark:
    models: [model-a, model-b]
    endpoint:
      urls:
        - "http://trtllm.perf-ns.svc:8000"
      streaming: true
      retries: 3
    datasets:
      main:
        type: synthetic
        isl: { mean: 128, stddev: 0 }
        osl: { mean: 64, stddev: 1.5 }
`);
        console.log(JSON.stringify(manifest));
        """
    )

    manifest = json.loads(run_node(script))

    assert manifest["metadata"] == {"name": "edge-case", "namespace": "perf-ns"}
    assert manifest["spec"]["benchmark"]["models"] == ["model-a", "model-b"]
    endpoint = manifest["spec"]["benchmark"]["endpoint"]
    assert endpoint["urls"] == ["http://trtllm.perf-ns.svc:8000"]
    assert endpoint["streaming"] is True
    assert endpoint["retries"] == 3
    assert manifest["spec"]["benchmark"]["datasets"]["main"]["osl"] == {
        "mean": 64,
        "stddev": 1.5,
    }


def test_launch_parser_preserves_quoted_commas_in_flow_collections() -> None:
    script = _launch_helper_script(
        """
        const manifest = parseLaunchManifest(`apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfJob
metadata: { name: quoted-comma, namespace: default,
  annotations: { note: "left,right" } }
spec: { benchmark: { models: ["model,with-comma", model-b] } }`);
        console.log(JSON.stringify(manifest));
        """
    )

    manifest = json.loads(run_node(script))

    assert manifest["metadata"]["annotations"]["note"] == "left,right"
    assert manifest["spec"]["benchmark"]["models"] == [
        "model,with-comma",
        "model-b",
    ]


def test_launch_submit_uses_operator_create_job_payload_wrapper() -> None:
    launch_source = LAUNCH_PATH.read_text()
    api_source = API_PATH.read_text()

    assert "api.createJob(manifest)" in launch_source
    assert "api.createSweep(manifest)" in launch_source
    # createJob/createSweep issue POSTs with { manifest } body, protected by a
    # sessionStorage bearer token. Prompt shown on first use (TokenRequiredError).
    assert "createJob(manifest) {" in api_source
    assert "createSweep(manifest) {" in api_source
    assert "mutatingFetch('/jobs'" in api_source
    assert "mutatingFetch('/sweeps'" in api_source
    assert "isTokenRequiredError" in launch_source


def test_launch_rejects_empty_or_comment_only_manifests_before_submit() -> None:
    script = _launch_helper_script(
        """
        const cases = ['', '   ', '# comment only'];
        console.log(JSON.stringify(cases.map((text) => peekManifest(text))));
        """
    )

    out = json.loads(run_node(script))

    assert all(item["parseError"] for item in out)


def test_launch_validates_kind_namespace_and_name_before_submit() -> None:
    script = _launch_helper_script(
        """
        const cases = [
          `apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfJob
metadata:
  name: Invalid_Name
  namespace: default
spec: {}`,
          `apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfJob
metadata:
  name: valid-name
  namespace: Bad_Namespace
spec: {}`,
          `apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfWidget
metadata:
  name: valid-name
  namespace: default
spec: {}`,
          `apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfJob
metadata:
  namespace: default
spec: {}`,
          `apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfJob
metadata:
  name: valid-name
spec: {}`,
        ];
        console.log(JSON.stringify(cases.map((text) => peekManifest(text))));
        """
    )

    out = json.loads(run_node(script))

    assert all(item["parseError"] for item in out)


def test_launch_submission_path_uses_validated_manifest_parser() -> None:
    launch_source = LAUNCH_PATH.read_text()

    assert "manifest = parseLaunchManifest(yaml);" in launch_source
    assert "manifest = parseYaml(yaml);" not in launch_source


def test_launch_json_manifest_mode_is_not_advertised_as_yaml() -> None:
    launch_source = LAUNCH_PATH.read_text()

    if "application/json" not in launch_source and "JSON manifest" not in launch_source:
        script = _launch_helper_script(
            """
            const result = peekManifest('{"apiVersion":"aiperf.nvidia.com/v1alpha1","kind":"AIPerfJob"}');
            console.log(JSON.stringify(result));
            """
        )
        out = json.loads(run_node(script))
        assert out["parseError"]
        return

    assert "JSON.parse(yaml)" in launch_source or "JSON.parse(manifest" in launch_source


def test_launch_success_url_and_navigation_encode_namespace_and_name() -> None:
    launch_source = LAUNCH_PATH.read_text()

    # Both job and sweep paths must use encodeURIComponent
    assert (
        "/jobs/${encodeURIComponent(state.namespace)}/${encodeURIComponent(state.name)}"
        in launch_source
    )
    assert (
        "/sweeps/${encodeURIComponent(state.namespace)}/${encodeURIComponent(state.name)}"
        in launch_source
    )
    assert "/jobs/${encodeURIComponent(" in launch_source
    assert "/sweeps/${encodeURIComponent(" in launch_source
    assert "href=${`#/jobs/${state.namespace}/${state.name}`}" not in launch_source
    assert "navigate(`/jobs/${state.namespace}/${state.name}`);" not in launch_source


def test_launch_prefill_handoff_is_one_shot_and_time_bounded() -> None:
    launch_source = LAUNCH_PATH.read_text()

    assert "sessionStorage.getItem('aiperf.launch.prefill')" in launch_source
    assert "sessionStorage.removeItem('aiperf.launch.prefill')" in launch_source
    assert "Date.now() - payload.at > 60000" in launch_source
    assert "typeof payload.yaml !== 'string'" in launch_source
