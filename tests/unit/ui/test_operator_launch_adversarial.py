# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for the operator Launch page parser and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.unit.ui.node_utils import run_node

UI_DIR = Path(__file__).resolve().parents[3] / "src" / "aiperf" / "operator" / "ui"
LAUNCH_PATH = UI_DIR / "pages" / "launch.js"
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


def test_launch_rejects_comments_only_document_with_directives() -> None:
    script = _launch_helper_script(
        """
        const cases = [
          '# only a comment\\n# another comment',
          '%YAML 1.2\\n---\\n# no manifest body',
          '---\\n...\\n',
        ];
        console.log(JSON.stringify(cases.map((yaml) => peekManifest(yaml))));
        """
    )

    results = json.loads(run_node(script))

    assert all(item["parseError"] for item in results)
    assert all(item["namespace"] is None and item["name"] is None for item in results)


def test_launch_rejects_wrong_kind_and_multidoc_yaml() -> None:
    script = _launch_helper_script(
        """
        const cases = [
          `apiVersion: aiperf.nvidia.com/v1alpha1
kind: ConfigMap
metadata:
  name: valid-name
  namespace: default
spec: {}`,
          `apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfJob
metadata:
  name: first-job
  namespace: default
spec: {}
---
apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfJob
metadata:
  name: second-job
  namespace: default
spec: {}`,
        ];
        console.log(JSON.stringify(cases.map((yaml) => peekManifest(yaml))));
        """
    )

    results = json.loads(run_node(script))

    # The launch form accepts both workload kinds now, and says so.
    assert (
        results[0]["parseError"]
        == "kind must be AIPerfJob or AIPerfSweep, got ConfigMap."
    )
    assert results[1]["parseError"] == "Launch YAML must contain exactly one document."


@pytest.mark.parametrize(
    "name, namespace",
    [
        ("Uppercase", "default"),
        ("valid-name", "Uppercase"),
        ("-starts-with-dash", "default"),
        ("ends-with-dash-", "default"),
        ("contains_underscore", "default"),
        ("valid-name", "contains.dot"),
        ("a" * 254, "default"),
        ("valid-name", "a" * 64),
    ],
)  # fmt: skip
def test_launch_rejects_invalid_kubernetes_metadata_names(
    name: str, namespace: str
) -> None:
    script = _launch_helper_script(
        f"""
        const yaml = `apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfJob
metadata:
  name: {name}
  namespace: {namespace}
spec: {{}}`;
        console.log(JSON.stringify(peekManifest(yaml)));
        """
    )

    result = json.loads(run_node(script))

    assert result["parseError"]
    assert result["namespace"] is None
    assert result["name"] is None


def test_launch_parser_handles_huge_scalar_without_accepting_huge_metadata_name() -> (
    None
):
    script = _launch_helper_script(
        """
        const huge = 'x'.repeat(20000);
        const validHugeSpec = `apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfJob
metadata:
  name: valid-name
  namespace: default
spec:
  benchmark:
    notes: "${huge}"`;
        const hugeName = `apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfJob
metadata:
  name: ${huge}
  namespace: default
spec: {}`;
        console.log(JSON.stringify({
          valid: parseLaunchManifest(validHugeSpec).spec.benchmark.notes.length,
          invalid: peekManifest(hugeName),
        }));
        """
    )

    result = json.loads(run_node(script))

    assert result["valid"] == 20000
    assert (
        result["invalid"]["parseError"]
        == "metadata.name must be a valid Kubernetes DNS subdomain."
    )


def test_launch_rejects_prototype_pollution_like_keys() -> None:
    script = _launch_helper_script(
        """
        const cases = [
          `apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfJob
metadata:
  name: valid-name
  namespace: default
  __proto__:
    polluted: yes
spec: {}`,
          `apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfJob
metadata:
  name: valid-name
  namespace: default
spec:
  benchmark:
    constructor: evil`,
          `apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfJob
metadata:
  name: valid-name
  namespace: default
spec:
  benchmark:
    prototype: evil`,
        ];
        const results = cases.map((yaml) => {
          const peek = peekManifest(yaml);
          return {
            parseError: peek.parseError,
            objectPrototypePolluted: Object.prototype.polluted === 'yes',
          };
        });
        console.log(JSON.stringify(results));
        """
    )

    results = json.loads(run_node(script))

    assert [item["parseError"] for item in results] == [
        "Manifest: key '__proto__' is not allowed in launch YAML.",
        "Manifest: key 'constructor' is not allowed in launch YAML.",
        "Manifest: key 'prototype' is not allowed in launch YAML.",
    ]
    assert all(not item["objectPrototypePolluted"] for item in results)


def test_launch_rejects_merge_key_prototype_pollution() -> None:
    script = _launch_helper_script(
        """
        const yaml = `apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfJob
metadata: { name: valid-name, namespace: default }
spec:
  benchmark:
    <<: &unsafe { __proto__: { polluted: yes } }`;
        const result = peekManifest(yaml);
        console.log(JSON.stringify({
          parseError: result.parseError,
          objectPrototypePolluted: Object.prototype.polluted === 'yes',
        }));
        """
    )

    result = json.loads(run_node(script))

    assert result == {
        "parseError": "Manifest: key '__proto__' is not allowed in launch YAML.",
        "objectPrototypePolluted": False,
    }


def test_launch_rejects_timestamp_values_to_keep_payloads_json_like() -> None:
    script = _launch_helper_script(
        """
        const result = peekManifest(`apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfJob
metadata: { name: valid-name, namespace: default }
spec:
  benchmark:
    requestedAt: 2026-08-17T01:02:03Z`);
        console.log(JSON.stringify(result));
        """
    )

    result = json.loads(run_node(script))

    assert (
        result["parseError"]
        == "manifest.spec.benchmark.requestedAt: only plain YAML mappings and sequences are allowed."
    )


def test_launch_vendored_yaml_parser_uses_patched_release() -> None:
    source = YAML_PATH.read_text(encoding="utf-8")

    assert "js-yaml 4.1.1" in source


def test_launch_prefill_handoff_has_tamper_resistant_static_guards() -> None:
    source = LAUNCH_PATH.read_text()

    assert "sessionStorage.getItem('aiperf.launch.prefill')" in source
    assert "sessionStorage.removeItem('aiperf.launch.prefill')" in source
    assert "JSON.parse(raw)" in source
    assert "typeof payload.yaml !== 'string'" in source
    assert "Date.now() - payload.at > 60000" in source
    assert "setYaml(payload.yaml)" in source
    assert "eval(" not in source
    assert "innerHTML" not in source
