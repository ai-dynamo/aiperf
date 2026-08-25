# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Additional edge tests for live variations card helper logic."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

COMPONENT_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "aiperf"
    / "operator"
    / "ui"
    / "components"
    / "live-variations-card.js"
)
HELPERS_PATH = COMPONENT_PATH.with_name("live-variations-helpers.js")


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


def _run_component_helper_script(body: str) -> dict[str, object]:
    script = f"""
        import fs from 'node:fs';

        const componentPath = {str(COMPONENT_PATH)!r};
        const helperUri = {HELPERS_PATH.as_uri()!r};
        let source = fs.readFileSync(componentPath, 'utf8');
        source = source.replace(/^import .*;\\n/gm, '');
        source = source.replaceAll('export function ', 'function ');
        source = `import {{ parseVariationValues, titleCase, trialContributesMetrics }} from ${{JSON.stringify(helperUri)}};\n${{source}}\nexport {{ groupVariations, parseVariationLabel, summaryMetric }};`;
        const moduleUri = `data:text/javascript;base64,${{Buffer.from(source).toString('base64')}}`;
        const helpers = await import(moduleUri);
        {body}
    """
    return json.loads(_run_node(script))


def test_trial_phase_semantics_for_terminal_and_failed_states() -> None:
    script = f"""
        import {{ trialContributesMetrics }} from {HELPERS_PATH.as_uri()!r};
        console.log(JSON.stringify({{
          succeeded: trialContributesMetrics('Succeeded'),
          completed: trialContributesMetrics('Completed'),
          archived: trialContributesMetrics('Archived'),
          cancelled: trialContributesMetrics('Cancelled'),
          partiallyFailed: trialContributesMetrics('PartiallyFailed'),
          failed: trialContributesMetrics('Failed'),
          unknown: trialContributesMetrics('Unknown'),
        }}));
    """

    assert json.loads(_run_node(script)) == {
        "succeeded": True,
        "completed": True,
        "archived": True,
        "cancelled": False,
        "partiallyFailed": False,
        "failed": False,
        "unknown": False,
    }


def test_parse_variation_label_handles_live_and_sanitized_forms() -> None:
    result = _run_component_helper_script(
        """
        console.log(JSON.stringify({
          live: helpers.parseVariationLabel('benchmark.phases.profiling.concurrency=10, benchmark.phases.profiling.request_rate=25'),
          sanitizedPhase: helpers.parseVariationLabel('benchmark.phases.profiling.concurrency-10'),
          sanitizedBenchmark: helpers.parseVariationLabel('benchmark.max_tokens-256'),
          noValue: helpers.parseVariationLabel('custom_label'),
          empty: helpers.parseVariationLabel(''),
          nonString: helpers.parseVariationLabel(null),
        }));
        """
    )

    assert result == {
        "live": [
            {"name": "Concurrency", "value": "10"},
            {"name": "Request Rate", "value": "25"},
        ],
        "sanitizedPhase": [{"name": "Concurrency", "value": "10"}],
        "sanitizedBenchmark": [{"name": "Max Tokens", "value": "256"}],
        "noValue": [{"name": "Custom Label", "value": ""}],
        "empty": [],
        "nonString": [],
    }


def test_group_variations_accepts_snake_and_camel_case_fields() -> None:
    result = _run_component_helper_script(
        """
        const manifest = [
          {name: 'child-b-t1', variationLabel: 'benchmark.model=beta', variationIndex: 2, trialIndex: 1, status: 'Pending'},
          {name: 'child-a-t1', variation_label: 'benchmark.model=alpha', variation_index: 1, trial_index: 1, status: 'Pending'},
          {name: 'child-a-t0', variation_label: 'benchmark.model=alpha', variation_index: 1, trial_index: 0, status: 'Running'},
        ];
        const childData = {
          'child-b-t1': {phase: 'Archived', progressPercent: 100, summary: {outputTokenThroughputTps: 40}},
          'child-a-t1': {phase: 'Cancelled'},
          'child-a-t0': {progressPercent: 12},
        };
        console.log(JSON.stringify(helpers.groupVariations(manifest, childData)));
        """
    )

    assert result == [
        {
            "label": "benchmark.model=alpha",
            "variation_index": 1,
            "chips": [{"name": "Model", "value": "alpha"}],
            # No `variation_values` on these entries, so the chips were parsed
            # out of the label; the card uses this to decide whether the cell id
            # still needs a separate line under them.
            "fromValues": False,
            "trials": [
                {
                    "trial_index": 0,
                    "child_name": "child-a-t0",
                    "phase": "Running",
                    "progressPercent": 12,
                    "summary": None,
                },
                {
                    "trial_index": 1,
                    "child_name": "child-a-t1",
                    "phase": "Cancelled",
                    "progressPercent": None,
                    "summary": None,
                },
            ],
        },
        {
            "label": "benchmark.model=beta",
            "variation_index": 2,
            "chips": [{"name": "Model", "value": "beta"}],
            "fromValues": False,
            "trials": [
                {
                    "trial_index": 1,
                    "child_name": "child-b-t1",
                    "phase": "Archived",
                    "progressPercent": 100,
                    "summary": {"outputTokenThroughputTps": 40},
                },
            ],
        },
    ]


def test_summary_metric_handles_camel_snake_and_missing_summaries() -> None:
    result = _run_component_helper_script(
        """
        console.log(JSON.stringify({
          camel: helpers.summaryMetric({outputTokenThroughputTps: 123}, 'outputTokenThroughputTps', 'output_token_throughput'),
          snakeAvg: helpers.summaryMetric({output_token_throughput: {avg: 45}}, 'outputTokenThroughputTps', 'output_token_throughput'),
          snakeP99: helpers.summaryMetric({request_latency: {p99: 67.8}}, 'requestLatencyP99Ms', 'request_latency', 'p99'),
          missingSummary: helpers.summaryMetric(null, 'ttftMs', 'time_to_first_token'),
          missingMetric: helpers.summaryMetric({time_to_first_token: {}}, 'ttftMs', 'time_to_first_token'),
          nonNumericCamel: helpers.summaryMetric({ttftMs: '5'}, 'ttftMs', 'time_to_first_token'),
        }));
        """
    )

    assert result == {
        "camel": 123,
        "snakeAvg": 45,
        "snakeP99": 67.8,
        "missingSummary": None,
        "missingMetric": None,
        "nonNumericCamel": None,
    }
