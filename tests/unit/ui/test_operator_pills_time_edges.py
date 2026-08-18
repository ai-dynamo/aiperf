# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge-case tests for operator UI pill and relative-time components."""

from __future__ import annotations

import json
from pathlib import Path

from tests.unit.ui.node_utils import run_node

UI_ROOT = Path(__file__).resolve().parents[3] / "src" / "aiperf" / "operator" / "ui"
PILLS_PATH = UI_ROOT / "components" / "pills.js"
TIME_PATH = UI_ROOT / "components" / "time.js"


def _component_probe() -> dict[str, object]:
    script = f"""
        import fs from 'node:fs';

        const html = (strings, ...values) => ({{ strings: [...strings], values }});
        const palette = {{ teal: '#008080', indigo: '#4f46e5' }};
        const modelColor = (model) => `color-for-${{model.length}}`;
        const useState = () => [0, () => undefined];
        const useEffect = () => undefined;

        function loadModule(path) {{
          let src = fs.readFileSync(path, 'utf8');
          src = src.replace(/^import .*;$/gm, '');
          src = src.replace(/export function /g, 'function ');
          src = src.replace(/export const /g, 'const ');
          return Function(
            'html', 'palette', 'modelColor', 'useState', 'useEffect',
            `${{src}}\nreturn {{\n` +
              `  NsPill: typeof NsPill === 'undefined' ? undefined : NsPill,\n` +
              `  EpochPill: typeof EpochPill === 'undefined' ? undefined : EpochPill,\n` +
              `  ModelPill: typeof ModelPill === 'undefined' ? undefined : ModelPill,\n` +
              `  fmtRelativeSeconds: typeof fmtRelativeSeconds === 'undefined' ? undefined : fmtRelativeSeconds,\n` +
              `  fmtElapsedSeconds: typeof fmtElapsedSeconds === 'undefined' ? undefined : fmtElapsedSeconds,\n` +
              `  fmtAbsolute: typeof fmtAbsolute === 'undefined' ? undefined : fmtAbsolute,\n` +
              `  RelativeTime: typeof RelativeTime === 'undefined' ? undefined : RelativeTime,\n` +
              `}};`,
          )(html, palette, modelColor, useState, useEffect);
        }}

        const pills = loadModule({str(PILLS_PATH)!r});
        const time = loadModule({str(TIME_PATH)!r});
        const longNamespace = 'tenant-' + 'x'.repeat(180) + '-production';
        const longModel = 'nvidia/' + 'llama-'.repeat(40) + '70b';
        const calls = [];
        const nsPill = pills.NsPill({{
          ns: longNamespace,
          testId: 'long-ns',
          onClick: (ns) => calls.push(['ns', ns]),
        }});
        let nsStopped = false;
        nsPill.values[4]({{ stopPropagation: () => {{ nsStopped = true; }} }});
        const modelPill = pills.ModelPill({{
          model: longModel,
          testId: 'long-model',
          onClick: (model) => calls.push(['model', model]),
        }});
        let modelStopped = false;
        modelPill.values[4]({{ stopPropagation: () => {{ modelStopped = true; }} }});
        const epochPill = pills.EpochPill({{ epoch: '000123', isLatest: true, testId: 'ep' }});

        const originalNow = Date.now;
        Date.now = () => Date.parse('2026-05-18T12:00:00Z');
        const fromTsShort = time.RelativeTime({{
          ts: '2026-05-18T11:58:30Z',
          suffix: 'ago',
          className: 'age',
        }});
        const fromTsElapsed = time.RelativeTime({{
          ts: '2026-05-18T09:45:00Z',
          mode: 'elapsed',
        }});
        const secondsWinsOverTs = time.RelativeTime({{
          ts: '2026-05-18T00:00:00Z',
          seconds: 65,
          mode: 'elapsed',
        }});
        const futureTs = time.RelativeTime({{ ts: '2026-05-18T12:00:30Z', suffix: 'ago' }});
        const invalidTs = time.RelativeTime({{ ts: 'not-a-date', className: 'bad-date' }});
        Date.now = originalNow;

        const out = {{
          emptyPills: {{
            nsNull: pills.NsPill({{ ns: null }}) === null,
            nsEmpty: pills.NsPill({{ ns: '' }}) === null,
            epochNull: pills.EpochPill({{ epoch: null }}) === null,
            epochEmpty: pills.EpochPill({{ epoch: '' }}) === null,
            modelNull: pills.ModelPill({{ model: null }}) === null,
            modelEmpty: pills.ModelPill({{ model: '' }}) === null,
          }},
          nsPill: {{
            className: nsPill.values[0],
            style: nsPill.values[1],
            title: nsPill.values[2],
            testId: nsPill.values[3],
            label: nsPill.values[nsPill.values.length - 1],
            stopped: nsStopped,
          }},
          modelPill: {{
            className: modelPill.values[0],
            style: modelPill.values[1],
            title: modelPill.values[2],
            testId: modelPill.values[3],
            label: modelPill.values[modelPill.values.length - 1],
            stopped: modelStopped,
          }},
          epochPill: {{
            style: epochPill.values[0],
            title: epochPill.values[1],
            testId: epochPill.values[2],
            label: epochPill.values[3],
            suffix: epochPill.values[4] ? epochPill.values[4].values[0] : null,
          }},
          calls,
          timeHelpers: {{
            nullRelative: time.fmtRelativeSeconds(null),
            nanRelative: time.fmtRelativeSeconds(Number.NaN),
            futureRelative: time.fmtRelativeSeconds(-30),
            elapsedTwoUnits: time.fmtElapsedSeconds(8130),
            elapsedFuture: time.fmtElapsedSeconds(-10),
            absoluteIso: time.fmtAbsolute('2026-05-18T12:34:56Z'),
            absoluteZero: time.fmtAbsolute(0),
            absoluteEmpty: time.fmtAbsolute(''),
            absoluteInvalid: time.fmtAbsolute('not-a-date'),
          }},
          relativeTime: {{
            fromTsShort: {{ text: fromTsShort.values[2], suffix: fromTsShort.values[3], title: fromTsShort.values[1] }},
            fromTsElapsed: {{ text: fromTsElapsed.values[2], title: fromTsElapsed.values[1] }},
            secondsWinsOverTs: {{ text: secondsWinsOverTs.values[2], title: secondsWinsOverTs.values[1] }},
            futureTs: {{ text: futureTs.values[2], suffix: futureTs.values[3], title: futureTs.values[1] }},
            invalidTs: {{ strings: invalidTs.strings, values: invalidTs.values }},
          }},
        }};
        console.log(JSON.stringify(out));
    """
    return json.loads(run_node(script))


def test_pills_hide_null_and_empty_labels() -> None:
    out = _component_probe()

    assert out["emptyPills"] == {
        "nsNull": True,
        "nsEmpty": True,
        "epochNull": True,
        "epochEmpty": True,
        "modelNull": True,
        "modelEmpty": True,
    }


def test_clickable_namespace_and_model_pills_preserve_long_labels_and_stop_row_clicks() -> (
    None
):
    out = _component_probe()

    ns = out["nsPill"]
    assert ns["className"] == "meta-pill meta-pill--clickable"
    assert "cursor:pointer" in ns["style"]
    assert ns["title"] == f"Filter by namespace: {ns['label']}"
    assert ns["testId"] == "long-ns"
    assert ns["label"].startswith("tenant-")
    assert ns["label"].endswith("-production")
    assert len(ns["label"]) > 180
    assert ns["stopped"] is True

    # The pill no longer carries a per-model colour dot: colour is reserved for
    # state across the console. The label, the filter affordance and the
    # click-does-not-select-the-row guarantee are what this protects.
    model = out["modelPill"]
    assert model["className"] == "meta-pill meta-pill--model meta-pill--clickable"
    assert model["style"] == "cursor:pointer;"
    assert model["title"] == f"Filter by model: {model['label']}"
    assert model["testId"] == "long-model"
    assert model["label"].startswith("nvidia/llama-")
    assert len(model["label"]) > 200
    assert model["stopped"] is True

    assert out["calls"] == [["ns", ns["label"]], ["model", model["label"]]]


def test_epoch_pill_keeps_zero_padded_epoch_label_and_latest_suffix() -> None:
    out = _component_probe()

    epoch = out["epochPill"]
    assert "cursor:pointer" not in epoch["style"]
    assert epoch["title"] == "Epoch: 000123 (latest)"
    assert epoch["testId"] == "ep"
    assert epoch["label"] == "000123"


def test_relative_time_helpers_handle_null_future_invalid_and_timestamp_inputs() -> (
    None
):
    out = _component_probe()["timeHelpers"]

    assert out["nullRelative"] == "---"
    assert out["nanRelative"] == "---"
    assert out["futureRelative"] == "0s"
    assert out["elapsedTwoUnits"] == "2h 15m"
    assert out["elapsedFuture"] == "0s"
    assert "2026" in out["absoluteIso"]
    assert out["absoluteZero"] == ""
    assert out["absoluteEmpty"] == ""
    assert out["absoluteInvalid"] == "not-a-date"


def test_relative_time_component_uses_ts_age_seconds_and_elapsed_modes() -> None:
    out = _component_probe()["relativeTime"]

    assert out["fromTsShort"]["text"] == "1m"
    assert out["fromTsShort"]["suffix"] == " ago"
    assert "2026" in out["fromTsShort"]["title"]
    assert out["fromTsElapsed"]["text"] == "2h 15m"
    assert "2026" in out["fromTsElapsed"]["title"]
    assert out["secondsWinsOverTs"]["text"] == "1m 5s"
    assert "2026" in out["secondsWinsOverTs"]["title"]


def test_relative_time_component_clamps_future_and_rejects_invalid_timestamps() -> None:
    out = _component_probe()["relativeTime"]

    assert out["futureTs"]["text"] == "0s"
    assert out["futureTs"]["suffix"] == " ago"
    assert "2026" in out["futureTs"]["title"]
    assert out["invalidTs"]["values"] == ["bad-date"]
    assert "---" in "".join(out["invalidTs"]["strings"])
