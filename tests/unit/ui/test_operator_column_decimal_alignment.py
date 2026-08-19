# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A table column must render every cell at one decimal precision.

``fmtNumber`` picks precision per value. That is right for a standalone KPI --
0.04 req/s must not collapse to "0.00" -- and wrong for a column. On the real
``gemma-bo5`` cluster sweep the req/s column held 0.5329 and 2, and rendered
them literally as "0.5329" and "2": the decimal point jumped four places
between adjacent rows, so the two magnitudes could not be compared by eye.

``columnDecimals`` + ``fmtFixed`` replace that with one count per column,
derived from all of the column's values.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from pytest import param

from tests.unit.ui.node_utils import FORMAT_JS_IN_TEMPLATE, run_node

_UI = Path(__file__).resolve().parents[3] / "src" / "aiperf" / "operator" / "ui"
FORMAT_PATH = _UI / "lib" / "format.js"

# The exact req/s means from the gemma-bo5 variations table.
GEMMA_BO5_REQ_S = [0.5329, 2.0, 0.5467, 4.0, 12.0, 13.0, 10.0]


def _run(script: str) -> dict:
    return json.loads(run_node(script))


def test_the_real_cluster_column_no_longer_moves_the_decimal_point() -> None:
    """The regression case, pinned to the values that produced it."""
    out = _run(f"""
        import {{ columnDecimals, fmtFixed }} from {FORMAT_PATH.as_uri()!r};
        const values = {json.dumps(GEMMA_BO5_REQ_S)};
        const d = columnDecimals(values, 0);
        console.log(JSON.stringify({{
          decimals: d,
          rendered: values.map(v => fmtFixed(v, d)),
        }}));
    """)

    # Every cell has the same number of fractional digits.
    fracs = {len(text.split(".")[1]) if "." in text else 0 for text in out["rendered"]}
    assert fracs == {out["decimals"]}, (
        f"column is ragged: {out['rendered']} at decimals={out['decimals']}"
    )

    # And the small value kept its magnitude rather than collapsing.
    assert out["rendered"][0].startswith("0.53")
    assert float(out["rendered"][0].replace(",", "")) == pytest.approx(0.5329, abs=1e-4)


@pytest.mark.parametrize(
    "values,base,expected",
    [
        param([138, 447, 1041, 3112], 0, 0, id="all-large-stays-integer"),
        param([1854.82, 2218.10], 2, 2, id="ms-column-unchanged"),
        param([0.5329, 2.0], 0, 4, id="sub-one-widens-the-whole-column"),
        param([0.004, 12.0], 0, 5, id="tiny-value-widens-further"),
        param([], 2, 2, id="empty-falls-back-to-base"),
        param([None, None], 3, 3, id="all-null-falls-back-to-base"),
        param([float("nan"), float("inf"), 5.0], 1, 1, id="non-finite-ignored"),
    ],
)  # fmt: skip
def test_column_decimals_picks_the_widest_requirement(
    values: list, base: int, expected: int
) -> None:
    out = _run(f"""
        import {{ columnDecimals }} from {FORMAT_PATH.as_uri()!r};
        console.log(JSON.stringify({{
          d: columnDecimals({json.dumps(values).replace("NaN", "Number.NaN").replace("Infinity", "Number.POSITIVE_INFINITY")}, {base}),
        }}));
    """)
    assert out["d"] == expected


def test_fmt_fixed_does_not_re_expand_precision() -> None:
    """The whole point: a tiny cell must NOT widen past its column's count.

    If ``fmtFixed`` delegated to ``fmtNumber`` this would return "0.00400" and
    silently restore the ragged column.
    """
    out = _run(f"""
        import {{ fmtFixed }} from {FORMAT_PATH.as_uri()!r};
        console.log(JSON.stringify({{
          tiny: fmtFixed(0.004, 2),
          large: fmtFixed(2, 4),
          grouped: fmtFixed(12345.5, 2),
          nullish: fmtFixed(null, 2, '-'),
          nonFinite: fmtFixed(Number.NaN, 2, '-'),
          negativeDecimals: fmtFixed(1.5, -3),
        }}));
    """)
    assert out["tiny"] == "0.00"
    assert out["large"] == "2.0000"
    assert out["grouped"] == "12,345.50"
    assert out["nullish"] == "-"
    assert out["nonFinite"] == "-"
    assert out["negativeDecimals"] == "2"


def test_variations_table_renders_one_decimal_count_per_column() -> None:
    """Guard the wiring, not just the helpers.

    The helpers can be perfect while the table still calls a per-value
    formatter, which is exactly the state that produced "0.5329" above "2".
    Rendered rather than grepped: the table now pins a per-unit count instead of
    deriving one with `columnDecimals`, and either wiring satisfies the column's
    actual requirement -- one decimal count for every cell in it.
    """
    component = _UI / "components" / "variations-table.js"
    variations = [
        {
            "variation_index": i,
            "label": f"v{i}",
            "n_trials": 1,
            "n_total": 1,
            "perMetric": {
                "request_throughput.avg": {"mean": mean, "std": None, "cv": None}
            },
        }
        for i, mean in enumerate(GEMMA_BO5_REQ_S)
    ]
    out = _run(f"""
        import fs from 'node:fs';
        let source = fs.readFileSync({str(component)!r}, 'utf8');
        source = source.replace(/^import .*;\\n/gm, '');
        source = source.replaceAll('export function ', 'function ');
        source = `
  const palette = new Proxy({{}}, {{ get: (_t, key) => '#' + String(key) }});
  const html = (strings, ...values) => ({{ strings: Array.from(strings), values }});
{FORMAT_JS_IN_TEMPLATE}
${{source}}
export {{ VariationsTable }};`;
        const mod = await import(
          `data:text/javascript;base64,${{Buffer.from(source).toString('base64')}}`
        );
        const rendered = mod.VariationsTable({{
          variations: {json.dumps(variations)},
          headlineMetrics: [
            {{ key: 'request_throughput', stat: 'avg', label: 'Req/s', unit: 'req/s' }},
          ],
        }});
        const flatten = (node) => Array.isArray(node)
          ? node.map(flatten).join('')
          : (node && node.strings)
            ? node.strings.map((s, i) => s + flatten(node.values[i] ?? '')).join('')
            : String(node ?? '');
        console.log(JSON.stringify({{ text: flatten(rendered) }}));
    """)

    cells = re.findall(r"\b\d[\d,]*\.(\d+)\b", out["text"])
    widths = {len(frac) for frac in cells}
    assert widths == {2}, f"column is ragged: {out['text']}"

    # The small value still reads as a small value rather than collapsing.
    assert "0.53" in out["text"]
