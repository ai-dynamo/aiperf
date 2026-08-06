#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare two mock server recording JSONL files and produce an HTML report.

Each input file is the per-request JSONL produced by
``aiperf-mock-server --record-requests <file.jsonl>``.  The companion
``.summary.json`` file (same base path with ``.summary.json`` suffix) is
read automatically when present to include vocab-distribution data.

Usage::

    python tools/compare_recordings.py \\
        --a  artifacts/run_a/recording.jsonl  --label-a "vLLM bench" \\
        --b  artifacts/run_b/recording.jsonl  --label-b "aiperf" \\
        --out comparison.html
"""

from __future__ import annotations

import argparse
import collections
import html
import json
import math
import statistics
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_summary(path: Path) -> dict | None:
    candidate = path.with_suffix("").with_suffix(".summary.json")
    if not candidate.exists():
        candidate = Path(str(path) + ".summary.json")
    if candidate.exists():
        return json.loads(candidate.read_text())
    return None


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def _finite_vals(rows: list[dict], key: str) -> list[float]:
    return [
        v
        for r in rows
        if (v := r.get(key)) is not None
        and isinstance(v, (int, float))
        and math.isfinite(v)
    ]


def field_stats(rows: list[dict], key: str) -> dict:
    vals = _finite_vals(rows, key)
    if not vals:
        return {}
    n = len(vals)
    if n >= 2:
        qs = statistics.quantiles(vals, n=100, method="inclusive")
        p5, p95 = qs[4], qs[94]
    else:
        p5 = p95 = vals[0]
    return {
        "n": n,
        "min": min(vals),
        "p5": p5,
        "mean": round(statistics.mean(vals), 2),
        "median": statistics.median(vals),
        "p95": p95,
        "max": max(vals),
        "std": round(statistics.stdev(vals) if n > 1 else 0, 2),
    }


def histogram(vals: list[int | float], lo: int, hi: int, step: int = 1) -> list[float]:
    n = len(vals)
    if n == 0:
        return [0.0] * ((hi - lo) // step)
    c: dict[int, int] = collections.Counter()
    for v in vals:
        bucket = int((v - lo) // step) * step + lo
        c[bucket] += 1
    bins = range(lo, hi, step)
    return [round(c.get(b, 0) / n * 100, 4) for b in bins]


def vocab_top_diffs(
    summary_a: dict | None,
    summary_b: dict | None,
    top_n: int = 20,
) -> list[dict]:
    """Return top tokens by absolute count difference across the two summaries."""
    if not summary_a or not summary_b:
        return []

    # Flatten per-endpoint vocab_counts
    def flatten(summary: dict) -> dict[int, int]:
        out: dict[int, int] = {}
        for ep_data in summary.get("per_endpoint", {}).values():
            vc = ep_data.get("vocab_distribution", {})
            if not vc:
                continue
            for tok_id_str, cnt in vc.get("frequencies", vc.get("counts", {})).items():
                out[int(tok_id_str)] = out.get(int(tok_id_str), 0) + cnt
        return out

    vc_a = flatten(summary_a)
    vc_b = flatten(summary_b)
    all_ids = set(vc_a) | set(vc_b)
    diffs = []
    for tok_id in all_ids:
        ca = vc_a.get(tok_id, 0)
        cb = vc_b.get(tok_id, 0)
        diffs.append(
            {"id": tok_id, "a": ca, "b": cb, "delta": cb - ca, "abs": abs(cb - ca)}
        )
    diffs.sort(key=lambda x: -x["abs"])
    return diffs[:top_n]


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

_CSS = """
body{font-family:sans-serif;margin:32px;background:#f9f9f9;color:#222}
h1{font-size:20px;margin-bottom:4px}
h2{font-size:16px;margin-top:32px;margin-bottom:12px;border-bottom:2px solid #e0e0e0;padding-bottom:6px}
.subtitle{color:#666;font-size:13px;margin-bottom:24px}
.stats-row{display:flex;gap:16px;margin-bottom:24px;flex-wrap:wrap}
.stat-card{background:white;border:1px solid #e0e0e0;border-radius:8px;padding:14px 18px;min-width:140px}
.stat-card h3{margin:0 0 6px;font-size:11px;color:#666;text-transform:uppercase;letter-spacing:.05em}
.stat-value{font-size:22px;font-weight:bold;margin:0}
.stat-sub{font-size:11px;color:#888;margin-top:4px}
.a-color{color:#2563eb}.b-color{color:#dc2626}.match{color:#059669}.warn{color:#d97706}
.chart-box{background:white;border:1px solid #e0e0e0;border-radius:8px;padding:24px;margin-bottom:24px}
.chart-title{font-size:14px;font-weight:600;margin-bottom:4px}
.chart-sub{font-size:12px;color:#888;margin-bottom:14px}
table{border-collapse:collapse;font-size:12px;width:100%}
th{background:#f4f4f4;text-align:left;padding:6px 10px;border-bottom:2px solid #ddd}
td{padding:5px 10px;border-bottom:1px solid #eee}
.pos{color:#dc2626}.neg{color:#2563eb}.zero{color:#059669}
"""


def _js_hist(
    svg_id: str,
    bins: list[int],
    a_pct: list[float],
    b_pct: list[float],
    x_label: str,
    label_a: str,
    label_b: str,
) -> str:
    return f"""
<script>
(function(){{
  const bins={json.dumps(bins)};
  const aV={json.dumps(a_pct)};
  const bV={json.dumps(b_pct)};
  const svg=document.getElementById({json.dumps(svg_id)});
  const W=860,H=280,pad={{l:52,r:16,t:16,b:48}};
  const cw=W-pad.l-pad.r,ch=H-pad.t-pad.b,n=bins.length,bW=cw/n;
  const maxY=Math.max(...aV,...bV)*1.12||1;
  const ns='http://www.w3.org/2000/svg';
  const el=(t,a)=>{{const e=document.createElementNS(ns,t);Object.entries(a||{{}}).forEach(([k,v])=>e.setAttribute(k,v));return e;}};
  for(let i=0;i<=4;i++){{
    const y=pad.t+(i/4)*ch;
    svg.appendChild(el('line',{{x1:pad.l,x2:W-pad.r,y1:y,y2:y,stroke:'#eee','stroke-width':1}}));
    const t=el('text',{{x:pad.l-4,y:y+4,'text-anchor':'end','font-size':9,fill:'#999'}});
    t.textContent=((4-i)/4*maxY).toFixed(1)+'%';svg.appendChild(t);
  }}
  for(let i=0;i<n;i++){{
    const x=pad.l+i*bW;
    [[bV,'rgba(220,38,38,0.45)'],[aV,'rgba(37,99,235,0.7)']].forEach(([v,c])=>{{
      const h=(v[i]/maxY)*ch;
      svg.appendChild(el('rect',{{x,y:pad.t+ch-h,width:bW-0.5,height:h,fill:c}}));
    }});
  }}
  const step=Math.max(1,Math.round(n/12));
  for(let i=0;i<n;i+=step){{
    const x=pad.l+i*bW+bW/2;
    const t=el('text',{{x,y:H-pad.b+13,'text-anchor':'middle','font-size':9,fill:'#555'}});
    t.textContent=bins[i];svg.appendChild(t);
  }}
  const yl=el('text',{{transform:`rotate(-90) translate(${{-(H/2)}},13)`,'text-anchor':'middle','font-size':10,fill:'#555'}});
  yl.textContent='% of requests';svg.appendChild(yl);
  const xl=el('text',{{x:W/2,y:H-3,'text-anchor':'middle','font-size':10,fill:'#555'}});
  xl.textContent={json.dumps(x_label)};svg.appendChild(xl);
  // legend
  [[{json.dumps(label_a)},'rgba(37,99,235,0.7)'],[{json.dumps(label_b)},'rgba(220,38,38,0.45)']].forEach(([lbl,c],i)=>{{
    svg.appendChild(el('rect',{{x:pad.l+i*180,y:4,width:12,height:10,fill:c}}));
    const t=el('text',{{x:pad.l+i*180+16,y:13,'font-size':10,fill:'#333'}});
    t.textContent=lbl;svg.appendChild(t);
  }});
}})();
</script>"""


def stat_card(title: str, value: str, sub: str, color_class: str = "") -> str:
    cls = f"stat-value {color_class}" if color_class else "stat-value"
    return (
        f'<div class="stat-card"><h3>{title}</h3>'
        f'<p class="{cls}">{value}</p>'
        f'<div class="stat-sub">{sub}</div></div>'
    )


def delta_class(d: float) -> str:
    if abs(d) < 0.5:
        return "zero"
    return "pos" if d > 0 else "neg"


def render_stat_section(
    stats_a: dict, stats_b: dict, label_a: str, label_b: str, field: str
) -> str:
    if not stats_a or not stats_b:
        return ""
    mean_d = round(stats_b["mean"] - stats_a["mean"], 2)
    std_d = round(stats_b["std"] - stats_a["std"], 2)
    dc = "match" if abs(mean_d) < 1 else "warn" if abs(mean_d) < 5 else "warn"
    cards = [
        stat_card(
            f"{label_a} mean",
            str(stats_a["mean"]),
            f"std={stats_a['std']}  p5={stats_a['p5']}, p95={stats_a['p95']}",
            "a-color",
        ),
        stat_card(
            f"{label_b} mean",
            str(stats_b["mean"]),
            f"std={stats_b['std']}  p5={stats_b['p5']}, p95={stats_b['p95']}",
            "b-color",
        ),
        stat_card("Mean delta", f"{mean_d:+.2f}", f"{label_b} − {label_a}", dc),
        stat_card("Std delta", f"{std_d:+.2f}", f"{label_b} − {label_a}", dc),
        stat_card(
            "p5 / p95",
            f"{stats_a['p5']} / {stats_a['p95']}",
            f"{label_a}: {stats_a['min']}–{stats_a['max']}",
            "",
        ),
        stat_card(
            "n", f"{stats_a['n']:,} / {stats_b['n']:,}", f"{label_a} / {label_b}", ""
        ),
    ]
    row2 = [
        stat_card(
            "Min", f"{stats_a['min']} / {stats_b['min']}", f"{label_a} / {label_b}"
        ),
        stat_card(
            "Max", f"{stats_a['max']} / {stats_b['max']}", f"{label_a} / {label_b}"
        ),
        stat_card("p5", f"{stats_a['p5']} / {stats_b['p5']}", f"{label_a} / {label_b}"),
        stat_card(
            "p95", f"{stats_a['p95']} / {stats_b['p95']}", f"{label_a} / {label_b}"
        ),
    ]
    return (
        f'<div class="stats-row">{"".join(cards)}</div>'
        f'<div class="stats-row">{"".join(row2)}</div>'
    )


def render_vocab_table(diffs: list[dict], label_a: str, label_b: str) -> str:
    if not diffs:
        return (
            "<p><em>No summary.json found — vocab distribution not available.</em></p>"
        )
    rows = []
    for d in diffs:
        dc = delta_class(d["delta"])
        rows.append(
            f"<tr><td>{d['id']}</td><td>{d['a']:,}</td><td>{d['b']:,}</td>"
            f"<td class='{dc}'>{d['delta']:+,}</td></tr>"
        )
    return (
        f"<table><thead><tr><th>Token ID</th><th>{label_a}</th>"
        f"<th>{label_b}</th><th>Delta (B−A)</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render_tokenization_mode_table(
    rows_a: list[dict], rows_b: list[dict], label_a: str, label_b: str
) -> str:
    def mode_counts(rows: list[dict]) -> dict[str, int]:
        c: dict[str, int] = collections.Counter()
        for r in rows:
            raw = r.get("tokenization_mode")
            c[raw if isinstance(raw, str) else "unknown"] += 1
        return dict(c)

    ca, cb = mode_counts(rows_a), mode_counts(rows_b)
    modes = sorted(set(ca) | set(cb))
    rows_html = []
    for m in modes:
        a_pct = ca.get(m, 0) / len(rows_a) * 100 if rows_a else 0
        b_pct = cb.get(m, 0) / len(rows_b) * 100 if rows_b else 0
        rows_html.append(
            f"<tr><td>{html.escape(m)}</td><td>{ca.get(m, 0):,} ({a_pct:.1f}%)</td>"
            f"<td>{cb.get(m, 0):,} ({b_pct:.1f}%)</td></tr>"
        )
    return (
        f"<table><thead><tr><th>Mode</th><th>{label_a}</th>"
        f"<th>{label_b}</th></tr></thead><tbody>{''.join(rows_html)}</tbody></table>"
    )


def build_html(
    rows_a: list[dict],
    rows_b: list[dict],
    summary_a: dict | None,
    summary_b: dict | None,
    label_a: str,
    label_b: str,
    path_a: str,
    path_b: str,
) -> str:
    isl_a = _finite_vals(rows_a, "isl")
    isl_b = _finite_vals(rows_b, "isl")
    osl_a = _finite_vals(rows_a, "requested_osl")
    osl_b = _finite_vals(rows_b, "requested_osl")

    st_isl_a = field_stats(rows_a, "isl")
    st_isl_b = field_stats(rows_b, "isl")
    st_osl_a = field_stats(rows_a, "requested_osl")
    st_osl_b = field_stats(rows_b, "requested_osl")

    _MAX_BINS = 500

    def hist_bounds(va: list, vb: list, base_step: int) -> tuple[int, int, int]:
        lo = (min(va + vb) // base_step) * base_step
        hi = (max(va + vb) // base_step + 2) * base_step
        n_bins = (hi - lo) // base_step
        step = base_step * max(1, math.ceil(n_bins / _MAX_BINS))
        hi = lo + step * math.ceil((hi - lo) / step)
        return lo, hi, step

    isl_lo, isl_hi, isl_step = hist_bounds(isl_a or [0], isl_b or [0], 2)
    osl_lo, osl_hi, osl_step = hist_bounds(osl_a or [0], osl_b or [0], 2)

    isl_bins = list(range(isl_lo, isl_hi, isl_step))
    osl_bins = list(range(osl_lo, osl_hi, osl_step))

    isl_a_pct = histogram(isl_a, isl_lo, isl_hi, isl_step)
    isl_b_pct = histogram(isl_b, isl_lo, isl_hi, isl_step)
    osl_a_pct = histogram(osl_a, osl_lo, osl_hi, osl_step)
    osl_b_pct = histogram(osl_b, osl_lo, osl_hi, osl_step)

    vocab_diffs = vocab_top_diffs(summary_a, summary_b)

    la = html.escape(label_a)
    lb = html.escape(label_b)
    pa = html.escape(path_a)
    pb = html.escape(path_b)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Recording comparison: {la} vs {lb}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Recording comparison: {la} vs {lb}</h1>
<div class="subtitle">
  A: {pa}<br>
  B: {pb}
</div>

<h2>Input Sequence Length (ISL)</h2>
{render_stat_section(st_isl_a, st_isl_b, la, lb, "isl")}
<div class="chart-box">
  <div class="chart-title">ISL Distribution (2-token bins)</div>
  <div class="chart-sub">
    Blue = {la} &nbsp;|&nbsp; Red = {lb} &nbsp;|&nbsp; Overlapping bars share color
  </div>
  <svg id="hist-isl" width="860" height="280" style="display:block;"></svg>
</div>
{_js_hist("hist-isl", isl_bins, isl_a_pct, isl_b_pct, "ISL (tokens)", la, lb)}

<h2>Requested Output Sequence Length (OSL)</h2>
{render_stat_section(st_osl_a, st_osl_b, la, lb, "requested_osl")}
<div class="chart-box">
  <div class="chart-title">Requested OSL Distribution (2-token bins)</div>
  <div class="chart-sub">
    Blue = {la} &nbsp;|&nbsp; Red = {lb} &nbsp;|&nbsp; Overlapping bars share color
  </div>
  <svg id="hist-osl" width="860" height="280" style="display:block;"></svg>
</div>
{_js_hist("hist-osl", osl_bins, osl_a_pct, osl_b_pct, "Requested OSL (tokens)", la, lb)}

<h2>Tokenization Mode</h2>
{render_tokenization_mode_table(rows_a, rows_b, la, lb)}

<h2>Vocab Distribution — Top {len(vocab_diffs)} token-count differences</h2>
{render_vocab_table(vocab_diffs, la, lb)}

</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--a", required=True, metavar="FILE", help="First recording JSONL (A)"
    )
    ap.add_argument(
        "--b", required=True, metavar="FILE", help="Second recording JSONL (B)"
    )
    ap.add_argument(
        "--label-a",
        default="A",
        metavar="LABEL",
        help="Display label for file A (default: A)",
    )
    ap.add_argument(
        "--label-b",
        default="B",
        metavar="LABEL",
        help="Display label for file B (default: B)",
    )
    ap.add_argument(
        "--out",
        default="comparison.html",
        metavar="FILE",
        help="Output HTML file (default: comparison.html)",
    )
    args = ap.parse_args()

    path_a, path_b = Path(args.a), Path(args.b)
    for p in (path_a, path_b):
        if not p.exists():
            sys.exit(f"File not found: {p}")

    print(f"Loading {path_a} ...", file=sys.stderr)
    rows_a = load_jsonl(path_a)
    print(f"Loading {path_b} ...", file=sys.stderr)
    rows_b = load_jsonl(path_b)
    summary_a = load_summary(path_a)
    summary_b = load_summary(path_b)

    if summary_a:
        print(f"  summary found: {path_a}.summary.json", file=sys.stderr)
    if summary_b:
        print(f"  summary found: {path_b}.summary.json", file=sys.stderr)

    print(f"Rows: A={len(rows_a)}, B={len(rows_b)}", file=sys.stderr)

    html = build_html(
        rows_a,
        rows_b,
        summary_a,
        summary_b,
        args.label_a,
        args.label_b,
        str(path_a),
        str(path_b),
    )

    out = Path(args.out)
    out.write_text(html)
    print(f"Written: {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
