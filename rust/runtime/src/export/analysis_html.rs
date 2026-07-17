// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Self-contained static HTML single-page report for the `--dry-run` dataset
//! analysis.
//!
//! [`render_analysis_html`] projects a [`DatasetAnalysis`] into one standalone
//! `.html` string: an agentx dark-theme stylesheet, the analysis embedded as
//! inline JSON (`const DATA = …;`), a D3 v7 `<script>` tag from CDN, and an
//! embedded rendering script that binds the DATA to headline stat tiles, the
//! ISL/OSL percentile bars, the turn-by-turn grouped bars, the prefix-cache
//! reuse charts, and the concurrency/throughput timeline. There is no server and
//! no build step — the file opens directly in a browser.
//!
//! [`write_dataset_analysis_html`] writes the rendered string to a path.
//!
//! The embedded JavaScript binds defensively: when `DATA.cache` or
//! `DATA.timeline` is `null` the corresponding section renders a muted "not
//! available" note rather than throwing.

use std::io::Write;
use std::path::Path;

use crate::dataset::analysis::DatasetAnalysis;

/// Render the full standalone HTML report for `a`.
///
/// The analysis is serialized with [`serde_json::to_string`] and embedded as
/// `const DATA = …;`. The sequence `</` is escaped to `<\/` in the serialized
/// JSON before embedding so a `</script>` occurring inside any string value can
/// never close the `<script>` tag early (mirrors agentx's `script_safe_json`).
///
/// Serialization is infallible for a [`DatasetAnalysis`] (every reachable `f64`
/// is guarded finite by `analyze`), so this returns a plain `String`.
pub fn render_analysis_html(a: &DatasetAnalysis) -> String {
    let json = serde_json::to_string(a).unwrap_or_else(|_| "null".to_string());
    let safe = json.replace("</", "<\\/");
    TEMPLATE.replace("__DATA_JSON__", &safe)
}

/// Write the rendered HTML report for `a` to `path` via a buffered writer.
///
/// A serialization failure (should never occur for a finite-guarded
/// [`DatasetAnalysis`]) is surfaced through [`render_analysis_html`], which
/// falls back to an empty `DATA`; any I/O error is returned. A latent serde
/// error would be mapped through [`std::io::Error::other`], but the current
/// path cannot produce one.
pub fn write_dataset_analysis_html(a: &DatasetAnalysis, path: &Path) -> std::io::Result<()> {
    let html = render_analysis_html(a);
    let file = std::fs::File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);
    writer.write_all(html.as_bytes())?;
    writer.flush()
}

/// The complete HTML document with a single `__DATA_JSON__` placeholder that
/// [`render_analysis_html`] substitutes with the script-safe serialized
/// analysis. Kept as one raw string (not `format!`) so the CSS/JS braces need no
/// escaping.
const TEMPLATE: &str = r####"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AIPerf Dry-Run Dataset Analysis</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
:root {
  --bg: #1a1a2e; --surface: #16213e; --border: #0f3460;
  --text: #e0e0e0; --muted: #8888aa;
  --green: #76b900; --green-dim: rgba(118,185,0,0.3);
  --blue: #4da6ff; --blue-dim: rgba(77,166,255,0.3);
  --gray: #555; --orange: #e89b00;
  --red: #ff4444; --red-dim: rgba(255,68,68,0.3); --magenta: #e040fb;
}
* { margin:0; padding:0; box-sizing:border-box; }
body { background:var(--bg); color:var(--text);
       font-family:'Segoe UI', system-ui, -apple-system, sans-serif; padding:20px; }
h1 { font-size:1.4rem; margin-bottom:16px; }
.chart-container { background:var(--surface); border:1px solid var(--border);
                   border-radius:8px; padding:16px; margin-bottom:16px; }
.chart-container h2 { font-size:0.95rem; color:var(--muted); margin-bottom:10px; font-weight:500; }
.stats { display:flex; flex-wrap:wrap; gap:24px; font-size:0.85rem; color:var(--muted);
         background:var(--surface); padding:10px 18px; border-radius:8px;
         border:1px solid var(--border); margin-bottom:16px; }
.stats .tile { display:flex; flex-direction:column; }
.stats .tile .val { color:var(--text); font-weight:600; font-size:1.15rem; }
.badge { display:inline-block; background:var(--green); color:#000; border-radius:4px;
         padding:2px 8px; font-size:0.75rem; font-weight:600; }
table { width:100%; border-collapse:collapse; font-size:0.82rem; }
th,td { text-align:right; padding:4px 8px; border-bottom:1px solid var(--border); }
th:first-child, td:first-child { text-align:left; color:var(--muted); }
.tooltip { position:fixed; background:rgba(22,33,62,0.95); border:1px solid var(--border);
           border-radius:4px; padding:8px 10px; font-size:0.8rem; pointer-events:none;
           display:none; z-index:100; line-height:1.5; }
.legend { display:flex; gap:16px; font-size:0.78rem; color:var(--muted); margin-bottom:6px; }
.legend .swatch { display:inline-block; width:11px; height:11px; border-radius:2px;
                  margin-right:5px; vertical-align:middle; }
.muted-note { color:var(--muted); font-size:0.85rem; font-style:italic; }
.grid2 { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
@media (max-width: 860px) { .grid2 { grid-template-columns:1fr; } }
</style>
</head>
<body>
<h1>AIPerf Dry-Run Dataset Analysis</h1>
<div id="stats" class="stats"></div>

<div class="chart-container">
  <h2>Sequence lengths &mdash; ISL vs OSL percentiles</h2>
  <div class="legend">
    <span><span class="swatch" style="background:var(--green)"></span>ISL</span>
    <span><span class="swatch" style="background:var(--blue)"></span>OSL</span>
  </div>
  <div id="len-chart"></div>
  <div id="len-table" style="margin-top:12px;"></div>
</div>

<div class="chart-container">
  <h2>Turn-by-turn &mdash; mean ISL/OSL and conversations reaching each turn</h2>
  <div class="legend">
    <span><span class="swatch" style="background:var(--green)"></span>mean ISL</span>
    <span><span class="swatch" style="background:var(--blue)"></span>mean OSL</span>
    <span><span class="swatch" style="background:var(--orange)"></span>conversations reaching</span>
  </div>
  <div id="turn-chart"></div>
  <div id="turn-table" style="margin-top:12px;"></div>
</div>

<div class="chart-container">
  <h2>Prefix-cache reuse</h2>
  <div id="cache-body"></div>
</div>

<div class="chart-container">
  <h2>Execution timeline</h2>
  <div id="timeline-body"></div>
</div>

<div class="tooltip" id="tooltip"></div>

<script>
const DATA = __DATA_JSON__;

const COL = { green:'#76b900', blue:'#4da6ff', orange:'#e89b00',
              red:'#ff4444', magenta:'#e040fb', border:'#0f3460', muted:'#8888aa' };
const TT = document.getElementById('tooltip');

function showTip(html, ev) {
  TT.innerHTML = html;
  TT.style.display = 'block';
  TT.style.left = (ev.clientX + 14) + 'px';
  TT.style.top = (ev.clientY + 14) + 'px';
}
function hideTip() { TT.style.display = 'none'; }

function num(x, d) {
  if (x === null || x === undefined || !isFinite(x)) return '—';
  return d3.format(',.' + (d === undefined ? 1 : d) + 'f')(x);
}
function intfmt(x) {
  if (x === null || x === undefined || !isFinite(x)) return '—';
  return d3.format(',')(Math.round(x));
}
function pctfmt(x) {
  if (x === null || x === undefined || !isFinite(x)) return '—';
  return (x * 100).toFixed(1) + '%';
}
function mutedNote(sel, msg) {
  d3.select(sel).html('<div class="muted-note">' + msg + '</div>');
}

// Style a rendered D3 axis group with the dark theme.
function styleAxis(g) {
  g.selectAll('path,line').attr('stroke', COL.border);
  g.selectAll('text').attr('fill', COL.muted).style('font-size', '10px');
}

// ---- Headline stat tiles -------------------------------------------------
function renderStats() {
  const tiles = [];
  const shape = DATA.shape || {};
  const lengths = DATA.lengths || {};
  tiles.push(['Conversations', intfmt(shape.conversations)]);
  tiles.push(['Total turns', intfmt(shape.total_turns)]);
  tiles.push(['Grand total tokens', intfmt(lengths.grand_total_tokens)]);

  if (DATA.cache) {
    tiles.push(['Ideal cache hit rate', pctfmt(DATA.cache.ideal && DATA.cache.ideal.hit_rate)]);
    const realized = DATA.cache.realized || [];
    if (realized.length) {
      const last = realized[realized.length - 1];
      tiles.push(['Realized @ ' + intfmt(last.capacity_blocks) + ' blk', pctfmt(last.hit_rate)]);
    }
  }
  if (DATA.timeline) {
    const c = DATA.timeline.concurrency || {};
    const t = DATA.timeline.throughput || {};
    tiles.push(['Peak concurrency', intfmt(c.peak)]);
    tiles.push(['Throughput', num(t.requests_per_s, 2) + ' req/s']);
    tiles.push(['Output tokens', num(t.output_tokens_per_s, 1) + ' tok/s']);
    tiles.push(['Run duration', num(t.run_duration_s, 2) + ' s']);
  }

  const root = d3.select('#stats');
  tiles.forEach(function (t) {
    const tile = root.append('div').attr('class', 'tile');
    tile.append('span').text(t[0]);
    tile.append('span').attr('class', 'val').text(t[1]);
  });
  if (DATA.cache && DATA.cache.identity_source) {
    const tile = root.append('div').attr('class', 'tile');
    tile.append('span').text('Identity source');
    tile.append('span').append('span').attr('class', 'badge')
        .text(DATA.cache.identity_source);
  }
}

// ---- Sequence-length percentile bars -------------------------------------
const PCTS = ['p1', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99'];

function renderLengths() {
  const isl = DATA.lengths && DATA.lengths.isl;
  const osl = DATA.lengths && DATA.lengths.osl;
  if (!isl && !osl) { mutedNote('#len-chart', 'No sequence-length data available.'); return; }

  const rows = PCTS.map(function (p) {
    return { pct: p, isl: isl ? isl[p] : null, osl: osl ? osl[p] : null };
  });

  const W = 720, H = 260, m = { top: 10, right: 12, bottom: 26, left: 48 };
  const iw = W - m.left - m.right, ih = H - m.top - m.bottom;
  const svg = d3.select('#len-chart').append('svg')
      .attr('viewBox', '0 0 ' + W + ' ' + H).attr('width', '100%');
  const g = svg.append('g').attr('transform', 'translate(' + m.left + ',' + m.top + ')');

  const x0 = d3.scaleBand().domain(PCTS).range([0, iw]).paddingInner(0.25);
  const x1 = d3.scaleBand().domain(['isl', 'osl']).range([0, x0.bandwidth()]).padding(0.1);
  const maxV = d3.max(rows, function (r) { return Math.max(r.isl || 0, r.osl || 0); }) || 1;
  const y = d3.scaleLinear().domain([0, maxV]).nice().range([ih, 0]);

  styleAxis(g.append('g').attr('transform', 'translate(0,' + ih + ')').call(d3.axisBottom(x0)));
  styleAxis(g.append('g').call(d3.axisLeft(y).ticks(5)));

  const series = [['isl', COL.green], ['osl', COL.blue]];
  series.forEach(function (s) {
    g.selectAll('.bar-' + s[0]).data(rows).enter().append('rect')
      .attr('x', function (r) { return x0(r.pct) + x1(s[0]); })
      .attr('width', x1.bandwidth())
      .attr('y', function (r) { return y(r[s[0]] || 0); })
      .attr('height', function (r) { return ih - y(r[s[0]] || 0); })
      .attr('fill', s[1])
      .on('mousemove', function (ev, r) {
        showTip(r.pct.toUpperCase() + '<br>' + s[0].toUpperCase() + ': ' + num(r[s[0]], 1), ev);
      })
      .on('mouseleave', hideTip);
  });

  renderLenTable();
}

function renderLenTable() {
  const cols = ['min', 'mean', 'p50', 'p90', 'p99', 'max'];
  const names = [['isl', 'ISL'], ['osl', 'OSL'], ['total', 'Total']];
  let html = '<table><thead><tr><th>metric</th>';
  cols.forEach(function (c) { html += '<th>' + c + '</th>'; });
  html += '</tr></thead><tbody>';
  names.forEach(function (n) {
    const s = DATA.lengths && DATA.lengths[n[0]];
    html += '<tr><td>' + n[1] + '</td>';
    cols.forEach(function (c) { html += '<td>' + (s ? num(s[c], 1) : '—') + '</td>'; });
    html += '</tr>';
  });
  html += '</tbody></table>';
  d3.select('#len-table').html(html);
}

// ---- Turn-by-turn grouped bars + reach line ------------------------------
function renderTurns() {
  const rows = (DATA.turns && DATA.turns.by_index) || [];
  if (!rows.length) { mutedNote('#turn-chart', 'No per-turn data available.'); return; }

  const W = 720, H = 280, m = { top: 10, right: 48, bottom: 26, left: 48 };
  const iw = W - m.left - m.right, ih = H - m.top - m.bottom;
  const svg = d3.select('#turn-chart').append('svg')
      .attr('viewBox', '0 0 ' + W + ' ' + H).attr('width', '100%');
  const g = svg.append('g').attr('transform', 'translate(' + m.left + ',' + m.top + ')');

  const idx = rows.map(function (r) { return r.turn_index; });
  const x0 = d3.scaleBand().domain(idx).range([0, iw]).paddingInner(0.25);
  const x1 = d3.scaleBand().domain(['isl', 'osl']).range([0, x0.bandwidth()]).padding(0.1);
  const meanOf = function (r, k) { return r[k] ? r[k].mean : 0; };
  const maxV = d3.max(rows, function (r) { return Math.max(meanOf(r, 'isl'), meanOf(r, 'osl')); }) || 1;
  const y = d3.scaleLinear().domain([0, maxV]).nice().range([ih, 0]);
  const maxReach = d3.max(rows, function (r) { return r.conversations_reaching || 0; }) || 1;
  const yR = d3.scaleLinear().domain([0, maxReach]).nice().range([ih, 0]);

  styleAxis(g.append('g').attr('transform', 'translate(0,' + ih + ')').call(d3.axisBottom(x0)));
  styleAxis(g.append('g').call(d3.axisLeft(y).ticks(5)));
  styleAxis(g.append('g').attr('transform', 'translate(' + iw + ',0)')
      .call(d3.axisRight(yR).ticks(5)));

  const series = [['isl', COL.green], ['osl', COL.blue]];
  series.forEach(function (s) {
    g.selectAll('.tb-' + s[0]).data(rows).enter().append('rect')
      .attr('x', function (r) { return x0(r.turn_index) + x1(s[0]); })
      .attr('width', x1.bandwidth())
      .attr('y', function (r) { return y(meanOf(r, s[0])); })
      .attr('height', function (r) { return ih - y(meanOf(r, s[0])); })
      .attr('fill', s[1])
      .on('mousemove', function (ev, r) {
        showTip('Turn ' + r.turn_index + '<br>mean ' + s[0].toUpperCase() + ': '
          + num(meanOf(r, s[0]), 1), ev);
      })
      .on('mouseleave', hideTip);
  });

  const line = d3.line()
      .x(function (r) { return x0(r.turn_index) + x0.bandwidth() / 2; })
      .y(function (r) { return yR(r.conversations_reaching || 0); });
  g.append('path').datum(rows).attr('fill', 'none')
      .attr('stroke', COL.orange).attr('stroke-width', 2).attr('d', line);
  g.selectAll('.reach-pt').data(rows).enter().append('circle')
      .attr('cx', function (r) { return x0(r.turn_index) + x0.bandwidth() / 2; })
      .attr('cy', function (r) { return yR(r.conversations_reaching || 0); })
      .attr('r', 3).attr('fill', COL.orange)
      .on('mousemove', function (ev, r) {
        showTip('Turn ' + r.turn_index + '<br>reaching: ' + intfmt(r.conversations_reaching), ev);
      })
      .on('mouseleave', hideTip);

  renderTurnTable(rows);
}

function renderTurnTable(rows) {
  let html = '<table><thead><tr><th>turn</th><th>reaching</th><th>mean ISL</th>'
    + '<th>mean OSL</th><th>hist growth</th><th>think ms</th></tr></thead><tbody>';
  rows.forEach(function (r) {
    html += '<tr><td>' + r.turn_index + '</td>'
      + '<td>' + intfmt(r.conversations_reaching) + '</td>'
      + '<td>' + (r.isl ? num(r.isl.mean, 1) : '—') + '</td>'
      + '<td>' + (r.osl ? num(r.osl.mean, 1) : '—') + '</td>'
      + '<td>' + (r.mean_history_growth == null ? '—' : num(r.mean_history_growth, 1)) + '</td>'
      + '<td>' + (r.authored_think_time_ms ? num(r.authored_think_time_ms.mean, 1) : '—') + '</td>'
      + '</tr>';
  });
  html += '</tbody></table>';
  d3.select('#turn-table').html(html);
}

// ---- Prefix-cache reuse --------------------------------------------------
function renderCache() {
  if (!DATA.cache) { mutedNote('#cache-body', 'Prefix-cache reuse not available for this dataset.'); return; }
  const c = DATA.cache;
  const ideal = c.ideal || {};

  const root = d3.select('#cache-body');
  root.html('');
  const strip = root.append('div').attr('class', 'stats').style('margin-bottom', '12px');
  const addTile = function (label, val) {
    const t = strip.append('div').attr('class', 'tile');
    t.append('span').text(label);
    t.append('span').attr('class', 'val').text(val);
  };
  addTile('Ideal hit rate', pctfmt(ideal.hit_rate));
  addTile('Total blocks', intfmt(ideal.total_blocks));
  addTile('Cached blocks', intfmt(ideal.cached_blocks));
  addTile('Unique blocks', intfmt(ideal.unique_blocks));
  addTile('Block size', intfmt(c.block_size));

  const grid = root.append('div').attr('class', 'grid2');
  const left = grid.append('div');
  const right = grid.append('div');

  // Intra vs cross two-segment bar.
  left.append('h2').style('font-size', '0.85rem').style('color', COL.muted)
      .text('Intra- vs cross-conversation cached blocks');
  const intra = ideal.intra_conversation_cached || 0;
  const cross = ideal.cross_conversation_cached || 0;
  const denom = (intra + cross) || 1;
  const barW = 320, barH = 26;
  const bsvg = left.append('svg').attr('viewBox', '0 0 ' + barW + ' ' + barH)
      .attr('width', '100%').style('max-width', barW + 'px');
  const iw2 = barW * intra / denom;
  bsvg.append('rect').attr('x', 0).attr('y', 0).attr('width', iw2).attr('height', barH)
      .attr('fill', COL.green)
      .on('mousemove', function (ev) { showTip('intra-conversation: ' + intfmt(intra), ev); })
      .on('mouseleave', hideTip);
  bsvg.append('rect').attr('x', iw2).attr('y', 0).attr('width', barW - iw2).attr('height', barH)
      .attr('fill', COL.blue)
      .on('mousemove', function (ev) { showTip('cross-conversation: ' + intfmt(cross), ev); })
      .on('mouseleave', hideTip);
  left.append('div').attr('class', 'legend').style('margin-top', '6px').html(
    '<span><span class="swatch" style="background:var(--green)"></span>intra ' + intfmt(intra) + '</span>'
    + '<span><span class="swatch" style="background:var(--blue)"></span>cross ' + intfmt(cross) + '</span>');

  // Shared-prefix-rate buckets.
  right.append('h2').style('font-size', '0.85rem').style('color', COL.muted)
      .text('Conversations with shared-prefix rate ≥ threshold');
  renderPrefixBuckets(right, ideal.shared_prefix_rate_ge || {});

  // Realized hit-rate vs capacity line chart.
  root.append('h2').style('font-size', '0.85rem').style('color', COL.muted)
      .style('margin-top', '14px').text('Realized hit rate vs LRU capacity');
  renderRealizedCurve(root, c.realized || []);
}

function renderPrefixBuckets(sel, buckets) {
  const keys = Object.keys(buckets).sort(function (a, b) { return (+a) - (+b); });
  if (!keys.length) { mutedNote(sel.node ? sel : sel, 'No shared-prefix buckets.'); return; }
  const rows = keys.map(function (k) { return { k: k, v: buckets[k] }; });
  const W = 340, H = 180, m = { top: 8, right: 8, bottom: 24, left: 34 };
  const iw = W - m.left - m.right, ih = H - m.top - m.bottom;
  const svg = sel.append('svg').attr('viewBox', '0 0 ' + W + ' ' + H).attr('width', '100%');
  const g = svg.append('g').attr('transform', 'translate(' + m.left + ',' + m.top + ')');
  const x = d3.scaleBand().domain(keys).range([0, iw]).padding(0.25);
  const maxV = d3.max(rows, function (r) { return r.v; }) || 1;
  const y = d3.scaleLinear().domain([0, maxV]).nice().range([ih, 0]);
  styleAxis(g.append('g').attr('transform', 'translate(0,' + ih + ')')
      .call(d3.axisBottom(x).tickFormat(function (d) { return '≥' + d + '%'; })));
  styleAxis(g.append('g').call(d3.axisLeft(y).ticks(4)));
  g.selectAll('.pb').data(rows).enter().append('rect')
    .attr('x', function (r) { return x(r.k); }).attr('width', x.bandwidth())
    .attr('y', function (r) { return y(r.v); }).attr('height', function (r) { return ih - y(r.v); })
    .attr('fill', COL.magenta)
    .on('mousemove', function (ev, r) { showTip('≥' + r.k + '%: ' + intfmt(r.v) + ' conv', ev); })
    .on('mouseleave', hideTip);
}

function renderRealizedCurve(sel, realized) {
  if (!realized.length) { mutedNote(sel.append('div'), 'No realized-capacity sweep available.'); return; }
  const pts = realized.slice().sort(function (a, b) { return a.capacity_blocks - b.capacity_blocks; });
  const W = 720, H = 240, m = { top: 10, right: 16, bottom: 30, left: 46 };
  const iw = W - m.left - m.right, ih = H - m.top - m.bottom;
  const svg = sel.append('svg').attr('viewBox', '0 0 ' + W + ' ' + H).attr('width', '100%');
  const g = svg.append('g').attr('transform', 'translate(' + m.left + ',' + m.top + ')');
  const x = d3.scaleLinear()
      .domain([0, d3.max(pts, function (p) { return p.capacity_blocks; }) || 1]).nice().range([0, iw]);
  const y = d3.scaleLinear().domain([0, 1]).range([ih, 0]);
  styleAxis(g.append('g').attr('transform', 'translate(0,' + ih + ')').call(d3.axisBottom(x).ticks(6)));
  styleAxis(g.append('g').call(d3.axisLeft(y).ticks(5)
      .tickFormat(function (d) { return (d * 100) + '%'; })));
  g.append('text').attr('x', iw).attr('y', ih + 26).attr('text-anchor', 'end')
      .attr('fill', COL.muted).style('font-size', '10px').text('capacity (blocks)');

  const line = d3.line()
      .x(function (p) { return x(p.capacity_blocks); })
      .y(function (p) { return y(p.hit_rate); });
  g.append('path').datum(pts).attr('fill', 'none').attr('stroke', COL.green)
      .attr('stroke-width', 2).attr('d', line);
  g.selectAll('.rc-pt').data(pts).enter().append('circle')
      .attr('cx', function (p) { return x(p.capacity_blocks); })
      .attr('cy', function (p) { return y(p.hit_rate); })
      .attr('r', 3).attr('fill', COL.green)
      .on('mousemove', function (ev, p) {
        showTip('capacity ' + intfmt(p.capacity_blocks) + ' blk<br>hit rate: '
          + pctfmt(p.hit_rate) + '<br>evictions: ' + intfmt(p.evictions), ev);
      })
      .on('mouseleave', hideTip);
}

// ---- Execution timeline --------------------------------------------------
function renderTimeline() {
  if (!DATA.timeline) { mutedNote('#timeline-body', 'Execution timeline not available (no records).'); return; }
  const t = DATA.timeline;
  const root = d3.select('#timeline-body');
  root.html('');

  const c = t.concurrency || {};
  const th = t.throughput || {};
  const strip = root.append('div').attr('class', 'stats').style('margin-bottom', '12px');
  const addTile = function (label, val) {
    const tt = strip.append('div').attr('class', 'tile');
    tt.append('span').text(label);
    tt.append('span').attr('class', 'val').text(val);
  };
  addTile('Peak concurrency', intfmt(c.peak));
  addTile('Time-weighted avg', num(c.time_weighted_avg, 2));
  addTile('Requests/s', num(th.requests_per_s, 2));
  addTile('Output tokens/s', num(th.output_tokens_per_s, 1));
  addTile('Run duration', num(th.run_duration_s, 2) + ' s');
  if (t.queue && t.queue.queue_delay_ms) {
    addTile('Queue delay p50', num(t.queue.queue_delay_ms.p50, 2) + ' ms');
  }

  root.append('h2').style('font-size', '0.85rem').style('color', COL.muted)
      .text('Inflight concurrency over time');
  renderConcurrency(root, c.samples || []);
}

function renderConcurrency(sel, samples) {
  if (!samples.length) { mutedNote(sel.append('div'), 'No concurrency samples available.'); return; }
  const pts = samples.map(function (s) { return { t: s[0], n: s[1] }; });
  const W = 720, H = 240, m = { top: 10, right: 16, bottom: 30, left: 40 };
  const iw = W - m.left - m.right, ih = H - m.top - m.bottom;
  const svg = sel.append('svg').attr('viewBox', '0 0 ' + W + ' ' + H).attr('width', '100%');
  const g = svg.append('g').attr('transform', 'translate(' + m.left + ',' + m.top + ')');
  const x = d3.scaleLinear()
      .domain([0, d3.max(pts, function (p) { return p.t; }) || 1]).nice().range([0, iw]);
  const y = d3.scaleLinear()
      .domain([0, d3.max(pts, function (p) { return p.n; }) || 1]).nice().range([ih, 0]);
  styleAxis(g.append('g').attr('transform', 'translate(0,' + ih + ')').call(d3.axisBottom(x).ticks(6)));
  styleAxis(g.append('g').call(d3.axisLeft(y).ticks(5)));
  g.append('text').attr('x', iw).attr('y', ih + 26).attr('text-anchor', 'end')
      .attr('fill', COL.muted).style('font-size', '10px').text('seconds');

  const area = d3.area().curve(d3.curveStepAfter)
      .x(function (p) { return x(p.t); })
      .y0(ih)
      .y1(function (p) { return y(p.n); });
  const line = d3.line().curve(d3.curveStepAfter)
      .x(function (p) { return x(p.t); })
      .y(function (p) { return y(p.n); });
  g.append('path').datum(pts).attr('fill', 'rgba(77,166,255,0.3)').attr('d', area);
  g.append('path').datum(pts).attr('fill', 'none').attr('stroke', COL.blue)
      .attr('stroke-width', 1.5).attr('d', line);

  // Transparent hover targets over each step.
  g.selectAll('.cc-pt').data(pts).enter().append('circle')
      .attr('cx', function (p) { return x(p.t); }).attr('cy', function (p) { return y(p.n); })
      .attr('r', 3).attr('fill', COL.blue).attr('opacity', 0.001)
      .on('mousemove', function (ev, p) {
        showTip('t = ' + num(p.t, 2) + ' s<br>inflight: ' + intfmt(p.n), ev);
      })
      .on('mouseleave', hideTip);
}

// ---- Bootstrap -----------------------------------------------------------
try { renderStats(); } catch (e) { console.error('stats', e); }
try { renderLengths(); } catch (e) { console.error('lengths', e); }
try { renderTurns(); } catch (e) { console.error('turns', e); }
try { renderCache(); } catch (e) { console.error('cache', e); }
try { renderTimeline(); } catch (e) { console.error('timeline', e); }
</script>
</body>
</html>
"####;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::analysis::*;

    /// Minimal end-to-end fixture mirroring Task 8's `tiny()`: one conversation,
    /// one turn carrying `block_ids`, one record. Exercises the shape, length,
    /// turn, cache, and timeline sections.
    fn tiny() -> DatasetAnalysis {
        let turns = vec![AnalyzedTurn {
            conversation_id: "a".into(),
            turn_index: 0,
            input_tokens: 32,
            max_output_tokens: 8,
            delay_ms: None,
            block_ids: Some(vec![1, 2]),
            system_handle: None,
        }];
        let records = vec![AnalyzedRecord {
            conversation_id: "a".into(),
            turn_index: 0,
            start_ns: 0,
            end_ns: 1_000_000_000,
            admit_ns: Some(0),
            first_token_ns: Some(0),
            input_tokens: 32,
            output_tokens: 8,
            token_arrival_ns: vec![],
        }];
        analyze(&turns, &records, &AnalysisOptions::default())
    }

    #[test]
    fn html_is_self_contained_and_bound() {
        let a = tiny();
        let html = render_analysis_html(&a);
        assert!(html.starts_with("<!DOCTYPE html>"));
        assert!(html.contains("--green: #76b900"));
        assert!(html.contains("https://d3js.org/d3.v7.min.js"));
        assert!(html.contains("const DATA ="));
        // embedded JSON escapes </ so no stray </script>
        assert!(!html.contains("</script></script>"));
        assert!(html.contains("AIPerf Dry-Run Dataset Analysis"));
        // data actually embedded
        assert!(html.contains("\"conversations\""));
    }

    #[test]
    fn write_html_creates_file() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("dataset_analysis.html");
        write_dataset_analysis_html(&tiny(), &p).unwrap();
        assert!(
            std::fs::read_to_string(&p)
                .unwrap()
                .starts_with("<!DOCTYPE html>")
        );
    }
}
