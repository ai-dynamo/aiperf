// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Self-contained static HTML single-page report for the `--dry-run` dataset
//! analysis.
//!
//! [`render_analysis_html`] projects a [`DatasetAnalysis`] into one standalone
//! `.html` string, restyled to the agentx `swim_lane_viewer` HUD theme: a dark
//! panelled stylesheet, the analysis embedded as inline JSON (`const DATA = …;`),
//! a D3 v7 `<script>` tag from CDN, and an embedded rendering script that binds
//! the DATA to a perf-engineer chart suite — headline stat tiles, ISL/OSL CDFs,
//! the prefill/decode split, prefix-cache hit-rate and eviction sweeps, reuse
//! savings, turn-by-turn growth, the concurrency timeline, the turn-depth funnel,
//! shared-prefix depth, think-time and queue-delay distributions, and a raw
//! statistics table. There is no server and no build step — the file opens
//! directly in a browser.
//!
//! [`write_dataset_analysis_html`] writes the rendered string to a path.
//!
//! The embedded JavaScript binds defensively: every renderer is wrapped in
//! `try/catch`, and when a section's DATA (`DATA.cache`, `DATA.timeline`, a
//! `StatSummary`, …) is `null` or empty the section renders a muted "not
//! available" note rather than throwing or blanking the page.

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
/// [`DatasetAnalysis`]) is absorbed by [`render_analysis_html`], which embeds
/// `null` as the DATA value; only I/O errors are returned.
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
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Oxanium:wght@600;700&family=IBM+Plex+Mono:wght@400;600&display=swap" rel="stylesheet">
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
:root{
  --bg:#0a0d13; --bg-grad-a:#0c1018; --bg-grad-b:#07090e;
  --panel:#10141d; --panel-edge:rgba(255,255,255,0.065); --hairline:rgba(255,255,255,0.07);
  --hairline-strong:rgba(255,255,255,0.16);
  --ink:#e8edf6; --ink-mute:#9aa6b8;
  --amber:#f5b942; --amber-soft:rgba(245,185,66,0.14);
  --cyan:#56c8dd; --red:#ef6a6a; --green:#57c994; --blue:#62a0ea; --violet:#b18cf2;
  --mono:"IBM Plex Mono",ui-monospace,"SF Mono",Menlo,monospace;
  --display:"Oxanium","IBM Plex Mono",ui-monospace,monospace;
}
*{margin:0;padding:0;box-sizing:border-box;}
body{
  background:
    radial-gradient(1200px 600px at 70% -10%, rgba(86,200,221,0.05), transparent 60%),
    radial-gradient(900px 500px at 0% 110%, rgba(245,185,66,0.04), transparent 55%),
    linear-gradient(180deg,var(--bg-grad-a),var(--bg-grad-b));
  background-color:var(--bg); color:var(--ink); font-family:var(--mono); padding:24px;
  min-height:100vh;
}
.film{position:fixed;inset:0;pointer-events:none;opacity:.3;z-index:0;mix-blend-mode:overlay;}
.wrap{position:relative;z-index:1;max-width:1180px;margin:0 auto;}
h1{font-family:var(--display);font-size:1.5rem;font-weight:700;letter-spacing:.01em;
   display:flex;align-items:center;gap:10px;margin-bottom:6px;}
.brand-dot{width:11px;height:11px;border-radius:50%;background:var(--amber);
   box-shadow:0 0 0 0 rgba(245,185,66,.5);animation:pulse 2.4s infinite;}
@keyframes pulse{
  0%{box-shadow:0 0 0 0 rgba(245,185,66,.45);}
  70%{box-shadow:0 0 0 8px rgba(245,185,66,0);}
  100%{box-shadow:0 0 0 0 rgba(245,185,66,0);}
}
.subtitle{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:18px;align-items:center;}
.chip{font-size:.72rem;color:var(--ink-mute);background:var(--panel);
  border:1px solid var(--panel-edge);border-radius:4px;padding:3px 9px;letter-spacing:.03em;}
.chip b{color:var(--ink);font-weight:600;}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;margin-bottom:16px;}
.tile{background:var(--panel);border:1px solid var(--panel-edge);border-radius:4px;
  padding:12px 16px;display:flex;flex-direction:column;gap:4px;
  box-shadow:0 18px 50px -22px rgba(0,0,0,.8);}
.tile .label{color:var(--ink-mute);font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;}
.tile .value{color:var(--ink);font-family:var(--display);font-size:1.4rem;font-weight:600;line-height:1.1;}
.tile .sub{color:var(--ink-mute);font-size:.72rem;}
.panel{background:var(--panel);border:1px solid var(--panel-edge);border-radius:8px;
  padding:16px;margin-bottom:16px;box-shadow:0 18px 50px -22px rgba(0,0,0,.8);}
.panel-title{font-family:var(--display);color:var(--ink-mute);letter-spacing:.04em;
  text-transform:uppercase;font-size:.8rem;font-weight:600;margin-bottom:12px;}
.panel-sub{color:var(--ink-mute);font-size:.74rem;margin-bottom:10px;margin-top:-6px;}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
.grid2 > .panel{margin-bottom:0;}
.grid2-row{margin-bottom:16px;}
@media (max-width:880px){.grid2{grid-template-columns:1fr;}.grid2 > .panel{margin-bottom:16px;}}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:.74rem;color:var(--ink-mute);margin-bottom:8px;}
.legend .swatch{display:inline-block;width:11px;height:11px;border-radius:2px;
  margin-right:6px;vertical-align:middle;}
.muted-note{color:var(--ink-mute);font-size:.82rem;font-style:italic;padding:14px 2px;}
.tooltip{position:fixed;background:rgba(13,17,25,0.96);border:1px solid var(--amber-soft);
  border-radius:6px;padding:8px 11px;font-size:.76rem;pointer-events:none;display:none;
  z-index:100;line-height:1.55;color:var(--ink);backdrop-filter:blur(6px);
  box-shadow:0 12px 34px -12px rgba(0,0,0,.9);}
.tooltip b{color:var(--amber);font-weight:600;}
table{width:100%;border-collapse:collapse;font-size:.78rem;}
th,td{text-align:right;padding:5px 9px;border-bottom:1px solid var(--hairline);}
th{color:var(--ink-mute);font-weight:600;text-transform:uppercase;font-size:.68rem;letter-spacing:.05em;}
th:first-child,td:first-child{text-align:left;color:var(--ink-mute);}
td{color:var(--ink);}
svg text{font-family:var(--mono);}
</style>
</head>
<body>
<svg class="film" width="100%" height="100%"><filter id="noise"><feTurbulence type="fractalNoise" baseFrequency="0.8" numOctaves="2" stitchTiles="stitch"/></filter><rect width="100%" height="100%" filter="url(#noise)"/></svg>
<div class="wrap">
<h1><span class="brand-dot"></span>AIPerf Dry-Run Dataset Analysis</h1>
<div id="subtitle" class="subtitle"></div>
<div id="stats" class="tiles"></div>

<div class="grid2 grid2-row">
  <div class="panel"><div class="panel-title">ISL Distribution &middot; CDF</div><div id="isl-cdf"></div></div>
  <div class="panel"><div class="panel-title">OSL Distribution &middot; CDF</div><div id="osl-cdf"></div></div>
</div>

<div class="panel"><div class="panel-title">Prefill vs Decode Token Split</div><div id="split"></div></div>

<div class="grid2 grid2-row">
  <div class="panel"><div class="panel-title">Prefix-Cache Hit Rate vs Capacity</div><div id="hitrate"></div></div>
  <div class="panel"><div class="panel-title">Evictions vs Capacity</div><div id="evictions"></div></div>
</div>

<div class="panel"><div class="panel-title">Reuse Savings</div><div id="savings"></div></div>

<div class="panel"><div class="panel-title">Turn-by-Turn ISL / OSL &amp; History Growth</div><div id="turns"></div></div>

<div class="panel"><div class="panel-title">Concurrency Over Time</div><div id="concurrency"></div></div>

<div class="grid2 grid2-row">
  <div class="panel"><div class="panel-title">Turn-Depth Funnel</div><div id="funnel"></div></div>
  <div class="panel"><div class="panel-title">Shared-Prefix Depth</div><div id="prefixdepth"></div></div>
</div>

<div class="grid2 grid2-row">
  <div class="panel"><div class="panel-title">Think-Time Distribution</div><div id="thinktime"></div></div>
  <div class="panel"><div class="panel-title">Queue-Delay Distribution</div><div id="queuedelay"></div></div>
</div>

<div class="panel"><div class="panel-title">Raw Statistics</div><div id="rawtables"></div></div>
</div>

<div class="tooltip" id="tooltip"></div>

<script>
const DATA = __DATA_JSON__;

const COL = {
  bg:'#10141d', ink:'#e8edf6', mute:'#9aa6b8',
  hair:'rgba(255,255,255,0.07)', hairStrong:'rgba(255,255,255,0.16)',
  amber:'#f5b942', cyan:'#56c8dd', red:'#ef6a6a',
  green:'#57c994', blue:'#62a0ea', violet:'#b18cf2'
};
const TT = document.getElementById('tooltip');

function showTip(html, ev){
  TT.innerHTML = html;
  TT.style.display = 'block';
  let x = ev.clientX + 14, y = ev.clientY + 14;
  const r = TT.getBoundingClientRect();
  if (x + r.width > window.innerWidth) x = ev.clientX - r.width - 14;
  if (y + r.height > window.innerHeight) y = ev.clientY - r.height - 14;
  TT.style.left = x + 'px';
  TT.style.top = y + 'px';
}
function hideTip(){ TT.style.display = 'none'; }

function num(x, d){
  if (x === null || x === undefined || !isFinite(x)) return '—';
  return d3.format(',.' + (d === undefined ? 1 : d) + 'f')(x);
}
function intfmt(x){
  if (x === null || x === undefined || !isFinite(x)) return '—';
  return d3.format(',')(Math.round(x));
}
function pctfmt(x){
  if (x === null || x === undefined || !isFinite(x)) return '—';
  return (x * 100).toFixed(1) + '%';
}
// Escape text destined for innerHTML (conversation ids come from the dataset).
function esc(s){
  return String(s == null ? '' : s).replace(/[&<>"']/g, function(c){
    return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c];
  });
}
function mutedNote(sel, msg){
  d3.select(sel).html('<div class="muted-note">' + msg + '</div>');
}
// Style a rendered D3 axis group with the HUD theme (recessive lines/ticks).
function styleAxis(g){
  g.selectAll('.domain').attr('stroke', COL.hair);
  g.selectAll('.tick line').attr('stroke', COL.hair);
  g.selectAll('text').attr('fill', COL.mute).style('font-size', '10px');
}
// Faint horizontal grid lines from a linear/band y-scale.
function gridY(g, y, iw, ticks){
  g.append('g').attr('class', 'grid')
    .call(d3.axisLeft(y).ticks(ticks || 5).tickSize(-iw).tickFormat(''))
    .call(function(s){ s.selectAll('.domain').remove();
      s.selectAll('.tick line').attr('stroke', COL.hair).attr('stroke-dasharray', '2,3'); });
}
function makeSvg(sel, W, H){
  return d3.select(sel).append('svg')
    .attr('viewBox', '0 0 ' + W + ' ' + H)
    .attr('width', '100%').attr('preserveAspectRatio', 'xMinYMin meet');
}
// Reusable crosshair+tooltip over a plot area. `xPix`/`yPix` map a datum to
// pixels; `tip` returns tooltip HTML. Snaps to the nearest datum in x.
function crosshair(g, iw, ih, data, xPix, yPix, tip, color){
  const vline = g.append('line')
    .attr('y1', 0).attr('y2', ih)
    .attr('stroke', COL.hairStrong).attr('stroke-dasharray', '3,3')
    .style('display', 'none').style('pointer-events', 'none');
  const dot = g.append('circle').attr('r', 4)
    .attr('fill', color || COL.amber).attr('stroke', COL.bg).attr('stroke-width', 1.5)
    .style('display', 'none').style('pointer-events', 'none');
  g.append('rect').attr('width', iw).attr('height', ih).attr('fill', 'transparent')
    .on('mousemove', function(ev){
      const mx = d3.pointer(ev, this)[0];
      let best = null, bd = Infinity;
      data.forEach(function(d){ const px = xPix(d); const dd = Math.abs(px - mx);
        if (dd < bd){ bd = dd; best = d; } });
      if (best === null) return;
      const px = xPix(best), py = yPix(best);
      vline.attr('x1', px).attr('x2', px).style('display', null);
      dot.attr('cx', px).attr('cy', py).style('display', null);
      showTip(tip(best), ev);
    })
    .on('mouseleave', function(){ vline.style('display', 'none'); dot.style('display', 'none'); hideTip(); });
}

// ---- Subtitle chips ------------------------------------------------------
function renderSubtitle(){
  const root = d3.select('#subtitle');
  const shape = DATA.shape || {};
  const cache = DATA.cache;
  if (cache && cache.identity_source)
    root.append('span').attr('class', 'chip').html('identity <b>' + cache.identity_source + '</b>');
  if (cache && cache.block_size != null)
    root.append('span').attr('class', 'chip').html('block size <b>' + intfmt(cache.block_size) + '</b>');
  const models = shape.models || [];
  if (models.length){
    models.forEach(function(m){ root.append('span').attr('class', 'chip').html('<b>' + m + '</b>'); });
  } else {
    root.append('span').attr('class', 'chip').html('models <b>unspecified</b>');
  }
}

// ---- Headline stat tiles -------------------------------------------------
function addTile(root, label, value, sub){
  const t = root.append('div').attr('class', 'tile');
  t.append('span').attr('class', 'label').text(label);
  t.append('span').attr('class', 'value').text(value);
  if (sub) t.append('span').attr('class', 'sub').text(sub);
}
function renderStats(){
  const root = d3.select('#stats');
  const shape = DATA.shape || {};
  const lengths = DATA.lengths || {};
  const isl = lengths.isl, osl = lengths.osl;

  addTile(root, 'Conversations', intfmt(shape.conversations),
    intfmt(shape.total_turns) + ' turns');
  const conv = shape.conversations || 0;
  const mt = shape.multi_turn_conversations || 0;
  addTile(root, 'Multi-turn', conv ? pctfmt(mt / conv) : '—',
    intfmt(mt) + ' conversations');
  addTile(root, 'ISL (med / p95)',
    isl ? intfmt(isl.p50) : '—',
    isl ? 'p95 ' + intfmt(isl.p95) : 'n/a');
  addTile(root, 'OSL (med / p95)',
    osl ? intfmt(osl.p50) : '—',
    osl ? 'p95 ' + intfmt(osl.p95) : 'n/a');
  const gt = lengths.grand_total_tokens || 0;
  const pf = gt ? (lengths.total_prompt_tokens || 0) / gt : null;
  addTile(root, 'Prefill Fraction', pf === null ? '—' : pctfmt(pf),
    intfmt(lengths.total_prompt_tokens) + ' prompt tok');
  if (DATA.cache && DATA.cache.ideal)
    addTile(root, 'Ideal Hit Rate', pctfmt(DATA.cache.ideal.hit_rate),
      intfmt(DATA.cache.ideal.cached_blocks) + ' cached blk');
  else
    addTile(root, 'Ideal Hit Rate', '—', 'no cache data');
  addTile(root, 'Grand Total Tokens', intfmt(gt),
    intfmt(lengths.total_completion_tokens) + ' completion tok');
  if (DATA.timeline){
    const th = DATA.timeline.throughput || {};
    const c = DATA.timeline.concurrency || {};
    addTile(root, 'Throughput', num(th.requests_per_s, 2) + ' req/s',
      num(th.output_tokens_per_s, 1) + ' tok/s');
    addTile(root, 'Run Duration', num(th.run_duration_s, 2) + ' s',
      'peak ' + intfmt(c.peak) + ' inflight');
  }
}

// ---- 1. ISL / OSL CDF pair ----------------------------------------------
// Build a step-CDF point ladder from a StatSummary percentile ladder.
function cdfLadder(s){
  if (!s) return [];
  const raw = [
    [s.min, 0], [s.p1, .01], [s.p5, .05], [s.p10, .10], [s.p25, .25],
    [s.p50, .50], [s.p75, .75], [s.p90, .90], [s.p95, .95], [s.p99, .99], [s.max, 1]
  ];
  // Keep monotone-nondecreasing x; drop non-finite.
  const pts = [];
  raw.forEach(function(p){
    if (p[0] === null || p[0] === undefined || !isFinite(p[0])) return;
    if (pts.length && p[0] < pts[pts.length - 1].x) p[0] = pts[pts.length - 1].x;
    pts.push({ x: p[0], y: p[1] * 100 });
  });
  return pts;
}
function renderCDF(sel, stat, color, unit){
  const pts = cdfLadder(stat);
  if (pts.length < 2){ mutedNote(sel, 'No distribution data available.'); return; }
  const W = 540, H = 250, m = { top: 12, right: 16, bottom: 34, left: 46 };
  const iw = W - m.left - m.right, ih = H - m.top - m.bottom;
  const g = makeSvg(sel, W, H).append('g')
    .attr('transform', 'translate(' + m.left + ',' + m.top + ')');
  const x = d3.scaleLinear().domain([pts[0].x, pts[pts.length - 1].x]).nice().range([0, iw]);
  const y = d3.scaleLinear().domain([0, 100]).range([ih, 0]);
  gridY(g, y, iw, 5);
  styleAxis(g.append('g').attr('transform', 'translate(0,' + ih + ')')
    .call(d3.axisBottom(x).ticks(6)));
  styleAxis(g.append('g').call(d3.axisLeft(y).ticks(5)
    .tickFormat(function(d){ return d + '%'; })));
  g.append('text').attr('x', iw).attr('y', ih + 30).attr('text-anchor', 'end')
    .attr('fill', COL.mute).style('font-size', '10px').text(unit);

  // Reference ticks at p50/p95/p99.
  [['p50', stat.p50], ['p95', stat.p95], ['p99', stat.p99]].forEach(function(r){
    if (r[1] == null || !isFinite(r[1])) return;
    g.append('line').attr('x1', x(r[1])).attr('x2', x(r[1]))
      .attr('y1', 0).attr('y2', ih).attr('stroke', COL.amber).attr('stroke-opacity', .32)
      .attr('stroke-dasharray', '2,3');
    g.append('text').attr('x', x(r[1])).attr('y', 11).attr('text-anchor', 'middle')
      .attr('fill', COL.mute).style('font-size', '9px').text(r[0]);
  });

  const area = d3.area().curve(d3.curveStepAfter)
    .x(function(p){ return x(p.x); }).y0(ih).y1(function(p){ return y(p.y); });
  const line = d3.line().curve(d3.curveStepAfter)
    .x(function(p){ return x(p.x); }).y(function(p){ return y(p.y); });
  const grad = 'g-' + sel.replace(/[^a-z0-9]/gi, '');
  const defs = g.append('defs').append('linearGradient').attr('id', grad)
    .attr('x1', 0).attr('y1', 0).attr('x2', 0).attr('y2', 1);
  defs.append('stop').attr('offset', '0%').attr('stop-color', color).attr('stop-opacity', .28);
  defs.append('stop').attr('offset', '100%').attr('stop-color', color).attr('stop-opacity', .03);
  g.append('path').datum(pts).attr('fill', 'url(#' + grad + ')').attr('d', area);
  g.append('path').datum(pts).attr('fill', 'none').attr('stroke', color)
    .attr('stroke-width', 2).attr('d', line);

  crosshair(g, iw, ih, pts,
    function(p){ return x(p.x); }, function(p){ return y(p.y); },
    function(p){ return '<b>' + intfmt(p.x) + '</b> ' + unit + '<br>' + p.y.toFixed(1) + '% cumulative'; },
    color);
}
function renderCDFs(){
  const l = DATA.lengths || {};
  renderCDF('#isl-cdf', l.isl, COL.blue, 'input tokens');
  renderCDF('#osl-cdf', l.osl, COL.amber, 'output tokens');
}

// ---- 2. Prefill vs decode split -----------------------------------------
function renderSplit(){
  const l = DATA.lengths || {};
  const prefill = l.total_prompt_tokens || 0;
  const decode = l.total_completion_tokens || 0;
  const total = prefill + decode;
  const sel = '#split';
  if (!total){ mutedNote(sel, 'No token-budget data available.'); return; }
  const frac = prefill / total;
  d3.select(sel).append('div').attr('class', 'panel-sub')
    .text('Prefill fraction ' + pctfmt(frac) + ' of ' + intfmt(total) + ' total tokens');
  d3.select(sel).append('div').attr('class', 'legend').html(
    '<span><span class="swatch" style="background:' + COL.blue + '"></span>prefill (prompt)</span>' +
    '<span><span class="swatch" style="background:' + COL.amber + '"></span>decode (completion)</span>');

  const W = 900, H = 66, m = { left: 2, right: 2 };
  const iw = W - m.left - m.right;
  const g = makeSvg(sel, W, H).append('g').attr('transform', 'translate(' + m.left + ',6)');
  const gap = 2, barH = 34, r = 4;
  const wP = Math.max(0, iw * frac - gap / 2);
  const wD = Math.max(0, iw * (1 - frac) - gap / 2);
  // Rounded outer ends via clip-friendly rects (round both ends slightly).
  g.append('rect').attr('x', 0).attr('y', 0).attr('width', wP).attr('height', barH)
    .attr('rx', r).attr('fill', COL.blue)
    .on('mousemove', function(ev){ showTip('<b>Prefill</b><br>' + intfmt(prefill) + ' tok &middot; ' + pctfmt(frac), ev); })
    .on('mouseleave', hideTip);
  g.append('rect').attr('x', wP + gap).attr('y', 0).attr('width', wD).attr('height', barH)
    .attr('rx', r).attr('fill', COL.amber)
    .on('mousemove', function(ev){ showTip('<b>Decode</b><br>' + intfmt(decode) + ' tok &middot; ' + pctfmt(1 - frac), ev); })
    .on('mouseleave', hideTip);
  // In-segment labels (ink, never the series color).
  g.append('text').attr('x', 8).attr('y', barH / 2 + 4).attr('fill', COL.bg)
    .style('font-size', '12px').style('font-weight', 600)
    .text(intfmt(prefill) + ' (' + pctfmt(frac) + ')');
  g.append('text').attr('x', wP + gap + wD - 8).attr('y', barH / 2 + 4)
    .attr('text-anchor', 'end').attr('fill', COL.bg)
    .style('font-size', '12px').style('font-weight', 600)
    .text(intfmt(decode) + ' (' + pctfmt(1 - frac) + ')');
}

// ---- 3. Hit-rate vs capacity (log-x line) -------------------------------
function renderHitRate(){
  const sel = '#hitrate';
  const cache = DATA.cache;
  if (!cache){ mutedNote(sel, 'Prefix-cache reuse not available for this dataset.'); return; }
  const realized = (cache.realized || []).filter(function(p){ return p.capacity_blocks > 0; })
    .slice().sort(function(a, b){ return a.capacity_blocks - b.capacity_blocks; });
  if (!realized.length){ mutedNote(sel, 'No realized-capacity sweep available.'); return; }
  const ideal = (cache.ideal && cache.ideal.hit_rate) || 0;

  d3.select(sel).append('div').attr('class', 'legend').html(
    '<span><span class="swatch" style="background:' + COL.blue + '"></span>realized hit rate</span>' +
    '<span><span class="swatch" style="background:' + COL.amber + '"></span>ideal ceiling</span>');

  const W = 540, H = 250, m = { top: 12, right: 16, bottom: 34, left: 46 };
  const iw = W - m.left - m.right, ih = H - m.top - m.bottom;
  const g = makeSvg(sel, W, H).append('g')
    .attr('transform', 'translate(' + m.left + ',' + m.top + ')');
  const xmin = d3.min(realized, function(p){ return p.capacity_blocks; });
  const xmax = d3.max(realized, function(p){ return p.capacity_blocks; });
  const x = d3.scaleLog().domain([Math.max(1, xmin), Math.max(2, xmax)]).range([0, iw]);
  const y = d3.scaleLinear().domain([0, Math.max(100, ideal * 100)]).nice().range([ih, 0]);
  gridY(g, y, iw, 5);
  styleAxis(g.append('g').attr('transform', 'translate(0,' + ih + ')')
    .call(d3.axisBottom(x).ticks(6, '~s')));
  styleAxis(g.append('g').call(d3.axisLeft(y).ticks(5)
    .tickFormat(function(d){ return d + '%'; })));
  g.append('text').attr('x', iw).attr('y', ih + 30).attr('text-anchor', 'end')
    .attr('fill', COL.mute).style('font-size', '10px').text('capacity (blocks, log)');

  // Ideal ceiling reference line.
  g.append('line').attr('x1', 0).attr('x2', iw)
    .attr('y1', y(ideal * 100)).attr('y2', y(ideal * 100))
    .attr('stroke', COL.amber).attr('stroke-dasharray', '5,4').attr('stroke-width', 1.5);
  g.append('text').attr('x', iw - 4).attr('y', y(ideal * 100) - 5).attr('text-anchor', 'end')
    .attr('fill', COL.mute).style('font-size', '10px')
    .text('ideal ceiling ' + pctfmt(ideal));

  const line = d3.line()
    .x(function(p){ return x(p.capacity_blocks); })
    .y(function(p){ return y(p.hit_rate * 100); });
  g.append('path').datum(realized).attr('fill', 'none').attr('stroke', COL.blue)
    .attr('stroke-width', 2).attr('d', line);
  g.selectAll('.hr-pt').data(realized).enter().append('circle')
    .attr('cx', function(p){ return x(p.capacity_blocks); })
    .attr('cy', function(p){ return y(p.hit_rate * 100); })
    .attr('r', 4.5).attr('fill', COL.blue).attr('stroke', COL.bg).attr('stroke-width', 1.5);
  crosshair(g, iw, ih, realized,
    function(p){ return x(p.capacity_blocks); },
    function(p){ return y(p.hit_rate * 100); },
    function(p){ return '<b>' + intfmt(p.capacity_blocks) + '</b> blk capacity<br>' +
      pctfmt(p.hit_rate) + ' hit rate<br>' + intfmt(p.evictions) + ' evictions'; },
    COL.blue);
}

// ---- 4. Evictions vs capacity (separate chart, shared log-x) ------------
function renderEvictions(){
  const sel = '#evictions';
  const cache = DATA.cache;
  if (!cache){ mutedNote(sel, 'Prefix-cache reuse not available for this dataset.'); return; }
  const realized = (cache.realized || []).filter(function(p){ return p.capacity_blocks > 0; })
    .slice().sort(function(a, b){ return a.capacity_blocks - b.capacity_blocks; });
  if (!realized.length){ mutedNote(sel, 'No realized-capacity sweep available.'); return; }

  const W = 540, H = 250, m = { top: 12, right: 16, bottom: 34, left: 52 };
  const iw = W - m.left - m.right, ih = H - m.top - m.bottom;
  const g = makeSvg(sel, W, H).append('g')
    .attr('transform', 'translate(' + m.left + ',' + m.top + ')');
  const xmin = d3.min(realized, function(p){ return p.capacity_blocks; });
  const xmax = d3.max(realized, function(p){ return p.capacity_blocks; });
  const x = d3.scaleLog().domain([Math.max(1, xmin), Math.max(2, xmax)]).range([0, iw]);
  const ymax = d3.max(realized, function(p){ return p.evictions; }) || 1;
  const y = d3.scaleLinear().domain([0, ymax]).nice().range([ih, 0]);
  gridY(g, y, iw, 5);
  styleAxis(g.append('g').attr('transform', 'translate(0,' + ih + ')')
    .call(d3.axisBottom(x).ticks(6, '~s')));
  styleAxis(g.append('g').call(d3.axisLeft(y).ticks(5, '~s')));
  g.append('text').attr('x', iw).attr('y', ih + 30).attr('text-anchor', 'end')
    .attr('fill', COL.mute).style('font-size', '10px').text('capacity (blocks, log)');
  g.append('text').attr('x', 2).attr('y', -2)
    .attr('fill', COL.mute).style('font-size', '10px').text('evictions');

  const line = d3.line()
    .x(function(p){ return x(p.capacity_blocks); })
    .y(function(p){ return y(p.evictions); });
  g.append('path').datum(realized).attr('fill', 'none').attr('stroke', COL.red)
    .attr('stroke-width', 2).attr('stroke-opacity', .85).attr('d', line);
  g.selectAll('.ev-pt').data(realized).enter().append('circle')
    .attr('cx', function(p){ return x(p.capacity_blocks); })
    .attr('cy', function(p){ return y(p.evictions); })
    .attr('r', 4).attr('fill', COL.red).attr('stroke', COL.bg).attr('stroke-width', 1.5);
  crosshair(g, iw, ih, realized,
    function(p){ return x(p.capacity_blocks); },
    function(p){ return y(p.evictions); },
    function(p){ return '<b>' + intfmt(p.capacity_blocks) + '</b> blk capacity<br>' +
      intfmt(p.evictions) + ' evictions<br>' + pctfmt(p.hit_rate) + ' hit rate'; },
    COL.red);
}

// ---- 5. Reuse savings (stacked horizontal bar) --------------------------
function renderSavings(){
  const sel = '#savings';
  const cache = DATA.cache;
  if (!cache || !cache.ideal){ mutedNote(sel, 'Prefix-cache reuse not available for this dataset.'); return; }
  const id = cache.ideal;
  const total = id.total_blocks || 0;
  if (!total){ mutedNote(sel, 'No blocks to analyze.'); return; }
  const intra = id.intra_conversation_cached || 0;
  const cross = id.cross_conversation_cached || 0;
  const cached = id.cached_blocks || (intra + cross);
  const uncached = Math.max(0, total - cached);
  const blockSize = cache.block_size || 0;
  const savedTokens = cached * blockSize;

  d3.select(sel).append('div').attr('class', 'panel-sub').html(
    'Savings ' + pctfmt(cached / total) + ' &middot; ' + intfmt(cached) + ' of ' +
    intfmt(total) + ' blocks reused &middot; ~' + intfmt(savedTokens) + ' prefill tokens saved');
  d3.select(sel).append('div').attr('class', 'legend').html(
    '<span><span class="swatch" style="background:' + COL.green + '"></span>intra-conversation ' + intfmt(intra) + '</span>' +
    '<span><span class="swatch" style="background:' + COL.violet + '"></span>cross-conversation ' + intfmt(cross) + '</span>' +
    '<span><span class="swatch" style="background:rgba(154,166,184,0.28)"></span>uncached ' + intfmt(uncached) + '</span>');

  const W = 900, H = 52, m = { left: 2, right: 2 };
  const iw = W - m.left - m.right, barH = 34, gap = 2, r = 4;
  const g = makeSvg(sel, W, H).append('g').attr('transform', 'translate(' + m.left + ',6)');
  const segs = [
    { label: 'intra-conversation', v: intra, c: COL.green },
    { label: 'cross-conversation', v: cross, c: COL.violet },
    { label: 'uncached', v: uncached, c: 'rgba(154,166,184,0.28)' }
  ].filter(function(s){ return s.v > 0; });
  let acc = 0;
  const scale = function(v){ return iw * v / total; };
  segs.forEach(function(s, i){
    const w = Math.max(0, scale(s.v) - (i < segs.length - 1 ? gap : 0));
    const x0 = scale(acc);
    g.append('rect').attr('x', x0).attr('y', 0).attr('width', w).attr('height', barH)
      .attr('rx', r).attr('fill', s.c)
      .on('mousemove', function(ev){ showTip('<b>' + s.label + '</b><br>' + intfmt(s.v) +
        ' blk &middot; ' + pctfmt(s.v / total), ev); })
      .on('mouseleave', hideTip);
    if (w > 46)
      g.append('text').attr('x', x0 + w / 2).attr('y', barH / 2 + 4).attr('text-anchor', 'middle')
        .attr('fill', s.c === 'rgba(154,166,184,0.28)' ? COL.mute : COL.bg)
        .style('font-size', '11px').style('font-weight', 600).text(pctfmt(s.v / total));
    acc += s.v;
  });
}

// ---- 6. Turn-by-turn ISL/OSL + history growth ---------------------------
function renderTurns(){
  const sel = '#turns';
  const rows = (DATA.turns && DATA.turns.by_index) || [];
  if (!rows.length){ mutedNote(sel, 'No per-turn data available.'); return; }
  d3.select(sel).append('div').attr('class', 'legend').html(
    '<span><span class="swatch" style="background:' + COL.blue + '"></span>ISL p50</span>' +
    '<span><span class="swatch" style="background:' + COL.amber + '"></span>OSL p50</span>' +
    '<span><span class="swatch" style="background:' + COL.green + '"></span>mean history growth</span>' +
    '<span style="color:' + COL.mute + '">bar opacity &prop; conversations reaching</span>');

  const W = 900, H = 300, m = { top: 14, right: 16, bottom: 34, left: 48 };
  const iw = W - m.left - m.right, ih = H - m.top - m.bottom;
  const g = makeSvg(sel, W, H).append('g')
    .attr('transform', 'translate(' + m.left + ',' + m.top + ')');
  const idx = rows.map(function(r){ return r.turn_index; });
  const x0 = d3.scaleBand().domain(idx).range([0, iw]).paddingInner(0.28);
  const x1 = d3.scaleBand().domain(['isl', 'osl']).range([0, x0.bandwidth()]).padding(0.12);
  const p50 = function(r, k){ return r[k] ? r[k].p50 : 0; };
  const maxReach = d3.max(rows, function(r){ return r.conversations_reaching || 0; }) || 1;
  const growth = rows.map(function(r){ return r.mean_history_growth; })
    .filter(function(v){ return v != null && isFinite(v); });
  const maxV = d3.max(rows, function(r){ return Math.max(p50(r, 'isl'), p50(r, 'osl')); }) || 1;
  const maxG = growth.length ? d3.max(growth) : 0;
  const y = d3.scaleLinear().domain([0, Math.max(maxV, maxG)]).nice().range([ih, 0]);
  gridY(g, y, iw, 5);
  styleAxis(g.append('g').attr('transform', 'translate(0,' + ih + ')').call(d3.axisBottom(x0)));
  styleAxis(g.append('g').call(d3.axisLeft(y).ticks(5)));
  g.append('text').attr('x', iw).attr('y', ih + 30).attr('text-anchor', 'end')
    .attr('fill', COL.mute).style('font-size', '10px').text('turn index / tokens');

  // Faint full-height survivorship background per turn.
  g.selectAll('.surv').data(rows).enter().append('rect')
    .attr('x', function(r){ return x0(r.turn_index); }).attr('y', 0)
    .attr('width', x0.bandwidth()).attr('height', ih)
    .attr('fill', COL.ink).attr('opacity', 0.02);

  const series = [['isl', COL.blue], ['osl', COL.amber]];
  series.forEach(function(s){
    g.selectAll('.tb-' + s[0]).data(rows).enter().append('rect')
      .attr('x', function(r){ return x0(r.turn_index) + x1(s[0]); })
      .attr('width', x1.bandwidth())
      .attr('y', function(r){ return y(p50(r, s[0])); })
      .attr('height', function(r){ return ih - y(p50(r, s[0])); })
      .attr('rx', 3).attr('fill', s[1])
      .attr('opacity', function(r){ return 0.4 + 0.6 * (r.conversations_reaching || 0) / maxReach; })
      .on('mousemove', function(ev, r){
        showTip('<b>Turn ' + r.turn_index + '</b><br>' + s[0].toUpperCase() + ' p50: ' +
          num(p50(r, s[0]), 1) + '<br>reaching: ' + intfmt(r.conversations_reaching), ev); })
      .on('mouseleave', hideTip);
  });

  // History-growth line on the same token axis.
  if (growth.length){
    const gr = rows.filter(function(r){ return r.mean_history_growth != null && isFinite(r.mean_history_growth); });
    const line = d3.line()
      .x(function(r){ return x0(r.turn_index) + x0.bandwidth() / 2; })
      .y(function(r){ return y(Math.max(0, r.mean_history_growth)); });
    g.append('path').datum(gr).attr('fill', 'none').attr('stroke', COL.green)
      .attr('stroke-width', 2).attr('d', line);
    g.selectAll('.gr-pt').data(gr).enter().append('circle')
      .attr('cx', function(r){ return x0(r.turn_index) + x0.bandwidth() / 2; })
      .attr('cy', function(r){ return y(Math.max(0, r.mean_history_growth)); })
      .attr('r', 3).attr('fill', COL.green)
      .on('mousemove', function(ev, r){ showTip('<b>Turn ' + r.turn_index +
        '</b><br>history growth: ' + num(r.mean_history_growth, 1) + ' tok', ev); })
      .on('mouseleave', hideTip);
    const last = gr[gr.length - 1];
    g.append('text').attr('x', x0(last.turn_index) + x0.bandwidth() / 2 + 6)
      .attr('y', y(Math.max(0, last.mean_history_growth)) - 6)
      .attr('fill', COL.mute).style('font-size', '10px')
      .text('growth ' + num(last.mean_history_growth, 0));
  }
}

// ---- 7. Concurrency over time (step-area) -------------------------------
function renderConcurrency(){
  const sel = '#concurrency';
  if (!DATA.timeline){ mutedNote(sel, 'Execution timeline not available (no records).'); return; }
  const c = DATA.timeline.concurrency || {};
  const samples = c.samples || [];
  if (!samples.length){ mutedNote(sel, 'No concurrency samples available.'); return; }
  const pts = samples.map(function(s){ return { t: s[0], n: s[1] }; });
  const peak = c.peak || d3.max(pts, function(p){ return p.n; }) || 1;
  const twa = c.time_weighted_avg;

  const W = 900, H = 280, m = { top: 14, right: 16, bottom: 34, left: 44 };
  const iw = W - m.left - m.right, ih = H - m.top - m.bottom;
  const g = makeSvg(sel, W, H).append('g')
    .attr('transform', 'translate(' + m.left + ',' + m.top + ')');
  const x = d3.scaleLinear()
    .domain([0, d3.max(pts, function(p){ return p.t; }) || 1]).nice().range([0, iw]);
  const y = d3.scaleLinear().domain([0, peak]).nice().range([ih, 0]);
  gridY(g, y, iw, 5);
  styleAxis(g.append('g').attr('transform', 'translate(0,' + ih + ')').call(d3.axisBottom(x).ticks(6)));
  styleAxis(g.append('g').call(d3.axisLeft(y).ticks(5)));
  g.append('text').attr('x', iw).attr('y', ih + 30).attr('text-anchor', 'end')
    .attr('fill', COL.mute).style('font-size', '10px').text('seconds');

  const grad = g.append('defs').append('linearGradient').attr('id', 'ccgrad')
    .attr('x1', 0).attr('y1', 0).attr('x2', 0).attr('y2', 1);
  grad.append('stop').attr('offset', '0%').attr('stop-color', 'rgba(86,200,221,0.34)');
  grad.append('stop').attr('offset', '100%').attr('stop-color', 'rgba(86,200,221,0.06)');
  const area = d3.area().curve(d3.curveStepAfter)
    .x(function(p){ return x(p.t); }).y0(ih).y1(function(p){ return y(p.n); });
  const line = d3.line().curve(d3.curveStepAfter)
    .x(function(p){ return x(p.t); }).y(function(p){ return y(p.n); });
  g.append('path').datum(pts).attr('fill', 'url(#ccgrad)').attr('d', area);
  g.append('path').datum(pts).attr('fill', 'none').attr('stroke', COL.cyan)
    .attr('stroke-width', 1.8).attr('d', line);

  // Reference lines at peak and time-weighted average.
  g.append('line').attr('x1', 0).attr('x2', iw).attr('y1', y(peak)).attr('y2', y(peak))
    .attr('stroke', COL.amber).attr('stroke-dasharray', '5,4').attr('stroke-opacity', .8);
  g.append('text').attr('x', iw - 4).attr('y', y(peak) + 12).attr('text-anchor', 'end')
    .attr('fill', COL.mute).style('font-size', '10px').text('peak ' + intfmt(peak));
  if (twa != null && isFinite(twa)){
    g.append('line').attr('x1', 0).attr('x2', iw).attr('y1', y(twa)).attr('y2', y(twa))
      .attr('stroke', COL.green).attr('stroke-dasharray', '3,3').attr('stroke-opacity', .7);
    g.append('text').attr('x', iw - 4).attr('y', y(twa) - 5).attr('text-anchor', 'end')
      .attr('fill', COL.mute).style('font-size', '10px').text('avg ' + num(twa, 1));
  }
  crosshair(g, iw, ih, pts,
    function(p){ return x(p.t); }, function(p){ return y(p.n); },
    function(p){ return '<b>t = ' + num(p.t, 2) + ' s</b><br>inflight: ' + intfmt(p.n); },
    COL.cyan);
}

// ---- 8. Turn-depth funnel -----------------------------------------------
function renderFunnel(){
  const sel = '#funnel';
  const rows = (DATA.turns && DATA.turns.by_index) || [];
  if (!rows.length){ mutedNote(sel, 'No per-turn data available.'); return; }
  const data = rows.slice().sort(function(a, b){ return a.turn_index - b.turn_index; });
  const W = 540, H = Math.max(160, 34 + data.length * 30), m = { top: 8, right: 60, bottom: 24, left: 58 };
  const iw = W - m.left - m.right, ih = H - m.top - m.bottom;
  const g = makeSvg(sel, W, H).append('g')
    .attr('transform', 'translate(' + m.left + ',' + m.top + ')');
  const maxR = d3.max(data, function(r){ return r.conversations_reaching || 0; }) || 1;
  const x = d3.scaleLinear().domain([0, maxR]).range([0, iw]);
  const yb = d3.scaleBand().domain(data.map(function(r){ return r.turn_index; }))
    .range([0, ih]).padding(0.22);
  styleAxis(g.append('g').call(d3.axisLeft(yb)
    .tickFormat(function(d){ return 'turn ' + d; })));
  data.forEach(function(r){
    const w = x(r.conversations_reaching || 0);
    g.append('rect').attr('x', 0).attr('y', yb(r.turn_index)).attr('width', w)
      .attr('height', yb.bandwidth()).attr('rx', 4).attr('fill', COL.blue)
      .attr('opacity', 0.55 + 0.45 * (r.conversations_reaching || 0) / maxR)
      .on('mousemove', function(ev){ showTip('<b>Turn ' + r.turn_index + '</b><br>' +
        intfmt(r.conversations_reaching) + ' conversations', ev); })
      .on('mouseleave', hideTip);
    g.append('text').attr('x', w + 6).attr('y', yb(r.turn_index) + yb.bandwidth() / 2 + 4)
      .attr('fill', COL.ink).style('font-size', '11px').text(intfmt(r.conversations_reaching));
  });
  const tpc = DATA.shape && DATA.shape.turns_per_conversation;
  if (tpc){
    d3.select(sel).insert('div', 'svg').attr('class', 'panel-sub')
      .text('Turns per conversation: p50 ' + num(tpc.p50, 1) + ' &middot; p90 ' + num(tpc.p90, 1));
  }
}

// ---- 9. Think-time distribution -----------------------------------------
function renderThinkTime(){
  const sel = '#thinktime';
  const rows = (DATA.turns && DATA.turns.by_index) || [];
  const withThink = rows.filter(function(r){ return r.authored_think_time_ms; });
  if (!withThink.length){ mutedNote(sel, 'No authored think-time data in this dataset.'); return; }
  d3.select(sel).append('div').attr('class', 'legend').html(
    '<span><span class="swatch" style="background:' + COL.violet + '"></span>p25 &ndash; p50 &ndash; p90 (ms)</span>');
  const W = 540, H = Math.max(150, 30 + withThink.length * 30), m = { top: 8, right: 20, bottom: 30, left: 58 };
  const iw = W - m.left - m.right, ih = H - m.top - m.bottom;
  const g = makeSvg(sel, W, H).append('g')
    .attr('transform', 'translate(' + m.left + ',' + m.top + ')');
  const maxV = d3.max(withThink, function(r){ return r.authored_think_time_ms.p90; }) || 1;
  const x = d3.scaleLinear().domain([0, maxV]).nice().range([0, iw]);
  const yb = d3.scaleBand().domain(withThink.map(function(r){ return r.turn_index; }))
    .range([0, ih]).padding(0.3);
  styleAxis(g.append('g').attr('transform', 'translate(0,' + ih + ')').call(d3.axisBottom(x).ticks(5)));
  styleAxis(g.append('g').call(d3.axisLeft(yb).tickFormat(function(d){ return 'turn ' + d; })));
  g.append('text').attr('x', iw).attr('y', ih + 26).attr('text-anchor', 'end')
    .attr('fill', COL.mute).style('font-size', '10px').text('think time (ms)');
  withThink.forEach(function(r){
    const s = r.authored_think_time_ms;
    const cy = yb(r.turn_index) + yb.bandwidth() / 2;
    g.append('line').attr('x1', x(s.p25)).attr('x2', x(s.p90)).attr('y1', cy).attr('y2', cy)
      .attr('stroke', COL.violet).attr('stroke-width', 2).attr('stroke-opacity', .5);
    g.append('circle').attr('cx', x(s.p50)).attr('cy', cy).attr('r', 5).attr('fill', COL.violet)
      .on('mousemove', function(ev){ showTip('<b>Turn ' + r.turn_index + '</b><br>p25 ' +
        num(s.p25, 0) + ' &middot; p50 ' + num(s.p50, 0) + ' &middot; p90 ' + num(s.p90, 0) + ' ms', ev); })
      .on('mouseleave', hideTip);
  });
}

// ---- 10. Shared-prefix depth bars ---------------------------------------
function renderPrefixDepth(){
  const sel = '#prefixdepth';
  const cache = DATA.cache;
  if (!cache || !cache.ideal){ mutedNote(sel, 'Prefix-cache reuse not available for this dataset.'); return; }
  const buckets = cache.ideal.shared_prefix_rate_ge || {};
  const keys = Object.keys(buckets).sort(function(a, b){ return (+a) - (+b); });
  if (!keys.length){ mutedNote(sel, 'No shared-prefix buckets available.'); return; }
  if (cache.identity_source)
    d3.select(sel).append('div').attr('class', 'panel-sub')
      .text('Requests reusing ≥ threshold share — identity source: ' + cache.identity_source);
  const rows = keys.map(function(k){ return { k: k, v: buckets[k] }; });
  const W = 540, H = 240, m = { top: 10, right: 16, bottom: 32, left: 44 };
  const iw = W - m.left - m.right, ih = H - m.top - m.bottom;
  const g = makeSvg(sel, W, H).append('g')
    .attr('transform', 'translate(' + m.left + ',' + m.top + ')');
  const x = d3.scaleBand().domain(keys).range([0, iw]).padding(0.28);
  const maxV = d3.max(rows, function(r){ return r.v; }) || 1;
  const y = d3.scaleLinear().domain([0, maxV]).nice().range([ih, 0]);
  gridY(g, y, iw, 4);
  styleAxis(g.append('g').attr('transform', 'translate(0,' + ih + ')')
    .call(d3.axisBottom(x).tickFormat(function(d){ return '≥' + d + '%'; })));
  styleAxis(g.append('g').call(d3.axisLeft(y).ticks(4, '~s')));
  g.selectAll('.pd').data(rows).enter().append('rect')
    .attr('x', function(r){ return x(r.k); }).attr('width', x.bandwidth())
    .attr('y', function(r){ return y(r.v); }).attr('height', function(r){ return ih - y(r.v); })
    .attr('rx', 4).attr('fill', COL.green)
    .on('mousemove', function(ev, r){ showTip('<b>≥' + r.k + '% shared</b><br>' +
      intfmt(r.v) + ' requests', ev); })
    .on('mouseleave', hideTip);
  g.selectAll('.pd-lbl').data(rows).enter().append('text')
    .attr('x', function(r){ return x(r.k) + x.bandwidth() / 2; })
    .attr('y', function(r){ return y(r.v) - 5; }).attr('text-anchor', 'middle')
    .attr('fill', COL.mute).style('font-size', '10px')
    .text(function(r){ return intfmt(r.v); });
}

// ---- 11. Queue-delay distribution ---------------------------------------
function renderQueueDelay(){
  const sel = '#queuedelay';
  const q = DATA.timeline && DATA.timeline.queue && DATA.timeline.queue.queue_delay_ms;
  if (!q){ mutedNote(sel, 'No queue-delay data (no admission times).'); return; }
  const rows = [
    { k: 'p50', v: q.p50 }, { k: 'p90', v: q.p90 }, { k: 'p99', v: q.p99 }, { k: 'max', v: q.max }
  ];
  const W = 540, H = 200, m = { top: 10, right: 40, bottom: 30, left: 48 };
  const iw = W - m.left - m.right, ih = H - m.top - m.bottom;
  const g = makeSvg(sel, W, H).append('g')
    .attr('transform', 'translate(' + m.left + ',' + m.top + ')');
  const maxV = d3.max(rows, function(r){ return r.v; }) || 1;
  const x = d3.scaleLinear().domain([0, maxV]).nice().range([0, iw]);
  const yb = d3.scaleBand().domain(rows.map(function(r){ return r.k; })).range([0, ih]).padding(0.3);
  styleAxis(g.append('g').attr('transform', 'translate(0,' + ih + ')').call(d3.axisBottom(x).ticks(5)));
  styleAxis(g.append('g').call(d3.axisLeft(yb)));
  g.append('text').attr('x', iw).attr('y', ih + 26).attr('text-anchor', 'end')
    .attr('fill', COL.mute).style('font-size', '10px').text('queue delay (ms)');
  g.selectAll('.qd').data(rows).enter().append('rect')
    .attr('x', 0).attr('y', function(r){ return yb(r.k); }).attr('width', function(r){ return x(r.v); })
    .attr('height', yb.bandwidth()).attr('rx', 4).attr('fill', COL.cyan).attr('opacity', .85)
    .on('mousemove', function(ev, r){ showTip('<b>' + r.k + '</b><br>' + num(r.v, 2) + ' ms', ev); })
    .on('mouseleave', hideTip);
  g.selectAll('.qd-lbl').data(rows).enter().append('text')
    .attr('x', function(r){ return x(r.v) + 6; })
    .attr('y', function(r){ return yb(r.k) + yb.bandwidth() / 2 + 4; })
    .attr('fill', COL.ink).style('font-size', '10px').text(function(r){ return num(r.v, 1); });
}

// ---- Raw statistics tables ----------------------------------------------
function renderRawTables(){
  const root = d3.select('#rawtables');
  const cols = ['count', 'mean', 'std', 'min', 'p50', 'p90', 'p95', 'p99', 'max'];
  const l = DATA.lengths || {};
  const named = [['isl', 'ISL'], ['osl', 'OSL'], ['total', 'Total'], ['isl_osl_ratio', 'ISL/OSL ratio']];
  let html = '<table><thead><tr><th>metric</th>';
  cols.forEach(function(c){ html += '<th>' + c + '</th>'; });
  html += '</tr></thead><tbody>';
  named.forEach(function(n){
    const s = l[n[0]];
    html += '<tr><td>' + n[1] + '</td>';
    cols.forEach(function(c){
      const isRatio = n[0] === 'isl_osl_ratio';
      html += '<td>' + (s ? (c === 'count' ? intfmt(s[c]) : num(s[c], isRatio ? 2 : 1)) : '—') + '</td>';
    });
    html += '</tr>';
  });
  html += '</tbody></table>';

  // Per-turn table.
  const rows = (DATA.turns && DATA.turns.by_index) || [];
  if (rows.length){
    html += '<table style="margin-top:14px;"><thead><tr><th>turn</th><th>reaching</th>' +
      '<th>ISL p50</th><th>OSL p50</th><th>hist growth</th><th>think ms p50</th></tr></thead><tbody>';
    rows.forEach(function(r){
      html += '<tr><td>' + r.turn_index + '</td><td>' + intfmt(r.conversations_reaching) + '</td>' +
        '<td>' + (r.isl ? num(r.isl.p50, 1) : '—') + '</td>' +
        '<td>' + (r.osl ? num(r.osl.p50, 1) : '—') + '</td>' +
        '<td>' + (r.mean_history_growth == null ? '—' : num(r.mean_history_growth, 1)) + '</td>' +
        '<td>' + (r.authored_think_time_ms ? num(r.authored_think_time_ms.p50, 1) : '—') + '</td></tr>';
    });
    html += '</tbody></table>';
  }

  // Per-conversation table, present only when the breakdown was requested
  // (`--dataset-analysis-per-conversation`).
  const conv = DATA.conversations || [];
  if (conv.length){
    html += '<table style="margin-top:14px;"><thead><tr><th>conversation</th><th>turns</th>' +
      '<th>ISL mean</th><th>OSL mean</th><th>total tokens</th></tr></thead><tbody>';
    conv.forEach(function(c){
      const cl = c.lengths || {};
      html += '<tr><td>' + esc(c.conversation_id) + '</td><td>' + intfmt(c.turns) + '</td>' +
        '<td>' + (cl.isl ? num(cl.isl.mean, 1) : '—') + '</td>' +
        '<td>' + (cl.osl ? num(cl.osl.mean, 1) : '—') + '</td>' +
        '<td>' + intfmt(cl.grand_total_tokens) + '</td></tr>';
    });
    html += '</tbody></table>';
  }
  root.html(html);
}

// ---- Bootstrap -----------------------------------------------------------
const RENDERERS = [
  ['subtitle', renderSubtitle], ['stats', renderStats], ['cdfs', renderCDFs],
  ['split', renderSplit], ['hitrate', renderHitRate], ['evictions', renderEvictions],
  ['savings', renderSavings], ['turns', renderTurns], ['concurrency', renderConcurrency],
  ['funnel', renderFunnel], ['thinktime', renderThinkTime], ['prefixdepth', renderPrefixDepth],
  ['queuedelay', renderQueueDelay], ['rawtables', renderRawTables]
];
RENDERERS.forEach(function(r){
  try { r[1](); } catch (e){ console.error(r[0], e); }
});
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
        // agentx swim_lane_viewer HUD theme tokens are present.
        assert!(html.contains("--amber:#f5b942"));
        assert!(html.contains("https://d3js.org/d3.v7.min.js"));
        assert!(html.contains("const DATA ="));
        // embedded JSON escapes </ so no stray </script>
        assert!(!html.contains("</script></script>"));
        assert!(html.contains("AIPerf Dry-Run Dataset Analysis"));
        // data actually embedded
        assert!(html.contains("\"conversations\""));
        // new chart-suite section markers
        assert!(html.contains("Prefill vs Decode Token Split"));
        assert!(html.contains("Hit Rate vs Capacity"));
        assert!(html.contains("Concurrency Over Time"));
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
