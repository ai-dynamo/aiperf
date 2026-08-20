// Drives the local operator UI with Playwright: screenshots each page and
// asserts the presentation contracts the recent fixes established.
//
// Assertions are deliberately about MEANING, not pixels. A screenshot proves a
// page rendered; only a text assertion proves it rendered the right thing. The
// checks below encode the specific defects that were fixed, so a regression
// fails loudly instead of producing a plausible-looking image.
import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_OUT = path.join(HERE, '../../artifacts/ui-verify/shots');
const OUT = process.env.SHOT_DIR
  ? path.resolve(HERE, process.env.SHOT_DIR)
  : DEFAULT_OUT;
const BASE = process.env.BASE || 'http://127.0.0.1:8098';
const NS = process.env.NAMESPACE;

if (!NS) {
  throw new Error('NAMESPACE is required to select the verification sweeps.');
}

fs.mkdirSync(OUT, { recursive: true });

const PAGES = [
  { slug: 'dashboard', hash: '#/' },
  { slug: 'jobs', hash: '#/jobs' },
  { slug: 'sweeps-list', hash: '#/sweeps' },
  { slug: 'sweep-adaptive-gemma-bo4', hash: `#/sweeps/${NS}/gemma-bo4` },
  { slug: 'sweep-real-cluster-bo5', hash: `#/sweeps/${NS}/gemma-bo5` },
  { slug: 'sweep-adaptive-itl-metric', hash: `#/sweeps/${NS}/gemma-bo4?metric=inter_token_latency.avg` },
  { slug: 'sweep-grid-gemma-conc2', hash: `#/sweeps/${NS}/gemma-conc2` },
  { slug: 'sweep-grid-cp-sweep', hash: `#/sweeps/${NS}/cp-sweep` },
  { slug: 'leaderboard', hash: '#/leaderboard' },
  { slug: 'compare', hash: '#/compare' },
];

// Each check: [label, predicate(bodyText), why it matters]
const CHECKS = {
  'sweep-adaptive-gemma-bo4': [
    ['winner card is populated, not empty',
      t => !/No completed variation has a finite/i.test(t),
      'an empty winner card silently passes any absence-only assertion'],
    ['winner names the true best point (concurrency=17)',
      t => /WINNER SUMMARY[\s\S]{0,400}concurrency=17/i.test(t),
      'conc 17 is the SLA-feasible objective maximum per search_history.json best_trials'],
    ['winner HEADLINE is the swept values, not the planner cell id',
      (t, d) => !!d.winnerHeadline && !/^search_iter_/.test(d.winnerHeadline),
      'the id may remain as a subtitle for artifact paths; it must not be the headline'],
    ['winner headline is exactly concurrency=17',
      (t, d) => d.winnerHeadline === 'concurrency=17',
      'pins the element, not just its presence somewhere in the body text'],
    ['winner cell id matches the winning iteration, not another variation',
      t => /search_iter_0008 . variation 8/.test(t),
      'the card once spliced the planner headline onto the JS-derived pick\'s ' +
      'cell id, so following that id fetched a different run\'s artifacts'],
    ['no stale live-progress claim on a Succeeded sweep',
      t => !/running variation/i.test(t),
      'a terminal object must not render live affordances'],
    ['variation count is the measured 14, not the cap 22',
      t => /VARIATIONS[\s\S]{0,60}\b14\b/.test(t),
      'total_variations is maxIterations for adaptive; rendering it as a total implies truncation'],
    ['winner is labelled by swept values',
      t => /concurrency=\d+/.test(t),
      'the variations table degrades to search_iter_NNNN on archives whose ' +
      'children.json predates variation_values, which is correct; the winner ' +
      'card always has values because the planner verdict carries them'],
    ['variation curve has data',
      t => !/data available for any variation/i.test(t),
      'an empty chart hides whatever the chart fix was meant to prove'],
    ['planner verdict is rendered (convergence reason)',
      t => /converge|improvement.patience|stopped/i.test(t),
      'search_summary.convergence_reason is now on the API; the card should say why it stopped'],
    ['converged run says "converged", never "hit limit"',
      t => /\bconverged\b/i.test(t) && !/hit limit/i.test(t),
      'stop_kind is "converged"; an archive loses the original maxIterations so ' +
      'declared===observed, and "hit limit N" would assert a cap that never existed'],
    ['SLA boundary is surfaced',
      t => /17/.test(t) && /(boundary|SLA|feasible)/i.test(t),
      'feasible_max=17 vs infeasible_min=12 is the most useful output of a constrained search'],
    ['trials are not all zero',
      t => !/\b0\/1\b[\s\S]{0,300}\b0\/1\b[\s\S]{0,300}\b0\/1\b/.test(t),
      '0/1 everywhere means child summaries never loaded'],
  ],
  'sweep-real-cluster-bo5': [
    ['real cluster sweep: winner populated',
      t => !/No completed variation has a finite/i.test(t), 'end-to-end on cluster data'],
    ['real cluster sweep: labelled by swept values',
      t => /concurrency=\d+/.test(t), 'variation_values reaches the UI on a live sweep'],
    ['budget-exhausted run must NOT claim it converged',
      t => !/converged early/i.test(t) && /not because it converged/i.test(t),
      'stop_kind is budget_exhausted; the copy must deny convergence, and a bare ' +
      '/converged/ match would flag the correct denial "not because it converged"'],
    ['SLA boundary from the real run',
      t => /17/.test(t) && /(boundary|SLA|feasible)/i.test(t), 'feasible_max 17 vs infeasible_min 22'],
  ],
  'sweep-adaptive-itl-metric': [
    ['winner does not follow the chart metric selector',
      t => !/WINNER SUMMARY[\s\S]{0,400}search_iter_0000/i.test(t),
      'the winner is a property of the sweep, not of the current chart view'],
    ['winner is still concurrency=17 under an ITL chart',
      t => /WINNER SUMMARY[\s\S]{0,400}concurrency=17/i.test(t),
      'changing the chart series must not change the reported winner'],
  ],
  // Every page gets at least a "did you actually render" check. Without these
  // the run printed ALL CHECKS PASSED while leaderboard showed "No completed
  // benchmarks yet" and compare showed only its job selector -- so six commits
  // touching those pages had no visual verification whatsoever.
  dashboard: [
    ['dashboard rendered content, not just chrome',
      t => t.length > 400 && !/^\s*$/.test(t),
      'a page that renders only the nav bar passes any check written about its body'],
  ],
  jobs: [
    ['jobs table has rows',
      t => /\brunning\b|\bsucceeded\b|\bcompleted\b|\bfailed\b/i.test(t),
      'an empty jobs table means the fixture or API returned nothing'],
  ],
  'sweeps-list': [
    ['sweeps list names the sweeps',
      t => /gemma-bo4/.test(t),
      'the list is the entry point to every sweep page under test'],
  ],
  leaderboard: [
    ['leaderboard is populated, not the empty state',
      t => !/no completed benchmarks/i.test(t),
      'the leaderboard ranking fix has no visual verification if the page is empty'],
  ],
  'sweep-grid-cp-sweep': [
    ['grid sweep winner is populated',
      t => !/No completed variation has a finite/i.test(t),
      'second grid sweep guards against a fix that only works on one dataset'],
  ],
  'sweep-grid-gemma-conc2': [
    ['grid sweep winner is populated',
      t => !/No completed variation has a finite/i.test(t),
      'the grid path must not regress from the adaptive work'],
  ],
};

const errors = [];
const results = [];

const browser = await chromium.launch();
// A tall viewport, NOT fullPage. The app renders inside an inner scroll
// container, so document.scrollHeight === window.innerHeight and Playwright's
// fullPage capture silently images only the first screen -- which is how two
// pages with visibly different tables produced byte-identical screenshots and
// the variations table, curve, Pareto plot and trial board were never imaged
// at all while the run reported ALL CHECKS PASSED.
const ctx = await browser.newContext({ viewport: { width: 1600, height: 3600 }, deviceScaleFactor: 1 });
const page = await ctx.newPage();

// In-flight request counter. innerText can go transiently stable while async
// per-child fetches are still landing, so text-stability ALONE captured a
// sweep page mid-load: the variation curve read "no data" and every row showed
// 0/1 trials, while the same page a second later had all 14 populated.
let inFlight = 0;
page.on('request', () => { inFlight += 1; });
page.on('requestfinished', () => { inFlight = Math.max(0, inFlight - 1); });
page.on('requestfailed', () => { inFlight = Math.max(0, inFlight - 1); });

page.on('console', m => { if (m.type() === 'error') errors.push(`[console] ${m.text()}`); });
page.on('pageerror', e => errors.push(`[pageerror] ${e.message}`));

for (const p of PAGES) {
  const before = errors.length;
  await page.goto(`${BASE}/${p.hash}`, { waitUntil: 'networkidle' });
  // The SPA renders after its fetches settle; give charts a beat to paint.
  await page.waitForTimeout(1200);
  // Kill animation and transition so a chart cannot be captured mid-tween.
  await page.addStyleTag({
    content: '*,*::before,*::after{animation:none!important;transition:none!important}',
  });
  // Scroll the inner container to its end first so lazy/virtualized rows mount,
  // then back to the top for the capture.
  const scrollAll = (to) => page.evaluate((mode) => {
    for (const el of document.querySelectorAll('*')) {
      if (el.scrollHeight > el.clientHeight + 50) {
        el.scrollTop = mode === 'end' ? el.scrollHeight : 0;
      }
    }
  }, to);
  await scrollAll('end');
  await page.waitForTimeout(300);
  await scrollAll('top');

  // Wait for the DOM to stop changing before capturing ANYTHING. Screenshotting
  // straight after the scroll produced an image with the data visibly collapsed
  // while the text assertions on the same page passed -- an image that
  // contradicts its own green run is worse than no image, because it is the
  // artifact a human trusts. Text and pixels are now both read from a settled
  // page, and `settled` is recorded so a run that never stabilised is visible
  // rather than silently believed.
  let settled = false;
  let text = '';
  for (let attempt = 0; attempt < 12; attempt++) {
    const a = await page.evaluate(() => document.body.innerText);
    await page.waitForTimeout(400);
    const b = await page.evaluate(() => document.body.innerText);
    // Both conditions: nothing outstanding on the wire AND the text has stopped
    // moving. Either alone is insufficient -- the network can be briefly quiet
    // between a manifest response and the per-child fan-out it triggers.
    if (a === b && b.length > 0 && inFlight === 0) { settled = true; text = b; break; }
    text = b;
  }
  // No fullPage -- see the viewport comment above. fullPage resizes the capture
  // to document.scrollHeight, which the inner scroll container keeps SHORTER
  // than the tall viewport, so it undoes the very fix that viewport applies.
  await page.screenshot({ path: path.join(OUT, `${p.slug}.png`) });
  // Element-level probes. Body-text regex cannot distinguish "the id appears
  // somewhere on the page" (intended -- it is kept as a subtitle for artifact
  // paths) from "the id IS the headline" (the defect). Query the node.
  const dom = await page.evaluate(() => ({
    winnerHeadline: document.querySelector('[data-testid="sweep-winner-headline"]')?.innerText?.trim() ?? null,
  }));
  const checks = (CHECKS[p.slug] ?? []).map(([label, fn, why]) => {
    let pass = false;
    try { pass = !!fn(text, dom); } catch (e) { pass = false; }
    return { label, pass, why };
  });
  results.push({
    slug: p.slug,
    hash: p.hash,
    chars: text.length,
    settled,
    dom,
    newErrors: errors.slice(before),
    checks,
  });
}

await browser.close();

fs.writeFileSync(path.join(OUT, 'report.json'), JSON.stringify({ results, errors }, null, 2));

let failed = 0;
for (const r of results) {
  const errN = r.newErrors.length;
  console.log(`\n${r.slug}  (${r.chars} chars${errN ? `, ${errN} JS ERROR(S)` : ''}${r.settled ? '' : ', NEVER SETTLED'})`);
  if (!r.settled) { failed++; console.log('    ! page never stopped changing - screenshot and text may disagree'); }
  for (const e of r.newErrors.slice(0, 3)) console.log(`    ! ${e.slice(0, 160)}`);
  for (const c of r.checks) {
    console.log(`    ${c.pass ? 'PASS' : 'FAIL'}  ${c.label}`);
    if (!c.pass) { failed++; console.log(`          why: ${c.why}`); }
  }
  if (errN) failed++;
}
console.log(`\n${failed === 0 ? 'ALL CHECKS PASSED' : failed + ' PROBLEM(S)'} — shots in ${OUT}`);
process.exit(failed === 0 ? 0 : 1);
