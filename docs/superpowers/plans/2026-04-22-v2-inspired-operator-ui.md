# v2-inspired operator UI upgrades

**Goal:** Port the best patterns from `src/aiperf/api/static-v2/` into the operator UI so Job Detail and Dashboard become diagnostic views, not "bare numbers on dark theme".

**Reference:** `src/aiperf/api/static-v2/` — the new live single-run dashboard. Keep it untouched; we're copying ideas + code, not deleting.

**Branch:** `ajc/k8s`. `git commit --no-verify -s`. No `git stash`.

---

## Task A — SLO-aware Job Detail bundle (HeroStrip + SLO chips + p99 headline + PhaseCards)

**Files to create:**
- `src/aiperf/operator/ui/components/hero-strip.js` — port of `static-v2/components/hero-strip.js`, adapted to read from the operator's `{job, status, config}` props instead of signals.
- `src/aiperf/operator/ui/components/phase-cards.js` — port of `static-v2/components/phase-cards.js`, reading `status.phases` (dict form the operator already writes).

**Files to modify:**
- `src/aiperf/operator/ui/components/kpi-card.js` — accept optional `slo={threshold, compare}` prop; render a green/red chip with "≤ 500" text next to the label when set.
- `src/aiperf/operator/ui/pages/job-detail.js`:
  - Replace the current 5-tile KPI grid with a new spec table mirroring v2's `TILES` array: **throughput → `current` headline + `avg` sub**, **latency → `p99` headline + `avg` sub**, **TTFT → `p99` headline + `avg` sub**, **ITL → `avg` headline + `p99` sub**, **Output Tokens/s → `current` + `avg`**. Preserve existing `kpi-throughput`/`kpi-ttft-avg`/`kpi-latency-p99`/etc. test-ids so the e2e suite still locates them.
  - Insert `<HeroStrip>` after the breadcrumb/header card, before the KPI grid. Pass `{info, status, config}`.
  - Replace the inline `<PhaseBar>` usage with `<PhaseCards phases=${status.phases} />`.
  - SLO source of truth: `config?.spec?.benchmark?.slos ?? config?.spec?.slos ?? {}` (check both shapes — the config endpoint may unwrap the benchmark wrapper in the summary-fallback path). Fall back to empty object when not configured; no SLO chips render.

**Tests:**
- Unit for `_synthesize_status_from_summary` (already exists) — no changes needed.
- New e2e in `test_job_detail.py`:
  - `test_job_detail_shows_hero_strip` — HeroStrip visible with a status label text (one of "On target", "Waiting for data", "SLO violated", "SLO slipping", "Errors reported", "Attention needed").
  - `test_job_detail_kpi_ttft_headlines_p99` — `kpi-ttft-avg` tile's big value matches the p99 number (or rename test-id to `kpi-ttft` if the label changes).
  - `test_job_detail_slo_chip_when_declared` — seed `status.summary.ttft_p99_ms` + a config with `slos: {time_to_first_token: 500}` → chip visible with "✓" or "✗".

**Commit:** `feat(operator-ui): SLO-aware job detail (hero strip, SLO chips, p99 headline, phase cards)`

---

## Task B — Sparklines on every KPI tile

**Files to create:**
- `src/aiperf/operator/ui/components/sparkline.js` — port of `static-v2/components/sparkline.js`, verbatim (pure SVG, zero deps).

**Files to modify:**
- `src/aiperf/operator/ui/components/kpi-card.js` — accept optional `points: Array<{t, v}>` prop; when non-empty, render the sparkline between the big value and the subtitle. Stroke color follows SLO kind if SLO prop is set (green/red), else neutral.
- `src/aiperf/operator/ui/pages/job-detail.js` — build a time-series per KPI from `status.summary` history. For completed jobs, we don't have a persisted time series — derive a sparkline from the per-phase completion history in `status.phases` if present, else render empty/omit. For **running** jobs, accumulate samples in a local `useRef` / `useState` on each poll tick (the page already polls `api.getJob` periodically). A rolling buffer of 60 samples is enough.

**Tests:**
- E2e: `test_job_detail_sparkline_visible` — `svg.sparkline` present inside each KPI tile. Skip or relax if the UI renders an empty placeholder on archived (which is fine — the placeholder is a stable `<svg>` element).

**Commit:** `feat(operator-ui): inline sparklines on KPI tiles`

---

## Task C — GPU Telemetry + Reliability Tile

**Files to create:**
- `src/aiperf/operator/ui/components/gpu-telemetry.js` — port of `static-v2/components/gpu-telemetry.js`. Data source: `server_metrics_export.json` (already optionally fetched by job-detail.js into the `serverMetrics` state). Parse the metrics list the same way v2 does — `(endpoint, gpuIndex, model)` from the metric header.
- `src/aiperf/operator/ui/components/reliability-tile.js` — extract `ReliabilityTile` from v2's `realtime-metrics.js`. Headlines `N failed` when SLOs declared and some requests missed them; falls back to Success Rate when no SLOs.

**Files to modify:**
- `src/aiperf/operator/ui/pages/job-detail.js`:
  - Insert `<GpuTelemetryCard>` between the existing Server Metrics card and Job Configuration, OR after the live-throughput chart if the server-metrics card doesn't exist on this page (check the current render). Hide when no GPU data.
  - Insert `<ReliabilityTile>` as an additional KPI tile in the grid (use `source !== 'archived'` guard if needed — for archived jobs, reliability reads from `status.summary.error_rate` which we already synthesize).

**Tests:**
- E2e `test_job_detail_gpu_card_hidden_without_data` — with the default seeded data (no server_metrics_export), the card is absent.
- E2e `test_job_detail_reliability_tile_shows_success_rate` — reliability tile visible and contains "100%" or "0 errors" for the golden c128 job (error_rate=0.0).

**Commit:** `feat(operator-ui): GPU telemetry card + reliability tile on job detail`

---

## Task D — Dashboard hero for the currently-running job

**Files to modify:**
- `src/aiperf/operator/ui/pages/dashboard.js`:
  - Find a single currently-running job: `const live = jobs.find(j => j.phase === 'Running' && j.source !== 'archived')`.
  - When present, render `<HeroStrip>` (imported from Task A) above the existing scatter chart. Fetch its `/api/v1/jobs/{ns}/{name}` detail to get `status.summary/phases/conditions` + fetch `/api/v1/config/{ns}/{name}` for SLOs.
  - When no live run, no hero — the dashboard keeps its current look.
  - Clicking the hero navigates to the job detail.

**Tests:**
- E2e `test_dashboard_hero_shows_for_running_job` — with the seeded `live-run` CR, a hero element (`data-testid="dashboard-hero"`) is visible; its text contains `live-run`.
- E2e `test_dashboard_hero_absent_when_no_running_job` — after `build_empty(results_dir)` + no-running-CR fake → hero absent.

**Commit:** `feat(operator-ui): dashboard hero for currently-running job`

---

## Cross-cutting constraints

- Every task runs e2e at the end: `uv run pytest tests/e2e/operator_ui/ -m e2e -n auto`. Baseline 34 passing / 1 skipped must be preserved (plus the new tests each task adds).
- No CSS class invention — reuse `text-dim`, existing card/chip/badge classes, or inline styles with CSS tokens.
- No modifications to `src/aiperf/api/static-v2/`.
- Source test-ids (`data-testid`) must be preserved where the e2e suite references them.
