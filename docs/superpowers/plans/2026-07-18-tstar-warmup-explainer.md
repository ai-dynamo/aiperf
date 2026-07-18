# t* Warmup Explainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a hub-visible Flow-backed explainer deck (`tstar-warmup`) that teaches trajectory `t*` warmup end-to-end: sampling, trie chop, cache-pressure recycle, handoff resume, plus a brief contrast with ordinary phase warmup.

**Architecture:** Author one `.flow` under `apps/explainers/decks-flow/`, compile it through the existing `@aiperf/flow-compiler` package pipeline into `decks-generated/tstar-warmup.package.json`, and register the id/route in the packages-only registry so the hub card appears. No MentalModel, no new tests.

**Tech Stack:** AIPerf Flow (`.flow` + `@scene`), `make build-explainer-packages`, `apps/explainers` packages-only registry, SceneRenderer timelines.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-18-tstar-warmup-explainer-design.md`
- **No tests of any kind** — do not create or modify `*.test.ts`, `*.test.tsx`, Rust tests, or Vitest files
- **No git commits** unless the controller explicitly asks
- Packages-only path — no `MentalModel.tsx`, no dual-load fallback
- Identity must match spec exactly: `id: "tstar-warmup"`, `route: "/tstar-warmup"`, `topic: "graph-runtime"`
- Real config knobs only: `trajectory_start_min_ratio`, `trajectory_start_max_ratio`, `t_star_random_seed` (not invented names)
- Evidence paths: `rust/runtime/src/graph/{tstar,snapshot,warmup_handoff}.rs` and graph phase runtime
- Every diagram slide: non-empty `narration`, `points`, and `scene.timeline`
- Visual motif: 4-turn toy DAG `n0…n3`; pre-`t*` muted; `t*` cut warning; post-`t*` primary/green; handoff tertiary
- Match scene style of `apps/explainers/decks-flow/dynosim.flow` and `segment-pools.flow` (header, panels, connectors, sparse `motion.signal`)
- Model for subagents: `cursor-grok-4.5-high`

---

## File map

| File | Responsibility |
|---|---|
| `apps/explainers/decks-flow/tstar-warmup.flow` | Full deck source (~20 slides, Act 1 + Act 2) |
| `apps/explainers/src/decks-generated/tstar-warmup.package.json` | Generated DeckPackage (build output) |
| `apps/explainers/src/core/deck-registry.ts` | Add `["tstar-warmup", "/tstar-warmup"]` to `EXPECTED_DECK_ROUTES` |
| `apps/explainers/scripts/assert-deck-packages.mjs` | Add `"tstar-warmup"` to `EXPECTED_DECK_IDS` |

---

### Task 1: Act 1 Flow — mental model (slides 1–5)

**Files:**
- Create: `apps/explainers/decks-flow/tstar-warmup.flow`

**Interfaces:**
- Consumes: scene patterns from `dynosim.flow` / `segment-pools.flow`; accuracy from `tstar.rs` / `snapshot.rs` module docs
- Produces: valid `.flow` with explainer header + slides 1–5 only (Act 2 comes in Task 2–3)

- [ ] **Step 1: Create the explainer header**

Create `apps/explainers/decks-flow/tstar-warmup.flow` with SPDX header and:

```flow
explainer "t* Warmup" {
  id: "tstar-warmup"
  route: "/tstar-warmup"
  topic: "graph-runtime"
  storagePrefix: "tstar-warmup-explainer"
  classPrefix: "deck-tstar-warmup"
  eyebrowLabel: "T* WARMUP"
  startGateTitle: "t* warmup & cache-pressure handoff"

  hub: {
    highlight: "t* warmup"
    title: "snapshot · pressure · handoff"
    description: "How graph runs warm KV via a deterministic t* chop, cache-pressure recycle, and resume handoff — vs ordinary request-count warmup."
  }
```

- [ ] **Step 2: Author slides 1–5 with full `@scene` + timelines**

Required slide coverage (titles may vary slightly; topics must match):

1. **Why graph warmup** — cold KV / cold prefix vs measuring mid-trace resume
2. **Ordinary warmup ≠ `t*`** — `--warmup-request-count` / duration phases discard metrics; `t*` partitions a recorded trajectory
3. **The `t*` picture** — horizontal `n0…n3` timeline + vertical `t*` cut
4. **What stays warm** — server KV for executed prefix; client still materializes full prompt path
5. **Two phases, one plan** — warmup builds state; profiling resumes same seeded `t*` (not re-drawn)

Each slide must include: `eyebrow`, `title`, `lede`, `narration`, `term` (where useful), `points` (≥3, with at least one evidence path where claims are code-backed), `caption`, `render: @scene { roots: [...]; timeline: [...] }`.

Scene conventions for Act 1:
- Slide 3 motif: four panels `n0`–`n3` on a row; a vertical `core.path` or thin panel as the `t*` rule in `@theme.accent.warning`
- Slide 2: two columns — “Ordinary phase warmup” vs “Trajectory `t*`”
- Use `core.header`, `core.panel`, `core.connector` / `core.elbow`, optional `motion.signal`
- Timeline: staggered `enter` / `draw` / one `emphasis`; non-empty

- [ ] **Step 3: Sanity-check the file parses conceptually**

Confirm the file closes the `explainer` block after slide 5 (temporary closing brace OK — Task 2 will reopen/extend). Prefer leaving the file with slides 1–5 and a closed `explainer { }` so intermediate builds could work; Task 2 inserts Act 2 slides before the closing brace.

- [ ] **Step 4: Self-review against Global Constraints**

Verify id/route/topic/hub strings match the spec verbatim; no invented knobs; no test files touched; no commit.

- [ ] **Step 5: Write report**

Report status DONE (or DONE_WITH_CONCERNS). Do not commit.

---

### Task 2: Act 2 Flow — sampling through pressure (slides 6–13)

**Files:**
- Modify: `apps/explainers/decks-flow/tstar-warmup.flow` (insert slides before closing `}`)

**Interfaces:**
- Consumes: Act 1 file from Task 1; `WindowTStarSampler`, `seed_for_trace_lane`, `trace_duration_us`, `chop_trie_at_tstar`, `PermutationDraw` / `GraphPressureRecycle` behavior from source
- Produces: slides 6–13 appended inside the same explainer

- [ ] **Step 1: Read source anchors before writing claims**

Read (do not modify):
- `rust/runtime/src/graph/tstar.rs` (sampler, seed, duration, PermutationDraw)
- `rust/runtime/src/graph/snapshot.rs` (chop rules — drop `< t*`, re-root, prompt path)
- `rust/runtime/src/graph/warmup_handoff.rs` (handoff fields — for forward references only)
- Knobs: `trajectory_start_min_ratio`, `trajectory_start_max_ratio`, `t_star_random_seed`

- [ ] **Step 2: Author slides 6–13**

6. **Duration & window** — `trace_duration_us` = max `arrival_offset_us`; window = ratios × duration
7. **`WindowTStarSampler`** — uniform in `[lo, hi]`; `dur ≤ 0` or `hi ≤ lo` → exact instant; float µs
8. **Lane-salted seed** — `SHA256("{base}:{trace_id}:{lane}")` → NumPy PCG64; base is `t_star_random_seed`
9. **Trie chop** — drop nodes with `arrival_offset_us < t*`; re-root survivors from `START` with `min_start_delay_us = arrival - t*`
10. **Inputs & channels** — drop requirements whose source was chopped (avoid deadlocks)
11. **Prompt path unchanged** — pre-`t*` segments stay in the path for exact resume prompt
12. **Cache-pressure warmup** — extended warmup replays post-`t*` under pressure lanes
13. **`GraphPressureRecycle` + `PermutationDraw`** — Sequential / Shuffle / Random; shuffle/random child seeds from **run root** (`random_seed`), not `t_star_random_seed`

Reuse the `n0…n3` motif: slides 9–11 should show the cut removing left nodes and re-root edges.

Each slide: full narration/points/scene/timeline as in Task 1.

- [ ] **Step 3: Self-review accuracy**

Especially: do not claim shuffle/random seeds come from `t_star_random_seed` (they do not — see `tstar.rs` comments).

- [ ] **Step 4: Write report — no commit**

---

### Task 3: Act 2 Flow — handoff, resume, edges, source map (slides 14–20)

**Files:**
- Modify: `apps/explainers/decks-flow/tstar-warmup.flow`

**Interfaces:**
- Consumes: `GraphWarmupHandoff` / `LaneHandoff` field semantics from `warmup_handoff.rs`; resume behavior from `graph_phase_runtime.rs` comments near handoff consumers
- Produces: complete ~20-slide deck ready to compile

- [ ] **Step 1: Read handoff + resume anchors**

Read:
- `rust/runtime/src/graph/warmup_handoff.rs` (all `LaneHandoff` / `GraphWarmupHandoff` field docs)
- Relevant resume/handoff sections in `rust/runtime/src/engine/graph_phase_runtime.rs` (search `pressure_lane_count`, `corpus_cursor`, `instance_id`)

- [ ] **Step 2: Author slides 14–20**

14. **Drain & handoff** — fields: `template_trace_id`, `instance_id`, `t_star_us`, `executed_node_ids`, `return_wall_us`; plus deck-level `drain_end_wall_us`, `corpus_cursor`, `pressure_lane_count`
15. **Why `instance_id` is reused** — cache-bust marker continuity (digest of `credit.trace_id` / `build_trace_instance_marker`); avoid cold prefill behind fresh `.0`
16. **Profiling resume** — frontier chop + skip executed; residual delays from return walls
17. **Lanes with no handoff entry** — completed-at-drain under `pressure_lane_count` → fresh-start next cursor template at `t*=0`
18. **Corpus cursor** — pressure → profiling continuity; single-pass profiling may ignore for full-corpus coverage
19. **Abort / edge cases** — warmup abort; `t* ≤ 0` returns graph unchanged; collapsed `[0,0]` window
20. **Source map** — cite `tstar.rs`, `snapshot.rs`, `warmup_handoff.rs`, `graph_phase_runtime.rs`, knobs `trajectory_start_min_ratio` / `trajectory_start_max_ratio` / `t_star_random_seed`

- [ ] **Step 3: Final Flow polish**

- Ensure explainer block closes once
- Unique root ids across all slides (prefix `s0-`…`s19-` or similar)
- Every rendered slide has non-empty timeline
- Hub + metadata still match spec

- [ ] **Step 4: Write report — no commit**

---

### Task 4: Registry wiring + package build + assert gates

**Files:**
- Modify: `apps/explainers/src/core/deck-registry.ts` — add to `EXPECTED_DECK_ROUTES`
- Modify: `apps/explainers/scripts/assert-deck-packages.mjs` — add to `EXPECTED_DECK_IDS`
- Create (via build): `apps/explainers/src/decks-generated/tstar-warmup.package.json`

**Interfaces:**
- Consumes: complete `tstar-warmup.flow` from Task 3
- Produces: registered, compiled, assert-clean deck package

- [ ] **Step 1: Register the deck**

In `deck-registry.ts`, append:

```ts
  ["tstar-warmup", "/tstar-warmup"],
```

to `EXPECTED_DECK_ROUTES` (keep existing entries; do not reorder unless required).

In `assert-deck-packages.mjs`, append `"tstar-warmup"` to `EXPECTED_DECK_IDS`.

Do **not** edit any `*.test.ts` / `*.test.tsx` files even if they hardcode deck counts.

- [ ] **Step 2: Build packages**

From repo root:

```bash
make build-explainer-packages
```

Expected: succeeds and writes `apps/explainers/src/decks-generated/tstar-warmup.package.json`.

If compile errors, fix **only** `tstar-warmup.flow` (or the two registry/assert list lines) — do not “fix” by weakening other decks.

- [ ] **Step 3: Run assert gate**

```bash
cd apps/explainers && npm run assert:deck-packages
```

Expected: exit 0, including the new package (non-empty narration; timelines present).

Optionally:

```bash
make assert-explainer-packages
```

If IR verifier fails only on the new deck, fix the `.flow` scenes; if unrelated decks fail from pre-existing WIP, note DONE_WITH_CONCERNS with the unrelated failure list — do not drive-by fix other decks unless required for the gate to see the new package.

- [ ] **Step 4: Manual checklist (report evidence)**

Confirm and record in the report:
1. Package id/route are `tstar-warmup` / `/tstar-warmup`
2. Slide count ≈ 20 covering Act 1 + Act 2 topics from the spec
3. No `MentalModel.tsx` added
4. No test files modified (`git status` should show zero `*.test.*` under this task)

- [ ] **Step 5: Write report — no commit**

---

## Self-review (plan author)

1. **Spec coverage:** Identity, two-act outline, visual rules, accuracy rules, full product wiring, no-tests, verification via build/assert — all mapped to Tasks 1–4.
2. **Placeholders:** None intentionally left; slide bodies are content-specified by topic with scene conventions rather than 2k lines of inline Flow (authoring agents must write the Flow).
3. **Type/name consistency:** `tstar-warmup`, `/tstar-warmup`, knobs `trajectory_start_*_ratio` + `t_star_random_seed` consistent across tasks.

## Execution

Controller: use **subagent-driven-development** with model **`cursor-grok-4.5-high`** for implementers and reviewers. Fresh progress ledger for this plan (do not confuse with the completed `flow-core-geometry-animation` ledger). No commits unless explicitly requested.
