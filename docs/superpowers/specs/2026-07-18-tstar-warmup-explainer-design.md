<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# t* Warmup Explainer Design

**Date:** 2026-07-18
**Status:** Approved
**Deliverable:** New Flow-backed explainer deck under `apps/explainers`
**Authoring:** `.flow` only (packages-only registry path)

## Goal

Ship a hub-visible explainer that teaches how AIPerf graph **trajectory `t*` warmup** works: deterministic per-(trace, lane) sampling, trie snapshot chop, cache-pressure recycle, and profiling handoff resume — with a short contrast against ordinary request-count / duration warmup phases.

Audience is mixed: Act 1 is a scannable mental model; Act 2 is engineer-depth machinery grounded in runtime source.

## Locked decisions

| Decision | Choice |
|---|---|
| Audience | Both (mental model first, then deep internals) |
| Scope | Full pipeline (sample → chop → pressure → handoff → resume) **plus** brief ordinary-warmup contrast |
| Length | Long-form ~15–25 slides (peer of `cellular-algorithms` / `slurm-velo`) |
| Narrative | Two-act structure; one toy multi-turn trace as a recurring motif in Act 2 |
| Shipping | Full product path: `.flow` + registry route + generated package + hub card |
| Tests | **None** — no new unit/e2e/registry tests and no extensions to existing test files |
| Escape hatch | None — no `MentalModel.tsx` |

## Identity & product wiring

| Field | Value |
|---|---|
| Source | `apps/explainers/decks-flow/tstar-warmup.flow` |
| Package | `apps/explainers/src/decks-generated/tstar-warmup.package.json` |
| `id` | `tstar-warmup` |
| `route` | `/tstar-warmup` |
| `topic` | `graph-runtime` |
| `storagePrefix` | `tstar-warmup-explainer` |
| `classPrefix` | `deck-tstar-warmup` |
| `eyebrowLabel` | `T* WARMUP` |
| `startGateTitle` | `t* warmup & cache-pressure handoff` |
| Hub highlight | `t* warmup` |
| Hub title | `snapshot · pressure · handoff` |
| Hub description | How graph runs warm KV via a deterministic `t*` chop, cache-pressure recycle, and resume handoff — vs ordinary request-count warmup. |

### Wiring checklist

1. Author `tstar-warmup.flow` with embedded `@scene` + timelines on diagram slides.
2. Add `["tstar-warmup", "/tstar-warmup"]` to `EXPECTED_DECK_ROUTES` in `apps/explainers/src/core/deck-registry.ts`.
3. Add `"tstar-warmup"` to `EXPECTED_DECK_IDS` in `apps/explainers/scripts/assert-deck-packages.mjs`.
4. Run `make build-explainer-packages` to emit the DeckPackage.
5. Hub loads via packages-only registry (no MentalModel).

Do **not** add or update Vitest/registry/e2e tests for this deck. Assert scripts and IR gates are operational gates, not product tests.

## Narrative approach

**Two-act + motif (approved):**

- **Act 1 (~5 slides):** why graph warmup exists; ordinary phase warmup vs `t*`; the timeline chop picture; what stays warm on the server; two phases sharing one seeded plan.
- **Act 2 (~15 slides):** sampler window → lane-salted RNG → trie chop → channel/input rescoping → unchanged prompt path → cache-pressure recycle → handoff payload → instance-id continuity → profiling resume → missing-lane fresh-start → corpus cursor → abort/edge cases → source map.

Recurring motif: a 4-turn toy DAG (`n0…n3`) on a horizontal arrival timeline with a vertical `t*` rule. Later slides drop left-of-cut nodes and show re-root edges from `START`.

## Slide outline (~20)

### Act 1 — Mental model

1. **Why graph warmup** — cold KV / cold prefix vs measuring a mid-trace resume.
2. **Ordinary warmup ≠ `t*`** — request-count/duration phases discard metrics; `t*` partitions a recorded trajectory.
3. **The `t*` picture** — arrivals timeline; vertical cut; pre-`t*` history vs post-`t*` live set.
4. **What stays warm** — server holds KV for executed prefix; client still materializes full prompt path.
5. **Two phases, one plan** — warmup builds state; profiling resumes the same seeded `t*` (not re-drawn).

### Act 2 — Machinery

6. **Duration & window** — `trace_duration_us`, `[start_min_ratio, start_max_ratio]`.
7. **`WindowTStarSampler`** — uniform draw; collapsed / `dur ≤ 0` → exact instant; float µs (no integer truncation).
8. **Lane-salted seed** — `SHA256("{base}:{trace_id}:{lane}")` → NumPy PCG64.
9. **Trie chop** — drop `arrival_offset_us < t*`; re-root survivors from `START` with residual delay.
10. **Inputs & channels** — drop requirements on chopped predecessors; avoid `await_inputs` deadlocks.
11. **Prompt path unchanged** — pre-`t*` segments remain so the resume prompt is exact.
12. **Cache-pressure warmup** — redispatch `rewrite_for_warmup`'s flattened boundary-priming credit under pressure lanes / recycle (not the post-`t*` live set).
13. **`GraphPressureRecycle` + `PermutationDraw`** — sequential / shuffle / random cursor continuity.
14. **Drain & handoff** — `LaneHandoff`: template, `instance_id`, `t_star_us`, executed nodes, return walls.
15. **Why `instance_id` is reused** — cache-bust marker continuity (avoid cold prefill behind a fresh `.0`).
16. **Profiling resume** — frontier chop + skip executed; residual re-root delays from return walls.
17. **Lanes with no handoff entry** — completed-at-drain → fresh-start next cursor template, `t*=0`.
18. **Corpus cursor** — pressure → profiling wrap; single-pass profiling may ignore for full coverage.
19. **Abort / edge cases** — warmup abort, `t* ≤ 0` full replay, collapsed window.
20. **Source map** — `tstar.rs`, `snapshot.rs`, `warmup_handoff.rs`, graph phase runtime (+ real config knobs only).

Slides may be split or lightly renamed during authoring if a scene needs more room, but the topic coverage above is required.

## Visual conventions

- Match existing Flow decks (`segment-pools`, `dynosim`): header bar + caption; panels / elbows / connectors; sparse `motion.signal` on pipeline beats.
- Color roles:
  - Pre-`t*` / history: muted secondary
  - `t*` cut: accent warning
  - Post-`t*` / live: accent primary or green
  - Handoff payload: tertiary
- Every slide: non-empty `narration` and `points`.
- Every slide with `render`: non-empty `scene.timeline` (SceneRenderer contract / assert-deck-packages).

## Accuracy rules

- Ground claims in executable code and comments:
  - `rust/runtime/src/graph/tstar.rs`
  - `rust/runtime/src/graph/snapshot.rs`
  - `rust/runtime/src/graph/warmup_handoff.rs`
  - Graph phase runtime / pressure recycle callers under `rust/runtime/src/graph/` and `rust/runtime/src/engine/`
- Cite those paths in slide `points` where other decks cite evidence.
- Use only real Config v2 / protocol field names; do not invent knobs.
- Keep three concepts distinct in copy:
  1. Ordinary phase warmup (request count / duration)
  2. Trajectory `t*` partition / chop
  3. Cache-pressure extension + handoff

## Out of scope

- Cellular cell-merge / fold details
- HTTP connection-pool warmup
- Dynosim clock specifics (beyond an optional one-line “same Clock seam” mention)
- New tests of any kind
- Fern/docs tutorial rewrite (`docs/tutorials/warmup.md` stays the ordinary-warmup guide)

## Verification (no tests)

Done when:

1. `tstar-warmup.flow` compiles via `make build-explainer-packages`.
2. Registry lists `tstar-warmup` → `/tstar-warmup` and the hub card appears.
3. `make assert-explainer-packages` (or equivalent assert + IR gate) passes for the new package.
4. Manual skim: Act 1 readable alone; Act 2 matches comments/behavior in `tstar` / `snapshot` / `warmup_handoff`.
5. No `MentalModel.tsx` and no new/updated test files for this work.

## Implementation notes

After this spec is approved for implementation, produce a writing-plans task breakdown covering: Flow authoring by act, registry/assert id lists, package build, and manual + assert verification — still with **zero** test-file changes.
