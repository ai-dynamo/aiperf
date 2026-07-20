# Remaining Spacing Scout — Round 4 (Post R3 Fix A–E)

**Date:** 2026-07-20  
**Scope:** Read-only reconnaissance in `apps/explainers` after Round 3 fixes (catalog variant relayout, viewport escape remediation, avatar intrinsic minimums, product header→content spacing, Node resolved-bounds parity).  
**Method:** Per-deck `npm --prefix apps/explainers run flow-verifier:ir -- --deck <id>` (full-repo run blocked by `sdk-diagram-catalog.flow` compile failure); cross-check against R3 scout baseline and fixer reports.

---

## Executive summary

Round 3 moved the needle materially but **187 resolution warnings remain** across 13 verifiable decks (175 sibling overlaps + 12 viewport escapes). The work surface has shifted:

1. **Resolver chrome debt dominates the SDK catalog** — 37 warnings in `sdk-generic-catalog.flow`; **15/33 overlaps (~45%)** are intentional-looking **gauge/progress/meter `__track` × `__value` sibling collisions** emitted by the factory, not deck Y spacing.
2. **Status-chip overlays are the largest product-deck cluster** — **29/34 overlaps** in `cellular-algorithms.flow` are `algo-*` × `st-*` badge pairs at the same authored coordinates (R3-D intentionally deferred). **No status-chip pattern in `aiperf-vs-locust.flow`** — that deck’s 26 overlaps are thread-chip stacks, dual-column compare layout, and header-adjacent crowding.
3. **Viewport escapes mostly cleared in R3-B scope** — repo escapes **45 → 12** (−73%); **12 stragglers** across 7 decks, including 4 inside `sdk-generic-catalog` (paragraph + rating slides).
4. **Avatar / geometry parity / header-text resolvers are closed** — R3-C avatar floor, R3-E snapshot parity, and post–final-review `core.header` / `core.text` resolvers are in place. **MediaObject slot sizing and track/value chrome layout remain open.**
5. **`sdk-diagram-catalog.flow` blocks the full IR gate** — compile fails with `PARSE_UNEXPECTED_TOKEN … found: 'draw'`; per-deck counts for that deck are unknown until parse is fixed.

---

## Verification baseline

### Full-repo gate status

```bash
cd apps/explainers
npm run flow-verifier:ir
# FAIL — sdk-diagram-catalog.flow parse error (draw token)
```

### Per-deck counts (2026-07-20, `--ir-only --deck <id>`)

| Deck | Overlaps | Escapes | Total | R3 scout overlaps | Δ overlaps |
|------|--------:|--------:|------:|------------------:|-----------:|
| sdk-generic-catalog.flow | 33 | 4 | **37** | 59 | **−26** |
| cellular-algorithms.flow | 34 | 2 | **36** | 44 | **−10** |
| aiperf-vs-locust.flow | 26 | 0 | **26** | 32 | **−6** |
| flow-sdk-examples.flow | 21 | 1 | **22** | 24 | **−3** |
| slurm-velo.flow | 14 | 1 | **15** | 14 | 0 |
| synthetic-dataset-generator.flow | 13 | 1 | **14** | 13 | 0 |
| cellular-internals.flow | 13 | 0 | **13** | 9 | **+4** |
| segment-pools.flow | 12 | 1 | **13** | 12 | 0 |
| rust-architecture-atlas.flow | 11 | 0 | **11** | 11 | 0 |
| tstar-warmup.flow | 3 | 2 | **5** | 2 esc | escapes +0 net |
| velo-deep-dive.flow | 2 | 0 | **2** | — | — |
| dynosim.flow | 2 | 0 | **2** | — | — |
| rust-architecture.flow | 1 | 0 | **1** | — | — |
| sdk-diagram-catalog.flow | **blocked** | **blocked** | **?** | 49 authored heights | — |
| **Totals (13 decks)** | **175** | **12** | **187** | **233 + 45 = 278** | **−58 ov, −33 esc** |

R3 scout used a single full-repo pass (278 resolution warnings). Round 4 per-deck aggregation yields **187** on verifiable decks — a **33% reduction**. `cellular-internals` overlap count rose (+4) likely because R3-E now forwards canonical resolver diagnostics that were previously invisible to the Node verifier.

---

## 1. `sdk-generic-catalog` — remaining 37 warnings

R3-A reduced overlaps **59 → 37 total warnings** (reported 37 overlaps; current split is **33 overlaps + 4 escapes**). Variant-column quote/icon-label/list stacks are largely cleared. Remaining debt clusters as follows.

### Overlap classification (33 sibling overlaps)

| Category | Count | Representative nodes | Root cause |
|----------|------:|----------------------|------------|
| **Gauge/progress/meter `__track` × `__value`** | **15** | `gauge-hero__track`/`__value`, `progress-v1__*`, `meter-v3__*`, `open-g__*`, `final-progress__*` | Factory emits track and value as **absolute siblings occupying the same band**; resolver reports collision even when visually stacked |
| **Opener / chapter intro** | **3** | `open-copy`/`open-g`, `open-copy`/`open-cap`, `c1-shape`/`c1-title` | Dense opener slide 0–1 composition; gauge on opener shares track/value class above |
| **Finale slide (53)** | **3** | `final-h`/`final-bc`, `final-h`/`final-tools`, `final-avatar`/`final-owner-label` | Multi-band product mockup; header row intersects breadcrumb/toolbar; avatar label band |
| **MediaObject (52)** | **5** | `media-av`/`media-props`, `media-av`/`media-ok`, `media-server`/`media-server-c`, `media-file`/`media-file-c` | Nested avatar + icon-label bodies without intrinsic **media-slot** floor; caption labels collide |
| **Other variant/composition** | **7** | `timeline-hero`/`timeline-v*`, `rating-v1`/`rating-v2`, `c2-copy`/`c2-code`, `c7-av`/`c7-m` | Residual authored Y/X; timeline hero height vs right column; horizontal rating row still too tight |

### Viewport escapes (4)

| Slide | Nodes | Issue |
|-------|-------|-------|
| 13 paragraph | `para-v2`, `para-v2-l` | Variant column pushed too low after R3-A relayout |
| 46 rating | `rating-v3`, `rating-v3__rating-4` | Horizontal trio + resolved star row height exceeds 400px |

---

## 2. Status-chip / badge overlays

### `cellular-algorithms.flow` — **29/34 overlaps involve `st-*`**

Pattern repeats on slides 1–11: status chip authored at the **same `(x,y)` as its host panel** (`algo-built`/`st-built`, `dispatch`/`st-d`, …). R3-D explicitly left these untouched (overlay semantics).

| Metric | Value |
|--------|------:|
| Total overlaps | 34 |
| `st-*` badge × host overlaps | **29** |
| Non-badge composition overlaps | **5** (`term`/`retain`, `term`/`fold`, `allow`/`upload`, `rt-eligibility`/`rt-ownership`, `rt-distribution`/`rt-execution`) |
| Viewport escapes | **2** (`velo` slide 2, `barrier` slide 9 — both also overlap their `st-*` badge) |

Slide 0 (`map-band` / chapter grid) — **no sibling-overlap warnings** in Round 4 (R3 scout’s band×chapter pattern no longer appears; one `SCENE_ROUTE_FALLBACK` on `ch-link-0` remains).

**Fix fork:** (a) teach overlap diagnostic to ignore `layout.overlay: true` badge chrome, or (b) offset `st-*` nodes +8px X / corner-anchor in the deck.

### `aiperf-vs-locust.flow` — **no status-chip overlay class**

26 overlaps are unrelated:

| Slide | Pattern | Example pairs |
|-------|---------|---------------|
| 0 | Locust thread chips stack | `s1-t1`/`s1-t2`, `s1-t*`/`s1-tdots` |
| 0 | AIPerf worker column | `s1-aiperf`/`s1-c1`, `s1-c1`/`s1-c2`, … |
| 5 | Header-adjacent bus | `s6-coord`/`s6-bus`, `s6-coord`/`s6-dset` |
| 6–7 | Stepper/worker/condition rows | `s7-steps`/`s7-t1`, `s8-c1`/`s8-c2`, … |
| 10 | Dual-column compare | `s11-locust`/`s11-ltime`, `s11-aiperf`/`s11-atime` |

R3-D cleared all `s*-hdr` × first-content pairs (11 → 0). Remaining 26 are **composition spacing**, not badge-on-panel overlays.

---

## 3. Other decks — high overlap / escape hotspots

### Viewport escapes (12 total)

| Deck | Count | Node IDs |
|------|------:|----------|
| sdk-generic-catalog | 4 | `para-v2`, `para-v2-l`, `rating-v3`, `rating-v3__rating-4` |
| tstar-warmup | 2 | `s14-fields`, `s20-phase` |
| cellular-algorithms | 2 | `velo`, `barrier` |
| flow-sdk-examples | 1 | `s2-note` (regression vs R3-B claim — bottom row still escapes) |
| slurm-velo | 1 | `s4-cells` |
| segment-pools | 1 | `domain-trace` |
| synthetic-dataset-generator | 1 | `media-join` |
| cellular-internals | **0** | R3-B merge slide fix held |

### Notable overlap decks beyond catalog + cellular-algorithms

| Deck | Overlaps | Dominant pattern |
|------|--------:|------------------|
| aiperf-vs-locust | 26 | Dual-column + thread/worker chip stacks (§2) |
| flow-sdk-examples | 21 | Slide 4 curve-matrix: `rx1-hdr` × nine column nodes; slide 1 `s2-hdr`/`s2-callout` |
| slurm-velo | 14 | Mixed panel stacks (unchanged since R2) |
| synthetic-dataset-generator | 13 | Pipeline card Y pitch vs resolver growth |
| segment-pools | 12 | Domain band layout |
| cellular-internals | 13 | **Header→content not R3-D scoped**: `hdr*`/`phase*`, `hdr*`/`chip*`; merge slide `cell180–183` stack collisions |
| rust-architecture-atlas | 11 | Residual edge bbox + panel pairs (R3-D cleared `hdr10`/`ext` only) |

---

## 4. Avatar / factory / layout — closed vs open

| Item | Round | Status | Evidence |
|------|-------|--------|----------|
| Avatar intrinsic floor (`resolveAvatarLayout`) | R3-C | **Closed** | `layout.ts:459-487`; min side 40px |
| Node verifier resolved-bounds parity | R3-E | **Closed** | `geometry.mjs` consumes compile snapshot; `verify-geometry.test.ts` passes |
| `core.header` / `core.text` intrinsic sizing | post-final-review | **Closed** | `layout.ts:568-602`, registered in `LAYOUT_CAPABILITIES` |
| Catalog variant-column relayout | R3-A | **Mostly closed** | 59 → 33 overlaps; escapes on para/rating remain |
| Viewport escape (cellular-internals, flow-sdk tail) | R3-B | **Partial** | cellular-internals escapes 0; flow-sdk `s2-note` escape persists |
| Product header→content Y | R3-D | **Partial** | aiperf hdr collisions 0; cellular-internals/rust-arch/product tails not in scope |
| **MediaObject media-slot minimum** | — | **Open** | Slide 52 five overlaps; no `resolveMediaObjectLayout` |
| **Gauge/progress/meter track/value sibling layout** | — | **Open** | 15 catalog overlaps; factory in `generic/catalog.ts:1312+` |
| Duplicate text metric constants | R3 scout §6 | **Open (low)** | `generic/chrome.ts` still local `TITLE_HEIGHT = 22` vs `text-metrics.ts` imports elsewhere |
| `sdk-diagram-catalog` compile | — | **Blocked** | Full IR gate fails before deck enumeration |

---

## Top 6 disjoint fix domains (Round 4)

Ranked for parallel workers; together they cover **all 187** verifiable warnings plus the blocked 14th deck.

---

### Domain 1 — Track/value chrome sibling layout (resolver)

**Scope:** Gauge, progress, meter, and opener gauge on catalog slides 0, 41–44, 53.  
**Owns:**
- `apps/explainers/src/flow/sdk/generic/catalog.ts` (track/value box emission)
- `apps/explainers/src/core/diagram/capabilities/layout.ts` and/or `capabilities/chrome.ts` (stack track+value in one layout box or mark value as non-sibling overlay)

**Acceptance:**
```bash
npm --prefix apps/explainers run flow-verifier:ir -- --deck sdk-generic-catalog 2>&1 \
  | rg 'SCENE_ABSOLUTE_SIBLING_OVERLAP.*__track|__value' | wc -l
# → 0
```
Catalog track/value overlaps **15 → 0**. No new viewport escapes on slides 41–44.

---

### Domain 2 — Catalog composition + MediaObject (deck + optional resolver)

**Scope:** `sdk-generic-catalog.flow` slides 0–1, 11, 39, 46, 48, 52–53; optional `resolveMediaObjectLayout`.  
**Owns:**
- `apps/explainers/decks-flow/sdk-generic-catalog.flow`
- Optionally `apps/explainers/src/core/diagram/capabilities/layout.ts` (media-object presentation group)

**Acceptance:**
```bash
npm --prefix apps/explainers run flow-verifier:ir -- --deck sdk-generic-catalog 2>&1 \
  | rg 'SCENE_ABSOLUTE_SIBLING_OVERLAP|SCENE_VIEWPORT_ESCAPE' | wc -l
# → 0
```
All **37 → 0** warnings on that deck (absorbing Domain 1 leftovers if split across workers: coordinate so track/value fixes land first, then deck pass mops remainder).

---

### Domain 3 — Cellular-algorithms status-chip overlay policy

**Scope:** Slides 1–11 (+ escapes on slides 2, 9).  
**Owns:**
- `apps/explainers/decks-flow/cellular-algorithms.flow` (reposition `st-*` **or**)
- `apps/explainers/src/flow/sdk/generic/chrome.ts` + `capabilities/layout.ts` (emit `layout.overlay: true` on status chips)
- `apps/explainers/src/core/diagram/resolution/resolve-scene.ts` (exclude overlay siblings from `SCENE_ABSOLUTE_SIBLING_OVERLAP` if policy choice)

**Acceptance:**
```bash
npm --prefix apps/explainers run flow-verifier:ir -- --deck cellular-algorithms 2>&1 \
  | rg 'SCENE_ABSOLUTE_SIBLING_OVERLAP.*st-' | wc -l
# → 0
npm --prefix apps/explainers run flow-verifier:ir -- --deck cellular-algorithms 2>&1 \
  | rg 'SCENE_VIEWPORT_ESCAPE' | wc -l
# → 0
```
Deck total warnings **36 → ≤5** (only non-badge composition pairs may remain if intentionally dense).

---

### Domain 4 — Header→content spacing (cellular-internals + flow-sdk-examples)

**Scope:** Extends R3-D pattern to decks not touched in Round 3.  
**Owns:**
- `apps/explainers/decks-flow/cellular-internals.flow` (`hdr*`/`phase*`, `hdr*`/`chip*`, merge slide `phase18`/`cell180`)
- `apps/explainers/decks-flow/flow-sdk-examples.flow` (`s2-hdr`/`s2-callout`, optional `s2-note` Y to kill escape)

**Acceptance:**
```bash
npm --prefix apps/explainers run flow-verifier:ir -- --deck cellular-internals 2>&1 \
  | rg 'hdr[0-9]+.*OVERLAP|OVERLAP.*hdr[0-9]+' | wc -l
# → 0
npm --prefix apps/explainers run flow-verifier:ir -- --deck flow-sdk-examples 2>&1 \
  | rg 's2-hdr.*OVERLAP|SCENE_VIEWPORT_ESCAPE.*s2-note' | wc -l
# → 0
```
Target: cellular-internals **13 → ≤3** (merge cell stack may need separate Y pass); flow-sdk slide 1 **hdr overlap 0**, **s2-note escape 0**.

---

### Domain 5 — Product narrative deck composition (aiperf-vs-locust + mid-tier decks)

**Scope:** Dual-column and chip-stack decks.  
**Owns:**
- `apps/explainers/decks-flow/aiperf-vs-locust.flow` (26 overlaps — slides 0, 5–7, 10)
- `apps/explainers/decks-flow/rust-architecture-atlas.flow` (11)
- `apps/explainers/decks-flow/slurm-velo.flow` (15)
- `apps/explainers/decks-flow/synthetic-dataset-generator.flow` (14)
- `apps/explainers/decks-flow/segment-pools.flow` (13)
- `apps/explainers/decks-flow/flow-sdk-examples.flow` slide 4 curve-matrix (`rx1-hdr` × columns — **21 overlaps**, coordinate with Domain 4 for slide 1)

**Acceptance (per deck):**
```bash
npm --prefix apps/explainers run flow-verifier:ir -- --deck <deck> 2>&1 \
  | rg 'SCENE_ABSOLUTE_SIBLING_OVERLAP|SCENE_VIEWPORT_ESCAPE' | wc -l
# → 0 for each owned deck
```
Repo overlap+escape subtotal for these six decks: **99 → 0**.

---

### Domain 6 — Viewport escape sweep + diagram-catalog unblock

**Scope:** Remaining **12 escapes** and IR gate restoration.  
**Owns:**
- Escape nodes listed in §3 (one node each in slurm-velo, segment-pools, synthetic-dataset-generator, tstar-warmup×2, flow-sdk `s2-note`, cellular-algorithms×2 — overlap with Domains 3–4)
- `apps/explainers/decks-flow/sdk-diagram-catalog.flow` + parser if needed (`PARSE_UNEXPECTED_TOKEN draw`)
- `apps/explainers/decks-flow/tstar-warmup.flow` (`s14-fields`, `s20-phase`)

**Acceptance:**
```bash
npm --prefix apps/explainers run flow-verifier:ir 2>&1 \
  | rg 'SCENE_VIEWPORT_ESCAPE' | wc -l
# → 0 (full repo, all 14 decks compile)
```
Full IR gate exits 0 on resolution warnings filter (or documents accepted overlay exceptions).

---

## Worker assignment matrix

| Domain | Primary files | Warnings addressed | Merge conflict risk |
|--------|---------------|-------------------:|--------------------|
| 1 Track/value chrome | `generic/catalog.ts`, `capabilities/layout.ts` | 15 | Low — resolver only |
| 2 Catalog composition | `sdk-generic-catalog.flow` | ~22 + 4 esc | Medium — same deck as 1 |
| 3 Cellular status chips | `cellular-algorithms.flow`, resolver policy | 36 | Low |
| 4 Header spacing | `cellular-internals.flow`, `flow-sdk-examples.flow` | ~34 | Low |
| 5 Product decks | 5 `.flow` files | ~99 | Low — disjoint paths |
| 6 Escapes + diagram catalog | 7 `.flow` + parser | 12 esc + gate | Medium — touches flow-sdk with Domain 4 |

**Suggested merge order:** 1 → 2 (catalog), 3 and 4 in parallel, 5 parallel per deck, 6 last (validates full gate).

---

## Polish list (if stopping early)

1. **Domain 1 only** — eliminates 45% of catalog overlaps with zero deck edits.
2. **Domain 3 badge policy** — highest single-deck product impact (36 warnings).
3. **Fix `sdk-diagram-catalog` parse** — restores full IR gate before claiming “zero warnings repo-wide.”

---

## Verification commands used

```bash
cd apps/explainers
npm run flow-verifier:ir                                    # fails on sdk-diagram-catalog
npm run flow-verifier:ir -- --deck sdk-generic-catalog      # 37 warnings
npm run flow-verifier:ir -- --deck cellular-algorithms      # 36 warnings
npm run flow-verifier:ir -- --deck aiperf-vs-locust         # 26 warnings

# Per-deck CSV (13 decks)
for deck in velo-deep-dive aiperf-vs-locust slurm-velo flow-sdk-examples \
  rust-architecture-atlas sdk-generic-catalog dynosim segment-pools \
  synthetic-dataset-generator cellular-algorithms cellular-internals \
  rust-architecture tstar-warmup; do
  ov=$(npm run flow-verifier:ir -- --deck "$deck" 2>&1 | rg -c SCENE_ABSOLUTE_SIBLING_OVERLAP || true)
  es=$(npm run flow-verifier:ir -- --deck "$deck" 2>&1 | rg -c SCENE_VIEWPORT_ESCAPE || true)
  echo "$deck overlaps=$ov escapes=$es total=$((ov+es))"
done
```

No production code was modified in this scout (report file only).
