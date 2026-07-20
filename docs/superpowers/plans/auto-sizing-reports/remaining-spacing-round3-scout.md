# Remaining Spacing Scout — Round 3 (Post Fix A–E + R2 Resolver)

**Date:** 2026-07-20  
**Scope:** Read-only reconnaissance in `apps/explainers` after Round 1 spacing fixes and Round 2 resolver work (diagram.* + presentation chrome, factory floors, product deck bumps, cellular merge tail, footer pad 60px).  
**Method:** Source grep, deck height sampling, `npm run flow-verifier -- --ir-only` (278 resolution warnings: 45 viewport escapes, 233 sibling overlaps).

---

## Executive summary

Round 2 closed the big resolver holes (`resolveDiagramLayout`, `resolvePresentationLayout` for code/quote/icon-label, subtitle/callout/legend, factory floors). **Most remaining crowding is now composition debt, not missing resolvers:**

1. **Catalog variant columns did not relayout after intrinsic growth** — `sdk-generic-catalog.flow` accounts for **59/233** sibling-overlap warnings; hero + variant stacks collide when quotes/icon-labels/paragraphs grow and caption labels stay at pre-growth Y.
2. **Tall absolute scenes exceed 700×400** — **45** `SCENE_VIEWPORT_ESCAPE` warnings; worst: `cellular-internals.flow` merge slide (11), `flow-sdk-examples.flow` (7), product decks with bottom notes/sinks.
3. **Avatar still identity-sized** — deliberate R2-A omission; no label/subtitle chrome to measure, but nested avatars in `MediaObject` rely on authored geometry only.
4. **`geometry.mjs` still reads raw `node.geometry`** — IR overlap/viewport checks use `resolveScene` snapshots, but `geomOf()` / `resolveEndpoint()` in the Node verifier remain pre-resolution.
5. **Duplicate metric constants** — `generic/chrome.ts` re-declares bands already exported from `text-metrics.ts` (sync comment only).

Authored `height=44` headers and many `height=60–70` multi-band boxes are **low priority**: resolvers treat them as minimums and grow at runtime. Deck counts below are audit residue, not blocking bugs.

---

## Round 2 baseline (what is already fixed)

| Fix | Evidence | Status |
|-----|----------|--------|
| Diagram.* intrinsic layout | `layout.ts:407-449`, `LAYOUT_CAPABILITIES` `1210-1217` | Done (R2-A) |
| Presentation sizing (code/quote/icon-label) | `layout.ts:455-505` | Done (R2-A) |
| Subtitle in panel/header resolvers | `layout.ts:361-405`, `541-571` | Done (R1-A / R2) |
| Factory floors (header 66, panel 70, card 88, note 48, label 22) | `generic/chrome.ts:63-88` | Done (R2) |
| Pipeline 120×64 | `topology.ts` (R1-C) | Done |
| Cellular merge tail card bumps | `cellular-internals.flow:807-814` cards at 88/90 | Partial — slide still escapes viewport |
| Footer pad 60px | `index.css` (R2-E) | Done |

---

## 1. Avatar (`core.group` + `presentation: avatar`) — still missing intrinsic layout

**Verdict:** Yes — known R2-A gap, still open.

`resolvePresentationLayout` explicitly handles only `code-block`, `quote`, and `icon-label`; all other presentations (including `avatar`) fall through to identity layout:

```455:467:apps/explainers/src/core/diagram/capabilities/layout.ts
export function resolvePresentationLayout(
  node: SceneNodeLike,
  children: readonly SceneNodeLike[],
): CapabilityLayout {
  const presentation = node.props?.presentation;
  if (
    presentation !== "code-block" &&
    presentation !== "quote" &&
    presentation !== "icon-label"
  ) {
    return resolveIdentityLayout(node, children);
  }
```

R2-A report documents the intentional omission: *"Avatar remains identity-sized because its chrome has no props-driven text."*

Chrome renders a circle only — no label band inside the node:

```157:166:apps/explainers/src/core/diagram/capabilities/chrome.ts
  if (presentation === "avatar") {
    return {
      rootBox: {
        id: `${node.id}__chrome`,
        geometry,
        radius: radiusOf(node, Math.max(geometry.width, geometry.height) / 2),
      },
      boxes: [],
      texts: [],
    };
  }
```

| Source | Geometry | Issue |
|--------|----------|-------|
| Catalog default | `48×48` (`catalog.ts:127`) | OK for icon-only marker |
| Deck hero | `130×130` (`sdk-generic-catalog.flow:465`) | Authored; no growth for long `label` prop if text chrome added later |
| Nested in MediaObject | `width/height` omitted (`flow:1027`) | Falls back to 48×48 factory — tight beside `IconLabel` body |
| Variant row | `58×58` ×3 (`flow:467-471`) | Fine statically; caption labels at fixed Y (`188`) assume 58px circle |

**Impact:** Low for current renders (circle-only). Becomes medium if avatar gains initials/label chrome or MediaObject expects intrinsic media slot sizing.

---

## 2. Authored `height=(40|44|48|52|56|58|60|62|64|70)` — sample counts by deck

**Verdict:** Mostly audit residue. Resolver minimums absorb growth; headers at 44 are the largest cluster.

### Total matches per deck (all node types)

| Deck | Total | Multi-band† | Header-only (44) |
|------|------:|------------:|-----------------:|
| sdk-generic-catalog.flow | 84 | 80 | 4 |
| sdk-diagram-catalog.flow | 49 | 36 | 13 |
| synthetic-dataset-generator.flow | 48 | 38 | 3 |
| velo-deep-dive.flow | 48 | 37 | 5 |
| flow-sdk-examples.flow | 35 | 14 | 20 |
| cellular-internals.flow | 30 | 9 | 21 |
| cellular-algorithms.flow | 25 | 25 | 0 |
| aiperf-vs-locust.flow | 24 | 13 | 4 |
| slurm-velo.flow | 22 | 6 | 8 |
| tstar-warmup.flow | 20 | 0 | 19 |
| dynosim.flow | 18 | 0 | 18 |
| rust-architecture.flow | 16 | 16 | 0 |
| segment-pools.flow | 12 | 5 | 6 |
| rust-architecture-atlas.flow | 11 | 11 | 0 |

†Panel/Card/diagram/CodeBlock/Quote/IconLabel/etc. or nodes with both `title` and `detail` in context.

### Notable multi-band leftovers (non-header)

| File:line | Node | Authored | Notes |
|-----------|------|----------|-------|
| `sdk-generic-catalog.flow:321-322` | `code-v1/v2` | `215×70` | Resolver grows multi-line `pre`; variant column tight |
| `sdk-generic-catalog.flow:487-491` | `il-v1..v3` | `210×42` | Factory default 40 (`catalog.ts:128`); 42px authored, labels at +46/+46/+46 Y |
| `sdk-generic-catalog.flow:339-340` | `quote-v1/v2` | `215×80` | Growth → overlap with caption labels (verifier confirmed) |
| `velo-deep-dive.flow:1377-1427` | lane chips | `40×40` ×6 | Single-band; likely intentional compact markers |
| `synthetic-dataset-generator.flow:488-626` | pipeline cards | `60×60` / `70×70` | Resolver grows; static Y spacing may be tight |
| `cellular-internals.flow:807-814` | merge cards | `88×88` / `90×90` | R2 tail bump — still overflows viewport (see §8) |

**Headers at `height=44`:** ~300+ deck occurrences vs factory `HEADER_DEFAULT_GEOMETRY.height: 66` (`generic/chrome.ts:63`). `resolveHeaderLayout` (`layout.ts:541-571`) grows to ~62–66px at runtime — authored 44 is stale, not crowding.

---

## 3. Stacked absolute overlaps after growth

**Verdict:** Yes — primary Round 3 work surface. Verifier: **233** `SCENE_ABSOLUTE_SIBLING_OVERLAP` warnings.

### Overlap counts by deck (top)

| Deck | Overlaps |
|------|--------:|
| sdk-generic-catalog.flow | 59 |
| cellular-algorithms.flow | 44 |
| aiperf-vs-locust.flow | 32 |
| flow-sdk-examples.flow | 24 |
| slurm-velo.flow | 14 |
| synthetic-dataset-generator.flow | 13 |
| segment-pools.flow | 12 |
| rust-architecture-atlas.flow | 11 |
| cellular-internals.flow | 9 |

### High-signal collision patterns

**A. Catalog variant column — label Y not recomputed after sibling growth**

Quote slide: `quote-v1-l` at `y=192` collides with `quote-v2` starting `y=205` after `quote-v1` resolves taller than authored 80px:

```339:342:apps/explainers/decks-flow/sdk-generic-catalog.flow
      sdk.Quote(id = "quote-v1", text = "The graph made bottlenecks obvious.", x = 435, y = 105, width = 215, height = 80)
      sdk.Quote(id = "quote-v2", text = "Trust evidence over intuition.", x = 435, y = 205, width = 215, height = 80, variant = "success")
      sdk.Label(id = "quote-v1-l", text = "TESTIMONY", x = 435, y = 192, width = 215, height = 16)
      sdk.Label(id = "quote-v2-l", text = "MAXIM", x = 435, y = 292, width = 215, height = 16)
```

Verifier: `sdk-generic-catalog` slide `sdk.quote` — `"quote-v2" and "quote-v1-l" overlap`.

Same pattern on paragraph, list, property-list, status-card, empty-state, rich-text, table-row, icon-label slides (caption labels at fixed offset from authored height, not resolved bottom).

**B. Icon-label variant stack — 42px boxes, 46px label offsets**

```487:492:apps/explainers/decks-flow/sdk-generic-catalog.flow
      sdk.IconLabel(id = "il-v1", label = "Verified", icon = "check", x = 430, y = 100, width = 210, height = 42)
      sdk.Label(id = "il-v1-l", text = "STATE", x = 430, y = 146, width = 210, height = 14)
      sdk.IconLabel(id = "il-v2", label = "Vector store", icon = "database", x = 430, y = 170, width = 210, height = 42)
```

`resolvePresentationLayout` can widen/tall `il-v2` ("Vector store") → encroaches on `il-v2-l` at `y=216`.

**C. Product decks — intentional dual-column status pairs**

`cellular-algorithms.flow`: `algo-*` cards overlap sibling `st-*` status strips at same Y (44 overlaps) — paired annotation layout, not resolver bug; may need `layout.overlay` or X offset bump.

**D. Header vs first content sibling**

Multiple `aiperf-vs-locust` slides: resolved header height exceeds authored `y=78/100` content start (`s3-hdr`/`s4-hdr`/`s5-hdr` overlaps). Headers grew via `resolveHeaderLayout`; content Y not adjusted.

---

## 4. SDK generic catalog — remaining tight variants

| Slide | Lines | Issue |
|-------|-------|-------|
| `sdk.codeBlock` | 321-322 | Variants `215×70`; single-line OK, multi-line clips before resolver; column crowded |
| `sdk.quote` | 339-342 | Variant quotes + caption labels collide after growth (§3A) |
| `sdk.iconLabel` | 487-492 | Variants `210×42` (below factory 40); long labels grow into caption band |
| `sdk.avatar` | 467-472 | `58×58` variants + labels at `y=188`; static spacing |
| `sdk.mediaObject` / final card | 996, 1027 | Avatar in media slot without explicit size → 48×48 default; `IconLabel` in 70px-tall `MediaObject` |

Hero rows generally have adequate slack (e.g. `il-hero` `340×70`, `code-hero` via factory `320×140`). **Variant column on the right (x≈430) is the systematic tight zone.**

---

## 5. Verifier / `geometry.mjs` — resolved layout gap

**Verdict:** Partially addressed; `geometry.mjs` still raw.

| Layer | Uses resolved layout? | Evidence |
|-------|----------------------|----------|
| `compile-decks.ts` | Yes | Calls `resolveScene(scene)` → snapshot (`compile-decks.ts:117-129`) |
| `ir.mjs` overlap/viewport | Yes | `appendResolutionDiagnostics(snapshot.diagnostics)` (`ir.mjs:34-48`); node checks use `node.resolvedBounds` (`ir.mjs:480`) |
| `verify-geometry.ts` | Yes | `resolveSceneForGeometryVerification` → `resolveScene` (`verify-geometry.ts:28-31`) |
| **`geometry.mjs`** | **No** | `geomOf(node)` reads `node.geometry ?? node.layout` only (`geometry.mjs:216-224`); no layout-capability dispatch |

**Impact:** Connector endpoint snapping / obstacle indexing in pure `geometry.mjs` paths can disagree with rendered bounds when nodes grow post-resolution. IR routing checks that consume `resolvedBounds` are OK; standalone `geomOf` consumers are stale.

---

## 6. Duplicate constants drift

**Verdict:** Low-severity maintenance debt; values currently match.

| Constant | `text-metrics.ts` | `generic/chrome.ts` | `layout.ts` |
|----------|-----------------|---------------------|-------------|
| `SUBTITLE_HEIGHT` | `:15` export | `:60` local duplicate | imports text-metrics |
| `TITLE_HEIGHT` | `:13` | `:58` local | imports text-metrics |
| `DETAIL_HEIGHT` | `:14` | `:59` local | imports text-metrics |
| `INSET` | `:12` | `:57` local | imports text-metrics |
| `HEADER_DEFAULT` height | — | `66` (`:63`) | resolver computes ~62–66 |
| `SWIMLANE_LABEL_WIDTH` | — | — | `72` (`layout.ts:36`) |

`generic/chrome.ts:54` comment: *"kept in sync with diagram text-metrics / chrome layout"* — but no import; drift risk on next band tweak.

---

## 7. Swimlane label width 72 — too narrow?

**Verdict:** Latent only; no deck usage.

- Default gutter: `SWIMLANE_LABEL_WIDTH = 72` (`layout.ts:36-38`).
- SDK factory default: `labelWidth: 72` (`generic/layout.ts:540`).
- **Zero** `sdk.Swimlane` / `Swimlane(` occurrences in `decks-flow/*.flow`.

Long lane titles would clip or force child width shrink (`resolveSwimlaneLayout` `layout.ts:1028-1067`). Not blocking until swimlanes appear in product decks.

---

## 8. Scene viewport 700×400 — overflow (`SCENE_VIEWPORT_ESCAPE`)

**Verdict:** Yes — **45** warnings; growth + absolute Y stacking.

Validation: `resolve-scene.ts:289-297` warns when resolved bounds exceed viewport.

### Escapes by deck

| Deck | Count |
|------|------:|
| cellular-internals.flow | 11 |
| flow-sdk-examples.flow | 7 |
| aiperf-vs-locust.flow | 3 |
| tstar-warmup.flow | 2 |
| cellular-algorithms.flow | 2 |
| synthetic-dataset-generator.flow | 1 |
| slurm-velo.flow | 1 |
| segment-pools.flow | 1 |

### Critical example — cellular merge slide (R2 tail still hot)

```807:814:apps/explainers/decks-flow/cellular-internals.flow
      sdk.Card(id = "cell180", ... x = 30, y = 100, width = 120, height = 88, ...)
      sdk.Card(id = "cell181", ... x = 30, y = 196, width = 120, height = 88, ...)
      sdk.Card(id = "cell182", ... x = 30, y = 292, width = 120, height = 88, ...)
      sdk.Card(id = "cell183", ... x = 30, y = 388, width = 120, height = 88, ...)
      sdk.Card(id = "agg1", ... x = 210, y = 340, width = 150, height = 88, ...)
      sdk.Panel(id = "sink", title = "external sink", x = 430, y = 490, width = 210, height = 52, ...)
```

- `cell183`: `388 + 88 = 476 > 400`
- `sink`: `490 + 52 = 542 > 400`
- Verifier also flags generated chrome parts (`cell183__title`, `__detail`, `__subtitle`) and merge arrow tip.

### Horizontal escape example

```696:696:apps/explainers/decks-flow/cellular-internals.flow
      sdk.Label(id = "note15", text = "keeps raw record artifacts available", x = 230, y = 290, width = 500, height = 20, ...)
```

`230 + 500 = 730 > 700` — triggers `SCENE_VIEWPORT_ESCAPE` on slide 15 despite modest height.

---

## Top 5 disjoint parallel fix domains (Round 3)

Ranked by user-visible impact; each is independently mergeable.

### Domain 1 — Catalog variant-column relayout pass
**Scope:** `sdk-generic-catalog.flow` only (+ optional shared variant-column Y helper in deck authoring docs).  
**Fix:** Re-space right-column variants using resolved-minimum heights (quote, icon-label, paragraph, list, status-card, empty-state, table-row slides). Bump caption labels to `resolvedBottom + gap`.  
**Evidence:** 59 verifier overlaps; §3A, §4.  
**Impact:** Highest — catalog is the SDK showcase.

### Domain 2 — Viewport escape remediation (tall compositions)
**Scope:** `cellular-internals.flow` merge slide, `flow-sdk-examples.flow`, scattered product slides.  
**Fix:** Compress vertical stack (reduce card Y pitch, shrink merge fan, move `sink` inside 400), widen viewport override where intentional (`scene.viewport`), or shorten bottom notes (`note15` width → ≤470).  
**Evidence:** 45 escapes; §8.  
**Impact:** High — literal clip at stage edge.

### Domain 3 — Avatar / MediaObject intrinsic minimums
**Scope:** `layout.ts` (`resolvePresentationLayout` or `resolveAvatarLayout`), `catalog.ts` default, `sdk-generic-catalog.flow` nested avatars.  
**Fix:** Square minimum from `max(authored, iconDiameter + pad)`; optional initials band if label prop should render. Media slot minimum ~56–64 when body contains IconLabel.  
**Evidence:** §1, R2-A known gap, `flow:1027`.  
**Impact:** Medium — prevents nested-composition clip.

### Domain 4 — Product deck header/content Y spacing
**Scope:** `aiperf-vs-locust.flow`, `cellular-algorithms.flow` (status pairs), `rust-architecture-atlas.flow`.  
**Fix:** Move first content row to `y ≥ 86` (post-header resolved height) or wrap in managed layout; cellular status pairs → overlay or +20px X on `st-*` nodes.  
**Evidence:** 32 + 44 + 11 overlaps; §3C–D.  
**Impact:** Medium — product narrative decks.

### Domain 5 — `geometry.mjs` resolved-bounds parity
**Scope:** `scripts/flow-verifier/geometry.mjs`, optionally extract shared `geomOfResolved` from `verify-geometry.ts`.  
**Fix:** Accept resolved snapshot bounds in `resolveEndpoint` / obstacle walks; deprecate raw `geomOf` for verifier snap checks.  
**Evidence:** §5.  
**Impact:** Medium for verifier fidelity; low for runtime rendering (SceneRenderer already resolves).

**Deferred (low ROI):** duplicate constant import cleanup (§6); swimlane 72 bump with no deck usage (§7); bulk header `44→66` deck edits (resolver handles).

---

## Polish list (if skipping domains — ≤3 items)

1. **Catalog quote + icon-label slides** — fix the two variant-column slides with confirmed verifier overlaps (`quote-v1-l`/`quote-v2`, `il-v2`/`il-v2-l`); highest density per line changed.
2. **Cellular merge slide** — pull `cell183`/`sink` inside 400px or add `scene.viewport height: 560` for that slide only.
3. **Import shared metrics in `generic/chrome.ts`** — replace local `TITLE_HEIGHT`/`SUBTITLE_HEIGHT`/… duplicates with `text-metrics.ts` imports to prevent next drift.

---

## Verification commands used

```bash
cd apps/explainers
npm run flow-verifier -- --ir-only 2>&1 | rg 'SCENE_VIEWPORT_ESCAPE|SCENE_ABSOLUTE_SIBLING_OVERLAP'
npm test -- --run src/core/diagram/resolution/resolve-scene-validation.test.ts
```

No production code was modified in this scout.
