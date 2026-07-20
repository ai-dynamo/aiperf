# Task 2 & 3 Scout Report — Diagram Node Auto-Sizing

**Date:** 2026-07-20
**Scope:** Read-only reconnaissance for Tasks 2 (intrinsic leaf layouts + container reflow) and 3 (verifier layout parity) from `docs/superpowers/plans/2026-07-20-diagram-node-auto-sizing.md`.
**Thoroughness:** Medium.

---

## Executive summary

1. **`indexSceneNodes` does not pre-resolve child capability layouts before parent placement.** Parents call `resolveCapabilityLayout` once with raw children; layout resolvers (notably `resolveRailLayout`) read each child's **authored** `geometry` via `geometryOf(child)`. Child overrides passed to the recursive `visit` are **parent-assigned slot positions**, not intrinsic leaf bounds. Task 2 must fix this in the indexer (preferred) or in every container resolver.

2. **`core.chip`, `core.panel`, and `core.note` have chrome hooks but no layout hooks.** They fall through to `resolveIdentityLayout` in the registry. Task 2 adds registry entries and new resolvers.

3. **Text-metrics constants are triplicated** across `layout.ts`, `chrome.ts`, and `desugar-scene-primitives.ts`. Task 2 targets the first two; the compiler desugar path is a separate convergence point.

4. **`verify-geometry.ts` reads authored geometry only** (`geomOf`); it has no hierarchical layout pass and no world-geometry index. Task 3 needs a recursive resolver mirroring `indexSceneNodes`, plus downstream consumers (`verify-deck.ts`, `resolveEndpoint`) switched to resolved world bounds.

5. **Hard-coded stepper widths in tests:** `layout.test.ts` expects total width `279`; `SceneRenderer.sdk-primitives.test.tsx` expects the **first chip** width `"80"`. Both must move to scale-aware values once Task 1 `text-metrics.ts` lands.

---

## 1. SceneRenderer — `indexSceneNodes` / `resolveContainerLayout`

### Does parent layout use child geometries after `resolveCapabilityLayout` on children?

**No.** The indexer resolves each node exactly once per visit. Children are **not** passed through their own `resolveCapabilityLayout` before the parent runs.

### Call chain

`resolveContainerLayout` is a thin wrapper around the registry:

```611:628:apps/explainers/src/core/diagram/SceneRenderer.tsx
function resolveContainerLayout(
  node: SceneNodeLike,
  parentGeom: SceneGeometryLike,
  children: readonly SceneNodeLike[] | undefined,
): Readonly<{
  parentGeom: SceneGeometryLike;
  childGeoms: readonly SceneGeometryLike[] | undefined;
}> {
  const members = Array.isArray(children) ? children : [];
  const layout = resolveCapabilityLayout(
    { ...node, geometry: parentGeom },
    members,
  );
  return {
    parentGeom: layout.bounds,
    childGeoms:
      members.length > 0 ? layout.childGeometries : undefined,
  };
}
```

`indexSceneNodes` calls it **before** recursing into children, passing the **original** child node objects (not pre-resolved):

```660:690:apps/explainers/src/core/diagram/SceneRenderer.tsx
    const kids = node.children;
    const { parentGeom: laidOutParent, childGeoms } = resolveContainerLayout(
      node,
      authored,
      kids,
    );
    // ...
    kids.forEach((child, index) => {
      const childOverride = childGeoms?.[index];
      if (local) {
        visit(child, worldGeom.x, worldGeom.y, true, childOverride, childAncestors);
      } else {
        visit(child, 0, 0, false, childOverride, childAncestors);
      }
    });
```

The registry dispatches to a single resolver with **no child pre-pass**:

```51:57:apps/explainers/src/core/diagram/capabilities/registry.ts
export function resolveCapabilityLayout(
  node: SceneNodeLike,
  children: readonly SceneNodeLike[],
): CapabilityLayout {
  const definition = NATIVE_SCENE_CAPABILITIES.get(capabilityOf(node));
  return (definition?.resolveLayout ?? resolveIdentityLayout)(node, children);
}
```

There are **zero** call sites of `resolveCapabilityLayout(child, …)` anywhere in the explainers tree (grep confirms).

### Do rails get intrinsic child widths today?

**No.** `resolveRailLayout` sums **authored** child widths from `geometryOf(child)`:

```188:206:apps/explainers/src/core/diagram/capabilities/layout.ts
export function resolveRailLayout(
  node: SceneNodeLike,
  children: readonly SceneNodeLike[],
): CapabilityLayout {
  const authored = geometryOf(node);
  if (children.length === 0) {
    return { bounds: authored, childGeometries: [] };
  }
  const direction = directionOf(node);
  const gap = styleNumber(node.style, "gap", 0);
  const childAuthored = children.map(geometryOf);
  const totalGap = gap * Math.max(children.length - 1, 0);
  const minWidth =
    childAuthored.reduce((sum, geometry) => sum + geometry.width, 0) +
    (direction === "row" ? totalGap : 0);
```

Row rails then assign **equal slots** from the expanded parent width, not per-child intrinsic widths:

```217:235:apps/explainers/src/core/diagram/capabilities/layout.ts
  const slot =
    direction === "row"
      ? Math.max((width - totalGap) / children.length, 0)
      : Math.max((height - totalGap) / children.length, 0);
  const childGeometries = childAuthored.map((geometry, index) =>
    direction === "row"
      ? {
          x: index * (slot + gap),
          y: 0,
          width: slot,
          height: geometry.height > 0 ? geometry.height : height,
        }
```

The same pattern holds for `resolveStackLayout` (lines 113–119) and `resolveLaneLayout` (lines 250–256): all use `geometryOf(child)` on raw nodes.

### What `childGeoms` actually are

When visiting a child, `childOverride` is the **parent layout's placement** for that index—not the result of resolving the child's own capability. For a chip inside a row rail, the child receives a **slot width** from the rail, then `resolveIdentityLayout` preserves that override on the chip visit.

### Render path uses the same resolver

The render-time path duplicates the same pattern:

```3661:3666:apps/explainers/src/core/diagram/SceneRenderer.tsx
  const kids = node.children;
  const { parentGeom: geom, childGeoms } = resolveContainerLayout(
    node,
    authoredGeom,
    kids,
  );
```

Semantic chrome (including stepper boxes) is painted from **`geom`** (parent-resolved bounds), not re-measured:

```3839:3840:apps/explainers/src/core/diagram/SceneRenderer.tsx
  } else if (hasNativeSemanticChrome(node)) {
    const semantic = resolveSemanticChrome(node, geom);
```

### Task 2 implication

The plan's preferred fix applies: **bottom-up resolve in `indexSceneNodes`** — for each child, call `resolveCapabilityLayout(child, grandchildren).bounds` (recursively) and merge resolved geometry into the child node **before** calling the parent's `resolveContainerLayout`. The same pre-pass may be needed in the render path or by extracting a shared indexer both paths call.

---

## 2. `capabilities/layout.ts` — constants, registry, leaf hooks

### Stepper / text-width constants (local, not yet shared)

```20:23:apps/explainers/src/core/diagram/capabilities/layout.ts
const STEPPER_CHIP_HEIGHT = 26;
const STEPPER_MIN_CHIP_WIDTH = 72;
const STEPPER_CHAR_WIDTH = 6.2;
const STEPPER_CHIP_PAD = 24;
```

```331:337:apps/explainers/src/core/diagram/capabilities/layout.ts
function stepperChipWidth(label: string, index: number): number {
  const text = `${index + 1}. ${label}`;
  return Math.max(
    STEPPER_MIN_CHIP_WIDTH,
    Math.ceil(text.length * STEPPER_CHAR_WIDTH) + STEPPER_CHIP_PAD,
  );
}
```

**Note:** No `SCENE_TEXT_SCALE` (0.9) applied here today. Task 1's `text-metrics.ts` is referenced by `text-metrics.test.ts` but **the module file does not exist yet** in the tree—Task 2 should import from there once Task 1 lands.

### `resolveStepperLayout` behavior

Uses `props.steps` labels when present; otherwise falls back to child authored widths or `stepperChipWidth` from accessibility labels:

```339:382:apps/explainers/src/core/diagram/capabilities/layout.ts
export function resolveStepperLayout(
  node: SceneNodeLike,
  children: readonly SceneNodeLike[],
): CapabilityLayout {
  const authored = geometryOf(node);
  const gap = styleNumber(node.style, "gap", 12);
  const labels = stringArrayProp(node, "steps");
  const widths =
    labels.length > 0
      ? labels.map(stepperChipWidth)
      : children.map((child, index) => {
          const geometry = geometryOf(child);
          return geometry.width > 0
            ? geometry.width
            : stepperChipWidth(
                child.accessibility?.label ?? `step ${index + 1}`,
                index,
              );
        });
  // ... places chips at cursorX, expands bounds.width to intrinsicWidth
```

Current unit test expects unscaled total **279** for `["layout", "slots", "timeline"]` with gap 16:

```34:43:apps/explainers/src/core/diagram/capabilities/layout.test.ts
  it("expands a semantic stepper to fit numbered labels", () => {
    const stepper = node("steps", "core.stepper", 160, 90, {
      props: { steps: ["layout", "slots", "timeline"], linked: true },
      style: { gap: 16 },
    });

    const layout = resolveCapabilityLayout(stepper, []);

    expect(layout.bounds).toEqual({ x: 0, y: 0, width: 279, height: 90 });
```

### `LAYOUT_CAPABILITIES` registry (complete list)

```449:460:apps/explainers/src/core/diagram/capabilities/layout.ts
export const LAYOUT_CAPABILITIES: readonly NativeSceneCapability[] = [
  { capabilityId: "layout.stack", resolveLayout: resolveStackLayout },
  { capabilityId: "layout.grid", resolveLayout: resolveGridLayout },
  { capabilityId: "layout.rail", resolveLayout: resolveRailLayout },
  { capabilityId: "layout.pad", resolveLayout: resolvePadLayout },
  { capabilityId: "core.lane", resolveLayout: resolveLaneLayout },
  { capabilityId: "core.band", resolveLayout: resolveIdentityLayout },
  { capabilityId: "core.swimlane", resolveLayout: resolveSwimlaneLayout },
  { capabilityId: "core.stepper", resolveLayout: resolveStepperLayout },
  { capabilityId: "core.circle", resolveLayout: resolveEllipseLayout },
  { capabilityId: "core.ellipse", resolveLayout: resolveEllipseLayout },
];
```

### Do `core.chip` / `core.panel` / `core.note` have layout hooks?

**No.** None appear in `LAYOUT_CAPABILITIES`. Unregistered capabilities hit `resolveIdentityLayout` (registry fallback at lines 55–56), which treats authored size as final unless children overflow:

```82:100:apps/explainers/src/core/diagram/capabilities/layout.ts
export function resolveIdentityLayout(
  node: SceneNodeLike,
  children: readonly SceneNodeLike[],
): CapabilityLayout {
  const bounds = geometryOf(node);
  const childGeometries = children.map(geometryOf);
  // ... max(bounds, child extent) when not clipping
```

Task 2 must add `resolveChipLayout`, `resolvePanelLayout`, and registry entries for `core.chip`, `core.panel`, `core.note` (and optionally `core.header`).

### Rail test documents today's authored-width behavior

```64:79:apps/explainers/src/core/diagram/capabilities/layout.test.ts
  it("expands a row rail to fit authored child widths and heights", () => {
    const rail = node("rail", "layout.rail", 160, 22, {
      style: { direction: "row", gap: 8 },
    });
    const children = [
      node("a", "core.chip", 84, 26),
      node("b", "core.chip", 84, 26),
      node("c", "core.chip", 84, 26),
    ];

    const layout = resolveCapabilityLayout(rail, children);

    expect(layout.bounds).toEqual({ x: 0, y: 0, width: 268, height: 26 });
    expect(layout.childGeometries.map((geometry) => geometry.x)).toEqual([
      0, 92, 184,
    ]);
```

Task 2's rail+auto-grow chip test (plan) requires pre-resolved child geometries or indexer fix first.

---

## 3. `capabilities/chrome.ts` — duplicated text-metrics targets

### Constants overlapping Task 1 `text-metrics.ts` exports

```38:44:apps/explainers/src/core/diagram/capabilities/chrome.ts
const INSET = 8;
const TITLE_HEIGHT = 22;
const DETAIL_HEIGHT = 20;
const STEPPER_HEIGHT = 26;
const STEPPER_MIN_WIDTH = 72;
const STEPPER_CHAR_WIDTH = 6.2;
const STEPPER_PAD = 24;
```

### Duplicate stepper width estimator

```71:77:apps/explainers/src/core/diagram/capabilities/chrome.ts
function stepWidth(label: string, index: number): number {
  return Math.max(
    STEPPER_MIN_WIDTH,
    Math.ceil(`${index + 1}. ${label}`.length * STEPPER_CHAR_WIDTH) +
      STEPPER_PAD,
  );
}
```

Mirrors `layout.ts` `stepperChipWidth` (same formula, different names: `STEPPER_PAD` vs `STEPPER_CHIP_PAD`, `STEPPER_HEIGHT` vs `STEPPER_CHIP_HEIGHT`).

### Where constants drive placement

**Stepper** — independent chip box layout from `props.steps`:

```108:138:apps/explainers/src/core/diagram/capabilities/chrome.ts
  if (capability === "core.stepper") {
    const steps = stringArrayProp(node, "steps");
    const gap = gapOf(node);
    let cursorX = geometry.x;
    // ...
    steps.forEach((step, index) => {
      const width = stepWidth(step, index);
      boxes.push({
        id: `${node.id}__step-${index}`,
        geometry: {
          x: cursorX,
          y: geometry.y,
          width,
          height: STEPPER_HEIGHT,
        },
```

**Panel / chip / note** — title/detail text bands use `INSET`, `TITLE_HEIGHT`, `DETAIL_HEIGHT`:

```142:187:apps/explainers/src/core/diagram/capabilities/chrome.ts
  if (title !== undefined) {
    const centered =
      capability === "core.panel" ||
      capability === "core.chip" ||
      capability === "core.note";
    texts.push({
      // ...
      x: centered ? geometry.x : geometry.x + INSET,
      width: centered ? geometry.width : Math.max(geometry.width - INSET * 2, 0),
      height: capability === "core.chip" ? geometry.height : TITLE_HEIGHT,
```

Chrome capabilities registered in `hasNativeSemanticChrome`:

```80:92:apps/explainers/src/core/diagram/capabilities/chrome.ts
export function hasNativeSemanticChrome(node: SceneNodeLike): boolean {
  return (
    node.props !== undefined &&
    [
      "core.panel",
      "core.header",
      "core.chip",
      "core.note",
      "core.lane",
      "core.band",
      "core.stepper",
    ].includes(node.capabilityId ?? node.capability ?? "")
  );
}
```

### Third duplicate (out of Task 2 file list, relevant for parity)

`apps/explainers/src/flow/compiler/desugar-scene-primitives.ts` lines 33–43 define the same `STEPPER_*` constants and `stepperChipWidth` for compiler-time desugaring. Converging this path is not in Task 2 scope but affects IR geometry before render.

---

## 4. `verify-geometry.ts` — bounds collection and insertion points

### Current bounds source: authored geometry only

```245:255:apps/explainers/src/flow/dev-tools/verify-geometry.ts
/** Resolves finite geometry from either `geometry` or the legacy `layout`. */
export function geomOf(node: unknown): Geometry | null {
  const value = record(node);
  const geometry = record(value.geometry ?? value.layout);
  const x = Number(geometry.x);
  const y = Number(geometry.y);
  const width = Number(geometry.width);
  const height = Number(geometry.height);
  if (![x, y, width, height].every(Number.isFinite)) return null;
  return { x, y, width, height };
}
```

No call to `resolveCapabilityLayout`. No recursive world-geometry accumulation. No `coordinateSpace: "local"` handling (grep across `flow/dev-tools` finds none).

### Downstream consumers (all use raw `geomOf`)

| Consumer | Location | Use |
|---|---|---|
| `resolveEndpoint` | `verify-geometry.ts:303–317` | Connector anchor snap to node box |
| `resolveFanEndpoint` | `verify-geometry.ts:336–352` | Fan topology anchors |
| `boxGeometries` | `verify-deck.ts:92–100` | Flat list of boxes for arrow snap checks |
| Per-node viewport / missing-geometry | `verify-deck.ts:444–514` | Dot centers, box validation, `inViewport` |
| Node.js mirror | `scripts/flow-verifier/ir.mjs:54–58`, `geometry.mjs:201+` | Same `geomOf` pattern |

Example from `verify-deck.ts`:

```92:100:apps/explainers/src/flow/dev-tools/verify-deck.ts
function boxGeometries(nodes: readonly RenderNodeIr[]): Geometry[] {
  const boxes: Geometry[] = [];
  for (const node of nodes) {
    if (!isBoxLike(node) || isArrowLike(node) || isDotLike(node)) continue;
    const geometry = geomOf(node);
    if (!geometry || geometry.width <= 0 || geometry.height <= 0) continue;
    boxes.push(geometry);
  }
  return boxes;
}
```

Used at slide verification entry:

```281:283:apps/explainers/src/flow/dev-tools/verify-deck.ts
    const nodes = walkNodes(roots);
    const ids = nodeIds(roots);
    const boxes = boxGeometries(nodes);
```

### Best insertion points (minimal rewrite)

**Option A — Add `indexResolvedGeometry` helper in `verify-geometry.ts` (recommended)**

Mirror `SceneRenderer.indexSceneNodes` visit order:

1. New export, e.g. `buildWorldGeometryIndex(roots: readonly RenderNodeIr[]): Map<string, Geometry>`.
2. Recursive `visit(node, originX, originY, coordsAreLocal, geometryOverride, ancestors)` — copy logic from `SceneRenderer.tsx:637–691`, including `relativePosition` handling and `childrenUseLocalLayout` rules (or import shared helper if extracted later).
3. **Task 2 critical:** Before parent `resolveCapabilityLayout`, map each child to `{ ...child, geometry: resolveCapabilityLayout(child, grandchildren).bounds }` (recursive).
4. Replace `geomOf(target)` in `resolveEndpoint` / `resolveFanEndpoint` with `worldIndex.get(nodeId) ?? geomOf(target)` fallback.

**Option B — Thin wrapper without full world index**

Add `resolvedBounds(node, children)` that only runs `resolveCapabilityLayout` in local space—insufficient alone because verifier checks use **flat authored coordinates** and connector snapping needs **world** bounds for nested local children.

**Option C — Extract shared indexer from SceneRenderer**

Move `indexSceneNodes` + types to e.g. `capabilities/scene-index.ts`; import from SceneRenderer and verify-geometry. Larger diff but eliminates drift.

### `SceneNodeLike` vs `RenderNodeIr` mapping

`SceneNodeLike` (renderer):

```90:113:apps/explainers/src/core/diagram/SceneRenderer.tsx
export type SceneNodeLike = Readonly<{
  id: string;
  kind?: string;
  capabilityId?: string;
  capability?: string;
  geometry?: SceneGeometryLike;
  layout?: SceneGeometryLike;
  relativePosition?: Readonly<{ nodeId: string; anchor?: string; dx?: number; dy?: number; }>;
  style?: Readonly<Record<string, SceneStyleValue>>;
  props?: Readonly<Record<string, unknown>>;
  text?: string;
  accessibility?: SceneNodeAccessibilityLike;
  children?: readonly SceneNodeLike[];
  // ... connector fields
}>;
```

`RenderNodeIr` (IR):

```147:173:apps/explainers/src/flow/schema/ir.ts
export type RenderNodeBaseIr = Readonly<{
  id: string;
  capabilityId?: string | undefined;
  capability?: FoundationCapabilityId | (string & {}) | undefined;
  geometry: GeometryIr;
  relativePosition?: RelativePositionIr | undefined;
  style: Readonly<Record<string, StyleValueIr>>;
  accessibility: NodeAccessibilityIr;
  props?: Readonly<Record<string, JsonValue>> | undefined;
  // ...
}>;
```

**Mapping assessment:** Structural superset/subset compatible for layout. `RenderNodeIr` always has `geometry`; `SceneNodeLike` fields are optional. A cast `node as SceneNodeLike` (or a small adapter copying `{ id, capabilityId, capability, geometry, style, props, accessibility, children, relativePosition }`) is sufficient—**no schema changes required**. Connector-only IR fields are ignored by layout resolvers.

### Node.js verifier parity

`scripts/flow-verifier/geometry.mjs` duplicates `geomOf` and is consumed by `ir.mjs`. Plan notes leaving `.mjs` for follow-up unless shared pure logic can be imported without a large rewrite. Task 3 can document browser/TS `verify-geometry.ts` as the parity path first.

---

## 5. `SceneRenderer.sdk-primitives.test.tsx` — hard-coded stepper widths

### Stepper test (Task 2 must update)

```189:224:apps/explainers/src/core/diagram/SceneRenderer.sdk-primitives.test.tsx
  it("renders intrinsically sized semantic stepper labels", () => {
    const { container } = render(
      <SceneRenderer
        scene={{
          roots: [
            {
              id: "steps",
              kind: "group",
              capabilityId: "core.stepper",
              geometry: { x: 10, y: 20, width: 160, height: 90 },
              style: { gap: 16 },
              props: {
                steps: ["layout", "slots", "timeline"],
                linked: true,
              },
              // ...
            },
          ],
          // ...
        }}
        // ...
      />,
    );

    expect(
      [...container.querySelectorAll('[data-flow-semantic-text="core.stepper"]')].map(
        (node) => node.textContent,
      ),
    ).toEqual(["1. layout", "2. slots", "3. timeline"]);
    expect(
      container
        .querySelector('[data-flow-semantic-chrome="core.stepper"]')
        ?.getAttribute("width"),
    ).toBe("80");
  });
```

**What `"80"` means:** It is the width of the **first step chip `<rect>`** (`querySelector` returns the first match), not the total stepper bounds. `80` = `stepWidth("layout", 0)` = `max(72, ceil(9 × 6.2) + 24)` from `chrome.ts:71–77`.

**After scale-aware metrics:** Replace `"80"` with `String(stepperChipWidth("layout", 0))` from `text-metrics.ts` (expected ≈ `75` with scale 0.9). Label text assertions can stay unchanged.

### Other width literals in the same file

Other tests use authored geometry widths (`160`, `200`, `100`, etc.) as fixture setup—not intrinsic sizing assertions. Only the stepper chip width assertion is Task 2-sensitive.

### Companion test file

`layout.test.ts` line 42: total stepper bounds width **`279`** — must be replaced with scale-aware sum per plan Task 2 Step 1.

---

## Task 2 checklist (from scout)

| Item | Status today | Action |
|---|---|---|
| `resolveChipLayout` / `resolvePanelLayout` | Missing | Implement + register |
| `core.chip` / `core.panel` / `core.note` in registry | Missing | Add to `LAYOUT_CAPABILITIES` |
| Shared `text-metrics.ts` | Test exists; module missing | Complete Task 1 first |
| Dedupe `layout.ts` / `chrome.ts` constants | Triplicated | Import from `text-metrics.ts` |
| Child pre-resolve before parent layout | **Not implemented** | Fix `indexSceneNodes` (+ render path) |
| Rail reflow with auto-grown chips | Uses authored 84px | Depends on indexer fix |
| Update stepper tests (`279`, `"80"`) | Hard-coded unscaled | Use `stepperChipWidth` |

## Task 3 checklist (from scout)

| Item | Status today | Action |
|---|---|---|
| `verify-geometry.ts` layout parity | Raw `geomOf` only | Add world-geometry index with `resolveCapabilityLayout` |
| `verify-deck.ts` box snap / viewport | Uses flat authored boxes | Switch to resolved world index |
| `RenderNodeIr` → `SceneNodeLike` | Compatible | Cast or thin adapter |
| `flow-verifier/*.mjs` | Duplicated `geomOf` | Document deferral or follow-up |
| Local coordinate nesting | Not handled in verifier | Must mirror `indexSceneNodes` local translate |

---

## Related files (not edited in scout)

- Plan: `docs/superpowers/plans/2026-07-20-diagram-node-auto-sizing.md`
- Spec: `docs/superpowers/specs/2026-07-20-diagram-node-auto-sizing-design.md`
- Compiler desugar stepper metrics: `apps/explainers/src/flow/compiler/desugar-scene-primitives.ts:33–43`
