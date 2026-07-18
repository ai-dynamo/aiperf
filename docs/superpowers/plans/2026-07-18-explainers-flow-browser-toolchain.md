# Explainers Browser Flow Toolchain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Vendor and wire the useful browser-safe Flow compiler, formatter, evaluator, and verifier into `apps/explainers`, with heavy developer diagnostics loaded lazily.

**Architecture:** The eager production path owns parsing, compilation, strict validation, set validation, and DeckPackage adaptation under `apps/explainers/src/flow`. Pure runtime evaluation and IR/geometry verification live in separate local modules dynamically imported only when `import.meta.env.DEV` is true.

**Tech Stack:** TypeScript, Vite, Chevrotain, Zod, js-sha256, React 19.

## Global Constraints

- All first-party browser toolchain source lives under `apps/explainers/src/flow`.
- `apps/explainers` must not import, alias, or depend on first-party modules from `apps/aiperf-flow`.
- Existing deck ids, routes, narration, scenes, and production SceneRenderer behavior remain stable.
- Exclude Node filesystem/CLI/process modules, Playwright, alternate React/Canvas apps, and narrative/WASM subsystems.
- Heavy runtime evaluation and verification must load only through a development-only dynamic import.
- Do not add, modify, delete, or run tests.
- Do not create commits unless the user explicitly requests one after implementation.
- Preserve unrelated working-tree changes.

---

### Task 1: Complete the local schema and language surface

**Files:**
- Create: `apps/explainers/src/flow/schema/component-descriptor.ts`
- Create: `apps/explainers/src/flow/language/formatter.ts`
- Modify: `apps/explainers/src/flow/schema/index.ts`
- Modify: `apps/explainers/src/flow/language/index.ts`
- Sync: `apps/explainers/src/flow/language/ast.ts`
- Sync: `apps/explainers/src/flow/language/embedded-scene.ts`
- Sync: `apps/explainers/src/flow/language/parser.ts`

**Interfaces:**
- Produces: `safeParseComponentDescriptor`, `createComponentCatalog`, `ComponentCatalog`
- Produces: `formatDocument(document: DocumentAst): string`
- Produces: browser-safe language and schema barrels for later compiler tasks

- [ ] **Step 1: Copy current working-tree upstream files**

Copy from:

```text
apps/aiperf-flow/packages/schema/src/component-descriptor.ts
apps/aiperf-flow/packages/language/src/formatter.ts
apps/aiperf-flow/packages/language/src/ast.ts
apps/aiperf-flow/packages/language/src/embedded-scene.ts
apps/aiperf-flow/packages/language/src/parser.ts
```

Use current working-tree contents. Preserve SPDX headers.

- [ ] **Step 2: Rewrite first-party imports locally**

Within copied language files, replace:

```text
@aiperf/flow-schema → ../schema/index.js
```

Retain existing relative imports. Do not introduce aliases.

- [ ] **Step 3: Complete narrow barrels**

`schema/index.ts` exports `component-descriptor.ts` in addition to existing
schema modules. `language/index.ts` exports `formatDocument` plus parser,
embedded-scene, explainer grammar, and AST APIs. Do not export test helpers.

- [ ] **Step 4: Perform static boundary checks**

Search `apps/explainers/src/flow/{language,schema}` for `@aiperf/` and `node:`.
Expected: no matches.

### Task 2: Complete the browser compiler

**Files:**
- Create: `apps/explainers/src/flow/compiler/components.ts`
- Create: `apps/explainers/src/flow/compiler/expand-symbols.ts`
- Create: `apps/explainers/src/flow/compiler/module-resolution.ts`
- Create: `apps/explainers/src/flow/compiler/pack.ts`
- Create: `apps/explainers/src/flow/compiler/symbols.ts`
- Create: `apps/explainers/src/flow/compiler/themes.ts`
- Create: `apps/explainers/src/flow/compiler/validate.ts`
- Create: `apps/explainers/src/flow/compiler/validate-explainer-set.ts`
- Create: `apps/explainers/src/flow/compiler/lower-explainer-slides.ts`
- Create: `apps/explainers/src/flow/compiler/compile-source.ts`
- Create: `apps/explainers/src/flow/compiler/serialization.ts`
- Create: `apps/explainers/src/flow/compiler/browser.ts`
- Sync: `apps/explainers/src/flow/compiler/desugar-scene-primitives.ts`
- Sync: `apps/explainers/src/flow/compiler/compile-explainer.ts`
- Sync: `apps/explainers/src/flow/compiler/desugar-scene-primitives.ts`
- Sync: `apps/explainers/src/flow/compiler/link.ts`
- Sync: `apps/explainers/src/flow/compiler/lower-explainer-scene.ts`
- Sync: `apps/explainers/src/flow/compiler/lower-explainer.ts`
- Sync: `apps/explainers/src/flow/compiler/lower.ts`
- Sync: `apps/explainers/src/flow/compiler/validate-explainer-timelines.ts`
- Modify: `apps/explainers/src/flow/index.ts`
- Modify: `apps/explainers/package.json`
- Modify: `apps/explainers/package-lock.json`

**Interfaces:**
- Produces: `compileSource(request): Result<FlowIr>`
- Produces: `compileExplainerSource(request): Result<DeckPackage>`
- Produces: `validateExplainerSet(packages): Result<readonly DeckPackage[]>`
- Produces: browser-safe module resolution and deterministic serialization

- [ ] **Step 1: Copy browser-safe compiler modules**

Copy the upstream files named in the Files section. Rewrite:

```text
@aiperf/flow-language → ../language/index.js
@aiperf/flow-schema → ../schema/index.js
```

Copy current working-tree versions, including strict fan endpoint handling.
Do not copy upstream `index.ts` or `pack-deck-package.ts`.

- [ ] **Step 2: Add the general compile entry**

Create `compile-source.ts` from the pure `compileSource` pipeline in upstream
`compiler/src/index.ts`:

```ts
parseDocument
  → collectSymbols
  → expandSymbolInvocations
  → link
  → validate(capabilities, strict)
  → lower
  → safeParseFlowIr
```

Export `CompileRequest` and `FLOW_COMPILER_VERSION`.

- [ ] **Step 3: Split browser serialization**

Create `serialization.ts` with pure functions only:

```ts
export function packDeckPackageToJson(pkg: DeckPackage): string {
  const payload: DeckPackage = { ...pkg, schemaVersion: 1 };
  return `${new TextDecoder().decode(canonicalJson(payload))}\n`;
}
```

Import `canonicalJson` from local `pack.ts`. Do not copy `writeDeckPackage` or
any filesystem import.

- [ ] **Step 4: Add the browser compiler barrel**

`compiler/browser.ts` exports compilation, module-resolution, validation,
format-independent packing, symbols/components/themes APIs, and related types.
It must not export Node writers.

`flow/index.ts` re-exports the browser compiler barrel, language formatter, and
shared schema APIs. It must not eagerly export runtime or dev-tools modules.

- [ ] **Step 5: Add hashing dependency**

Run:

```bash
npm install --prefix apps/explainers js-sha256
```

Expected: only `apps/explainers/package.json` and lockfile dependency metadata
change.

- [ ] **Step 6: Perform static boundary checks**

Search all of `apps/explainers/src/flow/compiler` for `node:` and
`@aiperf/`. Expected: no matches.

### Task 3: Honor strict validation and validate the deck set

**Files:**
- Modify: `apps/explainers/src/flow/compiler/compile-explainer.ts`
- Modify: `apps/explainers/src/flow/compiler/lower-explainer.ts`
- Modify: `apps/explainers/src/flow/compiler/lower-explainer-scene.ts`
- Create: `apps/explainers/src/flow/dev-tools/diagnostics.ts`
- Modify: `apps/explainers/src/core/load-deck-flows.ts`
- Modify: `apps/explainers/src/core/deck-registry.ts`

**Interfaces:**
- `LowerExplainerSceneOptions` gains `capabilities` and `strict`
- `lowerExplainerToDeckPackage(ast, options)` forwards validation options
- `loadDeckPackages(): readonly DeckPackage[]` compiles once and set-validates
- `loadDeckFlows(): DeckDefinition[]` adapts the cached package set
- `formatDiagnostic(diagnostic): string` is shared

- [ ] **Step 1: Thread compile options into scene lowering**

Add:

```ts
type ExplainerCompileOptions = Readonly<{
  capabilities: CapabilityRegistryManifest;
  strict: boolean;
}>;
```

Pass options from `compileExplainerSource` through
`lowerExplainerToDeckPackage` to each `lowerExplainerScene` call, including the
final card.

- [ ] **Step 2: Validate native embedded scenes**

For native `SceneAst` values:

1. Wrap the scene in a `DocumentAst`.
2. Collect and expand symbols.
3. Link the document.
4. Run `validate(linked, capabilities, strict)`.
5. Stop on any error diagnostic.
6. Lower and schema-validate the resulting scene.

Preserve package-scene normalization for `roots`/`timeline` authoring. For
package-scene nodes, reject capabilities absent from the supplied manifest and
preserve source/slide context in diagnostics.

- [ ] **Step 3: Add one diagnostic formatter**

Create:

```ts
export function formatDiagnostic(diagnostic: Diagnostic): string {
  const { source, start } = diagnostic.range;
  const repair =
    diagnostic.repair === undefined ? "" : ` (${diagnostic.repair})`;
  return `${source}:${start.line}:${start.column}: ${diagnostic.severity} ${diagnostic.code}: ${diagnostic.message}${repair}`;
}
```

- [ ] **Step 4: Compile and validate packages once**

Refactor `load-deck-flows.ts` to compile all raw sources into
`DeckPackage[]`, validate filename/id agreement, call `validateExplainerSet`,
then adapt. Cache the successful package array at module scope so registry
lookups do not recompile every deck.

Export the compiled packages read-only for development diagnostics.

- [ ] **Step 5: Preserve registry contracts**

Keep `EXPECTED_DECK_ROUTES`, `DECK_REGISTRY`, and fail-closed route checks.
Ensure `registryUsesLiveFlowSources()` reads the cached source/package set.

### Task 4: Vendor the pure runtime evaluator

**Files:**
- Create under `apps/explainers/src/flow/runtime/`:
  - `display-list.ts`
  - `quality-profiles.ts`
  - `index.ts`
  - `evaluate/types.ts`
  - `evaluate/registry.ts`
  - `evaluate/scene-evaluator.ts`
  - `evaluate/merge-contributions.ts`
  - `evaluate/damage-tracker.ts`
  - `evaluate/hit-region-index.ts`
  - `evaluate/quality-policy.ts`
  - `evaluate/frame.ts`
  - `evaluate/timeline-state.ts`
  - `evaluate/timeline-types.ts`
  - `evaluate/with-theme.ts`
  - `evaluate/contributions/*.ts`
  - `leaves/glyph-measure.ts`
  - `leaves/queue-policy.ts`
  - `leaves/segment-strip-layout.ts`
  - `leaves/span-interval.ts`
  - `leaves/waterfall-nest-layout.ts`

**Interfaces:**
- Produces: `evaluateScene`, `evaluateFrame`
- Produces: `evaluateTimelineState`, `applyTimelineState`
- Produces: display-list and evaluation types for development diagnostics

- [ ] **Step 1: Copy the pure upstream closure**

Copy the runtime files identified in the design inventory. Rewrite
`@aiperf/flow-schema` imports to `../../schema/index.js` or the correct local
relative path.

- [ ] **Step 2: Remove the player runtime dependency**

Extract only `TimelineSnapshot` and `TimelineTargetState` type declarations from
upstream `player.ts` into `evaluate/timeline-types.ts`. Point
`timeline-state.ts` to that file. Do not copy `PerformanceClock`,
`requestAnimationFrame`, or the player implementation.

- [ ] **Step 3: Isolate pure quality profiles**

Copy pure quality constants/types from `backends/canvas/quality.ts` into
`runtime/quality-profiles.ts`. Rewrite `quality-policy.ts` to import that local
module. Do not create a canvas backend directory.

- [ ] **Step 4: Add the narrow runtime barrel**

Export evaluator functions and data types only. Do not export production
renderers, React components, or dev-tools.

- [ ] **Step 5: Check browser purity**

Search `apps/explainers/src/flow/runtime` for:

```text
node:
react
requestAnimationFrame
HTMLCanvas
onnx
kokoro
```

Expected: no runtime imports/references to excluded systems.

### Task 5: Port IR and geometry verification

**Files:**
- Create: `apps/explainers/src/flow/dev-tools/verify-geometry.ts`
- Create: `apps/explainers/src/flow/dev-tools/verify-deck.ts`
- Create: `apps/explainers/src/flow/dev-tools/index.ts`

**Interfaces:**
- Produces: `verifyPackageIr(pkg, options?): readonly VerificationFinding[]`
- Produces: `runDevDiagnostics(packages): Promise<void>`

- [ ] **Step 1: Port pure geometry helpers**

Port constants, node classification, traversal, endpoint/path resolution,
timeline progress, viewport, and geometry predicates from
`scripts/flow-verifier/geometry.mjs` to typed TypeScript. Depend only on local
schema/package types and Math/string primitives.

- [ ] **Step 2: Port DeckPackage IR checks**

Port `verifyPackageIr` and all existing finding codes from
`scripts/flow-verifier/ir.mjs`. Consume in-memory `DeckPackage`; do not read
JSON or use filesystem APIs.

- [ ] **Step 3: Add evaluator-backed developer diagnostics**

`runDevDiagnostics`:

1. Runs `verifyPackageIr` for each package.
2. Evaluates representative scene/timeline states through the local runtime.
3. Catches evaluator failures per deck/slide.
4. Groups findings with `console.groupCollapsed`.
5. Emits stable code, deck, slide, and message.
6. Returns without mutating packages.

### Task 6: Wire lazy development diagnostics

**Files:**
- Modify: `apps/explainers/src/main.tsx`
- Modify: `apps/explainers/src/core/load-deck-flows.ts`

**Interfaces:**
- Consumes: cached compiled packages
- Dynamically imports: `./flow/dev-tools/index.js`

- [ ] **Step 1: Export cached packages**

Expose a read-only accessor from `load-deck-flows.ts` that returns the already
compiled and set-validated packages without recompiling.

- [ ] **Step 2: Add development-only dynamic import**

After application mount:

```ts
if (import.meta.env.DEV) {
  void import("./flow/dev-tools/index.js")
    .then(({ runDevDiagnostics }) =>
      runDevDiagnostics(compiledDeckPackages()),
    )
    .catch((error: unknown) => {
      console.warn("Flow developer diagnostics failed to load", error);
    });
}
```

Do not import runtime/dev-tools from an eager barrel.

- [ ] **Step 3: Confirm production isolation**

Inspect the production build output to confirm developer diagnostic code does
not execute in production. A separate lazy chunk is acceptable; eager inclusion
in the main application chunk is not.

### Task 7: Document and verify without tests

**Files:**
- Modify: `apps/explainers/README.md`
- Modify: current Flow/explainer design documentation that describes the
  browser compiler subset

**Interfaces:**
- Documents local ownership, eager compiler, lazy dev tools, and drift policy

- [ ] **Step 1: Update current-truth documentation**

Document the locally owned compiler/language/schema/runtime/dev-tools
boundaries and excluded Node/UI subsystems.

- [ ] **Step 2: Run TypeScript and production build checks**

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
npm --prefix apps/explainers run build
```

Expected: TypeScript no-emit and Vite production build exit zero.

- [ ] **Step 3: Run static import scans**

Search under `apps/explainers/src` for first-party imports or aliases pointing
to `apps/aiperf-flow` or `@aiperf/flow-*`. Expected: no matches.

Search `apps/explainers/src/flow` for `node:` imports. Expected: no matches.

- [ ] **Step 4: Run documentation consistency**

Run:

```bash
/usr/bin/python3 tools/check_docs_current.py
```

Expected: exit zero.

Do not run Vitest, Playwright, Cargo test, or any other test command.
