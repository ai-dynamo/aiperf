# Live Browser Flow Compilation for Explainers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `apps/explainers` compile its `.flow` decks in the browser from a locally owned compiler copy, while deleting generated packages and legacy/pre-build paths.

**Architecture:** Copy the exact browser-safe `compileExplainerSource` dependency closure into `apps/explainers/src/flow`, rewrite workspace imports to local relative imports, and expose one local browser entrypoint. An eager Vite raw-source loader compiles every deck synchronously into an in-memory `DeckPackage`, then uses the existing adapter and registry.

**Tech Stack:** TypeScript, Vite raw imports, Chevrotain, Zod, React 19.

## Global Constraints

- Compilation runs in the browser in development and production.
- All first-party compiler, language, and schema code needed at runtime lives under `apps/explainers`.
- The explainers app has no runtime import, alias, or workspace dependency on `apps/aiperf-flow`.
- Delete generated DeckPackage JSON, the package pre-build, and incomplete regex loaders.
- Preserve all existing deck ids and routes.
- Do not add tests or run test commands.
- Do not create commits unless the user explicitly requests one.

---

### Task 1: Vendor the browser-safe Flow compiler

**Files:**
- Create: `apps/explainers/src/flow/compiler/*.ts`
- Create: `apps/explainers/src/flow/language/*.ts`
- Create: `apps/explainers/src/flow/language/grammar/explainer.ts`
- Create: `apps/explainers/src/flow/schema/*.ts`
- Create: `apps/explainers/src/flow/index.ts`
- Modify: `apps/explainers/package.json`
- Modify: `apps/explainers/package-lock.json`

**Interfaces:**
- Produces: `compileExplainerSource(request): Result<DeckPackage>`
- Produces: `FOUNDATION_CAPABILITIES`, `hasErrors`, `DeckPackage`, `Diagnostic`
- Consumes: third-party `chevrotain` and `zod`

- [ ] **Step 1: Copy only the compile closure**

Copy these compiler files:

```text
compile-explainer.ts
desugar-scene-primitives.ts
link.ts
lower-explainer-scene.ts
lower-explainer.ts
lower.ts
validate-explainer-timelines.ts
```

Copy these language files:

```text
ast.ts
embedded-scene.ts
grammar/explainer.ts
parser.ts
tokens.ts
```

Copy these schema files:

```text
capability.ts
capability-id.ts
deck-package.ts
diagnostic.ts
ir.ts
json-value.ts
layout-plan.ts
semantic-model.ts
source.ts
theme.ts
```

- [ ] **Step 2: Replace workspace package imports**

In copied compiler files:

```ts
"@aiperf/flow-language" → "../language/index.js"
"@aiperf/flow-schema" → "../schema/index.js"
```

In copied language files:

```ts
"@aiperf/flow-schema" → "../schema/index.js"
```

Keep existing same-directory relative imports unchanged.

- [ ] **Step 3: Add narrow local barrels**

`flow/language/index.ts` exports AST types/constants, embedded-scene APIs, and `parseDocument` / `parseNativeEmbeddedScene`.

`flow/schema/index.ts` exports the copied schema modules only.

`flow/index.ts` contains:

```ts
export {
  compileExplainerSource,
  type CompileExplainerRequest,
} from "./compiler/compile-explainer.js";
export {
  FOUNDATION_CAPABILITIES,
  hasErrors,
  type DeckPackage,
  type Diagnostic,
} from "./schema/index.js";
```

- [ ] **Step 4: Declare browser dependencies**

Run from `apps/explainers`:

```bash
npm install chevrotain zod
```

Expected: `package.json` and `package-lock.json` directly declare/resolve both packages; no `@aiperf/*` dependency is added.

### Task 2: Compile raw Flow sources into the synchronous registry

**Files:**
- Create: `apps/explainers/src/core/load-deck-flows.ts`
- Modify: `apps/explainers/src/core/deck-registry.ts`
- Delete: `apps/explainers/src/core/load-deck-packages.ts`

**Interfaces:**
- Produces: `loadDeckFlows(): DeckDefinition[]`
- Produces: `loadDeckFlowById(id): DeckDefinition | undefined`
- Consumes: local `compileExplainerSource`, raw `.flow` strings, and `packageToDeckDefinition`

- [ ] **Step 1: Add the eager raw-source loader**

Use:

```ts
const flowSources = import.meta.glob("../../decks-flow/*.flow", {
  query: "?raw",
  import: "default",
  eager: true,
}) as Record<string, string>;
```

For each sorted source path, compile with:

```ts
compileExplainerSource({
  source,
  sourceName: path,
  capabilities: FOUNDATION_CAPABILITIES,
  strict: true,
});
```

Reject error diagnostics, filename/id mismatches, and duplicate ids. Format
diagnostics as `source:line:column: severity code: message (repair)`.

- [ ] **Step 2: Switch the registry to live source**

Replace generated-package imports and messages with `loadDeckFlowById`.
Retain `EXPECTED_DECK_ROUTES`, route checks, and synchronous exports.
Rename `registryUsesGeneratedPackages` to `registryUsesLiveFlowSources`.

- [ ] **Step 3: Remove the generated package loader**

Delete `load-deck-packages.ts`; remove every import of it.

### Task 3: Delete generated artifacts and pre-build wiring

**Files:**
- Delete: `apps/explainers/src/decks-generated/`
- Delete: `apps/aiperf-flow/scripts/build-explainer-packages.mjs`
- Delete: `apps/explainers/scripts/assert-deck-packages.mjs`
- Delete: `apps/aiperf-flow/packages/runtime/src/explainer/flow-loader.ts`
- Modify: `apps/explainers/package.json`
- Modify: `apps/aiperf-flow/package.json`
- Modify: `Makefile`
- Modify: `apps/explainers/scripts/assert-no-mentalmodel-registry.mjs`

**Interfaces:**
- Removes: all generate/assert package commands
- Preserves: no-MentalModel static assertion

- [ ] **Step 1: Delete generated and legacy files**

Delete all listed files and the generated directory, including `.gitkeep`.

- [ ] **Step 2: Remove obsolete scripts and targets**

Remove `assert:deck-packages`, `build:explainer-packages`,
`assert:deck-packages`, `build-explainer-packages`, and
`assert-deck-packages`. Retarget `assert-explainer-packages` to applicable
non-generation checks only.

- [ ] **Step 3: Correct static assertion wording**

Update `assert-no-mentalmodel-registry.mjs` to describe:

```text
one .flow per deck → browser compile → packageToDeckDefinition → SceneRenderer
```

### Task 4: Remove remaining generated-package consumers

**Files:**
- Modify: `apps/aiperf-flow/preview/deck-packages.ts`
- Modify: `apps/aiperf-flow/vite.config.ts`
- Modify: `apps/aiperf-flow/e2e/explainer-visual-parity.spec.ts`
- Delete: `apps/aiperf-flow/packages/runtime/test/explainer/dynosim-deck.test.ts`
- Modify: `apps/explainers/scripts/flow-verifier.mjs`

**Interfaces:**
- Removes: all imports/reads of `apps/explainers/src/decks-generated`
- Preserves: browser preview access to explainer decks where still required

- [ ] **Step 1: Convert preview package loading**

Make preview loading consume raw `.flow` sources and the locally owned
`apps/explainers/src/flow` compiler. Do not reintroduce imports from explainers
to aiperf-flow.

- [ ] **Step 2: Remove stale Vite filesystem access**

Delete `decks-generated` from `server.fs.allow`. Add only the explainers source
paths needed by the aiperf-flow preview if its preview remains.

- [ ] **Step 3: Remove artifact-specific test code without replacement**

Delete the runtime test whose sole fixture is generated JSON. Remove generated
package assertions from the e2e source without adding or running tests.

- [ ] **Step 4: Retarget or simplify the verifier**

Remove `PACKAGES_DIR`, JSON reads, `--from-flow`, package rebuild behavior, and
the obsolete IR-package mode. Retain the verifier's play/static verification.

### Task 5: Update documentation and perform non-test verification

**Files:**
- Modify: `apps/explainers/README.md`
- Modify: `docs/superpowers/specs/2026-07-18-flow-backed-explainers-design.md`
- Modify: other current specs that prescribe rebuilding `decks-generated`

**Interfaces:**
- Documents: live browser compile and local compiler ownership

- [ ] **Step 1: Replace current-truth package-generation claims**

Document that `.flow` files are raw Vite inputs compiled synchronously in the
browser and that compiler source is owned under `apps/explainers/src/flow`.
Remove commands that generate or assert committed package JSON.

- [ ] **Step 2: Run allowed checks only**

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
npm --prefix apps/explainers run build
/usr/bin/python3 tools/check_docs_current.py
```

Expected: both commands exit zero. Do not run Vitest, Playwright, Cargo test, or
any command whose purpose is executing tests.

- [ ] **Step 3: Scan for forbidden stale references**

Search for:

```text
decks-generated
build-explainer-packages
assert-deck-packages
flow-loader
@aiperf/flow-
```

Expected: no runtime/pre-build references remain in `apps/explainers`; any
historical plan references are clearly historical rather than current truth.
