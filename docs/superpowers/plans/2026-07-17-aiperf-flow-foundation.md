# AIPerf Flow Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a working end-to-end vertical slice that compiles an explicit
`.flow` source into versioned normal and packed Flow IR and a self-contained
static site whose foundation runtime interprets either representation. The
foundation SVG path proves the pipeline and becomes the simplified fallback; it
does not define the final cinematic renderer.

**Architecture:** Create a new npm workspace app at `apps/aiperf-flow/` with focused packages
beneath `packages/`. Leave the legacy `apps/explainers/` SPA unchanged. The
language package parses source into a source-mapped AST; the schema package owns
diagnostics, capability descriptors, and Flow IR; the compiler links,
validates, lowers, and packs deterministic chunks; the runtime binds validated
normal or packed IR to registered capabilities and renders the initial semantic
SVG/HTML fallback; and the CLI exposes format, check, build, inspect, and
capabilities commands.

**Tech Stack:** TypeScript strict mode, npm workspaces, Chevrotain, Zod, React
19, SVG/HTML, esbuild, Commander, Vitest, Testing Library, jsdom.

## North-star boundary

The final product is a live, interactive, narrated generic scene compiler and
animator with the visual fidelity of a professionally produced high-resolution
explainer. The production visual architecture is a deterministic,
backend-neutral scene evaluator feeding Canvas 2D, with React/HTML for viewer
chrome and an always-mounted semantic accessibility twin. SVG/HTML is the
required simplified fallback. A future WebGPU backend may accelerate the same
evaluated scene but cannot own semantics, timing, or interaction.

This foundation plan intentionally lands only the narrow end-to-end pipeline,
semantic SVG fallback, basic deterministic clock, and inspect behavior. No
subsequent plan may treat foundation React/SVG nodes, DOM cardinality, or
fixture-owned chrome as the long-term visual architecture.

## Global Constraints

- Activate the repository environment before every command:
  `source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate`.
- Leave the legacy `apps/explainers/` SPA unchanged; Flow is a separate app.
- Use current stable npm packages installed through npm; do not hand-edit
  dependency versions.
- The authored format is `.flow`; do not generate TypeScript scene files.
- The browser interprets only validated normal or packed Flow IR.
- Normal and packed IR have identical observable semantics.
- Every IR object has a version, stable ID where addressable, and source map.
- Diagnostics have stable code, severity, message, source range, and optional
  repair text.
- The foundation vocabulary is intentionally narrow, but every implemented
  property is explicit and every package boundary is final.
- Runtime capabilities are closed, typed registry entries; source cannot execute
  arbitrary JavaScript, JSX, or CSS.
- The static output works from a relative path and without a server API.
- Every visual primitive has accessibility data and a fallback.
- Foundation render nodes must preserve enough semantic identity to project
  later into Canvas draw commands, synchronized semantic HTML, and SVG fallback
  without changing Flow IR meaning.
- Timeline state derives from integer virtual time; direct seek and continuous
  playback must remain architecturally equivalent.
- Output ordering, JSON serialization, and content hashes are deterministic.
- Do not create commits unless the user explicitly requests them.

## File structure

```text
apps/aiperf-flow/
├── package.json                       # Flow workspace root
├── tsconfig.base.json                 # shared strict compiler options
├── packages/
│   ├── schema/
│   │   ├── package.json
│   │   ├── tsconfig.json
│   │   ├── src/{diagnostic,source,capability,ir,index}.ts
│   │   └── test/{capability,ir}.test.ts
│   ├── language/
│   │   ├── package.json
│   │   ├── tsconfig.json
│   │   ├── src/{tokens,ast,parser,formatter,index}.ts
│   │   └── test/{parser,formatter}.test.ts
│   ├── compiler/
│   │   ├── package.json
│   │   ├── tsconfig.json
│   │   ├── src/{link,lower,validate,pack,index}.ts
│   │   └── test/{compile,pack}.test.ts
│   ├── runtime/
│   │   ├── package.json
│   │   ├── tsconfig.json
│   │   ├── src/{registry,store,renderer,player,app,site,index}.tsx
│   │   ├── src/theme.css
│   │   ├── scripts/build-site.mjs
│   │   └── test/{registry,renderer,player}.test.tsx
│   └── cli/
│       ├── package.json
│       ├── tsconfig.json
│       ├── src/{commands,site,main,index}.ts
│       └── test/{commands,site}.test.ts
└── examples/
    └── foundation/
        ├── request-flow.flow
        └── request-flow.expected.json
```

The dependency direction is:

```text
schema ← language ← compiler ← cli
   ↑                    ↑
   └──── runtime ───────┘
```

`runtime` never imports `language` or `compiler`.

---

### Task 1: Establish the workspace and package contracts

**Files:**
- Create: `apps/aiperf-flow/package.json`
- Create: `apps/aiperf-flow/tsconfig.base.json`
- Create: `apps/aiperf-flow/tsconfig.json`
- Create: `apps/aiperf-flow/packages/{schema,language,compiler,runtime,cli}/package.json`
- Create: `apps/aiperf-flow/packages/{schema,language,compiler,runtime,cli}/tsconfig.json`

**Interfaces:**
- Produces workspace packages `@aiperf/flow-schema`,
  `@aiperf/flow-language`, `@aiperf/flow-compiler`,
  `@aiperf/flow-runtime`, and `@aiperf/flow-cli`.
- Produces root scripts `flow:test`, `flow:build`, and `flow:check`.

- [ ] **Step 1: Add a workspace smoke test command that fails**

Create the Flow workspace root package with these scripts and workspace
declarations:

```json
{
  "workspaces": ["packages/*"],
  "scripts": {
    "flow:test": "npm test --workspaces --if-present",
    "flow:build": "npm run build --workspace @aiperf/flow-schema && npm run build --workspace @aiperf/flow-language && npm run build --workspace @aiperf/flow-compiler && npm run build --workspace @aiperf/flow-runtime && npm run build --workspace @aiperf/flow-cli",
    "flow:check": "npm run flow:test && npm run flow:build"
  }
}
```

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd apps/aiperf-flow
npm run flow:build
```

Expected: FAIL because the workspace packages do not exist.

- [ ] **Step 2: Add shared strict TypeScript configuration**

Create `tsconfig.base.json`:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "exactOptionalPropertyTypes": true,
    "noFallthroughCasesInSwitch": true,
    "noImplicitOverride": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "skipLibCheck": true
  }
}
```

Create the root `tsconfig.json` extending `./tsconfig.base.json` for package
orchestration. Runtime and site packages that need JSX set `jsx: "react-jsx"`
locally.

- [ ] **Step 3: Create package manifests**

Use this manifest shape for `schema`, changing only package names in the other
packages:

```json
{
  "name": "@aiperf/flow-schema",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "main": "./dist/index.js",
  "types": "./dist/index.d.ts",
  "exports": {
    ".": {
      "types": "./dist/index.d.ts",
      "import": "./dist/index.js"
    }
  },
  "scripts": {
    "build": "tsc -p tsconfig.json",
    "test": "vitest run --passWithNoTests"
  }
}
```

Use this package TypeScript configuration:

```json
{
  "extends": "../../tsconfig.base.json",
  "compilerOptions": {
    "rootDir": "src",
    "outDir": "dist"
  },
  "include": ["src"]
}
```

Add workspace dependencies according to the dependency diagram using the
npm-compatible `"*"`. Add `"bin": {"aiperf-flow": "./dist/main.js"}` to the CLI package.
Add `"jsx": "react-jsx"` to the runtime package's TypeScript configuration.

- [ ] **Step 4: Install current stable dependencies**

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd apps/aiperf-flow
npm install -w @aiperf/flow-schema zod
npm install -w @aiperf/flow-language chevrotain
npm install -w @aiperf/flow-runtime react react-dom
npm install -D -w @aiperf/flow-runtime @types/react @types/react-dom @testing-library/react jsdom esbuild
npm install -w @aiperf/flow-cli commander
npm install -D -w @aiperf/flow-cli esbuild
npm install -D -w @aiperf/flow-schema -w @aiperf/flow-language -w @aiperf/flow-compiler -w @aiperf/flow-runtime -w @aiperf/flow-cli typescript vitest
```

Expected: npm updates `package-lock.json` with stable package versions and no
peer-dependency errors.

- [ ] **Step 5: Add package entry points and verify the workspace**

Create `src/index.ts` in each package containing a package-specific exported
constant, for example:

```ts
export const FLOW_SCHEMA_VERSION = 1 as const;
```

Use distinct names in other packages:
`FLOW_LANGUAGE_VERSION`, `FLOW_COMPILER_VERSION`,
`FLOW_RUNTIME_VERSION`, and `FLOW_CLI_VERSION`.

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd apps/aiperf-flow
npm run flow:build
```

Expected: Flow workspace build PASS; legacy `apps/explainers` remains untouched.

---

### Task 2: Define diagnostics, capability descriptors, and Flow IR

**Files:**
- Create: `apps/aiperf-flow/packages/schema/src/source.ts`
- Create: `apps/aiperf-flow/packages/schema/src/diagnostic.ts`
- Create: `apps/aiperf-flow/packages/schema/src/capability.ts`
- Create: `apps/aiperf-flow/packages/schema/src/ir.ts`
- Modify: `apps/aiperf-flow/packages/schema/src/index.ts`
- Test: `apps/aiperf-flow/packages/schema/test/capability.test.ts`
- Test: `apps/aiperf-flow/packages/schema/test/ir.test.ts`

**Interfaces:**
- Produces `SourceRange`, `Diagnostic`, `CapabilityDescriptor`,
  `CapabilityRegistryManifest`, `FlowIr`, and `parseFlowIr`.
- Consumed by every later package.

- [ ] **Step 1: Write failing schema tests**

Test that:

```ts
const range = {
  source: "request-flow.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 9, line: 1, column: 10 },
};

expect(
  parseFlowIr({
    irVersion: 1,
    id: "request-flow",
    title: "Request flow",
    capabilities: [{ id: "core.shape", range: "^1.0.0" }],
    scenes: [],
    sourceMap: range,
  }).id,
).toBe("request-flow");
```

Also assert that duplicate capability IDs are rejected by
`createCapabilityManifest`, and that an IR element missing `accessibility` and
`fallback` fails with diagnostic code `IR_INVALID`.

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd apps/aiperf-flow
npm test -w @aiperf/flow-schema
```

Expected: FAIL because the schema modules do not exist.

- [ ] **Step 2: Implement source and diagnostic contracts**

Define:

```ts
export type SourcePosition = Readonly<{
  offset: number;
  line: number;
  column: number;
}>;

export type SourceRange = Readonly<{
  source: string;
  start: SourcePosition;
  end: SourcePosition;
}>;

export type DiagnosticSeverity = "error" | "warning" | "info";

export type Diagnostic = Readonly<{
  code: string;
  severity: DiagnosticSeverity;
  message: string;
  range: SourceRange;
  repair?: string;
}>;

export type Result<T> =
  | Readonly<{ ok: true; value: T; diagnostics: readonly Diagnostic[] }>
  | Readonly<{ ok: false; diagnostics: readonly Diagnostic[] }>;
```

Provide `diagnostic()` and `hasErrors()` helpers without mutable global state.

- [ ] **Step 3: Implement capability descriptors**

Define capability kinds:

```ts
export const CAPABILITY_KINDS = [
  "primitive",
  "layout",
  "effect",
  "transform",
  "action",
  "asset-loader",
  "exporter",
] as const;
```

`CapabilityDescriptor` must include `id`, semantic `version`, `kind`, human
description, supported IR node kinds, deterministic flag, accessibility
contract, fallback capability ID, and cost model. Implement
`createCapabilityManifest(descriptors)` so it sorts by ID and returns a
diagnostic for duplicate IDs.

Register these foundation capabilities:

- `core.group@1.0.0`
- `core.rect@1.0.0`
- `core.text@1.0.0`
- `core.connector@1.0.0`
- `core.camera@1.0.0`
- `core.timeline@1.0.0`
- `core.inspect@1.0.0`

- [ ] **Step 4: Implement versioned IR with Zod**

The foundation `FlowIr` contains:

```ts
export type FlowIr = Readonly<{
  irVersion: 1;
  id: string;
  title: string;
  capabilities: readonly CapabilityRequirement[];
  tokens: Readonly<Record<string, string | number | boolean>>;
  scenes: readonly SceneIr[];
  sourceMap: SourceRange;
}>;
```

Each `SceneIr` includes stable ID, title, summary, ordered render-tree roots,
camera keyframes, timeline cues, narration transcript, interactions, responsive
variants, scene accessibility, and fallback summary. Foundation render nodes
are discriminated `group`, `rect`, `text`, and `connector` objects. Every node
includes `id`, exact geometry, style, `accessibility`, `fallback`, and
`sourceMap`.

Use strict Zod objects so unknown fields fail. Export:

```ts
export function parseFlowIr(input: unknown): FlowIr;
export function safeParseFlowIr(input: unknown): Result<FlowIr>;
```

Map Zod issue paths into `IR_INVALID` diagnostics.

- [ ] **Step 5: Verify the schema package**

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd apps/aiperf-flow
npm test -w @aiperf/flow-schema
npm run build -w @aiperf/flow-schema
```

Expected: PASS.

---

### Task 3: Parse and format the foundation `.flow` language

**Files:**
- Create: `apps/aiperf-flow/packages/language/src/tokens.ts`
- Create: `apps/aiperf-flow/packages/language/src/ast.ts`
- Create: `apps/aiperf-flow/packages/language/src/parser.ts`
- Create: `apps/aiperf-flow/packages/language/src/formatter.ts`
- Modify: `apps/aiperf-flow/packages/language/src/index.ts`
- Test: `apps/aiperf-flow/packages/language/test/parser.test.ts`
- Test: `apps/aiperf-flow/packages/language/test/formatter.test.ts`

**Interfaces:**
- Consumes `SourceRange`, `Diagnostic`, and `Result`.
- Produces `DocumentAst`, `parseDocument(source, sourceName)`, and
  `formatDocument(ast)`.

- [ ] **Step 1: Write parser tests against one explicit source**

Use this complete foundation grammar fixture:

```text
flow "Request flow" as request-flow {
  language 1
  require core.rect "^1.0.0"
  require core.text "^1.0.0"
  require core.connector "^1.0.0"
  token accent = "#7aa2f7"

  scene "Execution boundary" as execution {
    summary "The CLI starts a runtime that dispatches work."

    rect cli {
      x 40
      y 100
      width 160
      height 72
      fill token(accent)
      label "CLI"
      role "img"
      description "Command-line process"
      fallback "CLI"
    }

    rect runtime {
      x 300
      y 100
      width 180
      height 72
      fill "#244a35"
      label "Runtime"
      role "img"
      description "Execution runtime"
      fallback "Runtime"
    }

    connector spawn {
      from cli
      to runtime
      label "spawn --execute"
      stroke token(accent)
      fallback "CLI starts Runtime"
    }

    camera main {
      at 0 frame cli,runtime zoom 1
      at 2000 frame runtime zoom 1.4
    }

    timeline primary {
      at 0 reveal cli duration 400
      at 800 trace spawn duration 1200
      at 2200 reveal runtime duration 400
    }

    interaction inspect-runtime {
      on select runtime
      do inspect runtime
    }

    responsive compact when width < 720 {
      set runtime.x = 40
      set runtime.y = 240
    }

    narrate "The CLI starts a fresh runtime and dispatches work."
    reading-order cli,runtime,spawn
    fallback "CLI starts Runtime."
  }
}
```

Assert exact document ID, token value, scene ID, three render declarations,
camera keyframes, timeline cues, interaction, responsive overrides, narration,
reading order, and source ranges.

Also test malformed input returns at least two diagnostics with code
`PARSE_UNEXPECTED_TOKEN` and source names/lines.

- [ ] **Step 2: Define tokens and AST**

Use Chevrotain token categories for whitespace/comments, identifiers, quoted
strings, numbers, punctuation, comparison operators, and keywords. Comments are
skipped but line/column metadata remains correct.

Define readonly AST nodes for:

- document, language declaration, requirement, token, and scene;
- rect and connector declarations;
- camera and timeline declarations;
- interaction and responsive declarations;
- narration, reading order, and fallback;
- literal, token reference, and comma-separated reference list.

Every AST node includes `kind` and `sourceMap`.

- [ ] **Step 3: Implement parser and diagnostic recovery**

Implement:

```ts
export function parseDocument(
  source: string,
  sourceName: string,
): Result<DocumentAst>;
```

The lexer reports `LEX_INVALID_CHARACTER`. The parser uses Chevrotain recovery
and converts every parser error into `PARSE_UNEXPECTED_TOKEN` with expected
token names and the offending source range. It returns `ok: false` whenever any
lexing or parsing error exists.

The grammar must parse exactly the fixture syntax above; no unparsed property
bag or generic “unknown statement” node is allowed.

- [ ] **Step 4: Implement canonical formatting**

Implement:

```ts
export function formatDocument(document: DocumentAst): string;
```

Rules:

- two-space indentation;
- one blank line between top-level declarations and scene blocks;
- declaration properties in schema order;
- one statement per line;
- comma-separated references without spaces;
- escaped quoted strings;
- trailing newline;
- formatting is idempotent.

Add a test that parses messy but valid source, formats it, reparses it, compares
semantic AST content excluding source ranges, formats again, and gets the same
string.

- [ ] **Step 5: Verify language behavior**

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd apps/aiperf-flow
npm test -w @aiperf/flow-language
npm run build -w @aiperf/flow-language
```

Expected: PASS.

---

### Task 4: Link, validate, and lower AST into Flow IR

**Files:**
- Create: `apps/aiperf-flow/packages/compiler/src/link.ts`
- Create: `apps/aiperf-flow/packages/compiler/src/validate.ts`
- Create: `apps/aiperf-flow/packages/compiler/src/lower.ts`
- Create: `apps/aiperf-flow/packages/compiler/src/index.ts`
- Test: `apps/aiperf-flow/packages/compiler/test/compile.test.ts`

**Interfaces:**
- Consumes `DocumentAst`, `CapabilityRegistryManifest`, and schema IR types.
- Produces `compileSource(request): Result<FlowIr>`.

- [ ] **Step 1: Write failing compile tests**

Define:

```ts
export type CompileRequest = Readonly<{
  source: string;
  sourceName: string;
  capabilities: CapabilityRegistryManifest;
  strict: boolean;
}>;
```

Tests must assert:

- the Task 3 source compiles into valid IR;
- output requirements are sorted by capability ID;
- `token(accent)` resolves to `#7aa2f7`;
- connector endpoints resolve to stable node IDs;
- responsive overrides resolve to typed property paths;
- unresolved node IDs produce `LINK_UNKNOWN_REFERENCE`;
- unknown capabilities produce `CAPABILITY_MISSING`;
- duplicate IDs produce `LINK_DUPLICATE_ID`;
- missing reading-order entries, fallback, role, or description produce
  `ACCESSIBILITY_REQUIRED`;
- strict mode promotes `NARRATION_SHORT` to an error.

- [ ] **Step 2: Implement symbol linking**

`linkDocument(document)` creates immutable symbol tables scoped by document and
scene. It resolves node IDs, connector endpoints, camera targets, timeline
targets, interaction targets, reading-order IDs, responsive target paths, and
token references.

Store references as stable IDs in a `LinkedDocument`; never retain mutable AST
node pointers. Emit duplicate and unresolved diagnostics in source order.

- [ ] **Step 3: Implement semantic validation**

Validate:

- capability availability and compatible major versions;
- positive finite geometry;
- connector endpoints refer to render nodes;
- timeline times and durations are finite and nonnegative;
- camera and timeline labels are unique;
- interaction event/action pairs are supported;
- responsive property paths exist and value types match;
- reading order names every interactive or meaningful node exactly once;
- every render node has role, description, and fallback;
- every scene has summary, narration, reading order, and fallback.

Use diagnostic codes from the tests. Preserve warnings in successful results.

- [ ] **Step 4: Implement deterministic lowering**

`lowerDocument(linked)` must:

- produce `irVersion: 1`;
- resolve all token references to literal values;
- sort capability requirements by ID;
- preserve authored render-tree order;
- lower coordinates to finite CSS-pixel logical units;
- validate camera and timeline values as non-negative safe-integer
  milliseconds; `Clock.nowNs()` is used only to advance this virtual time;
- convert responsive overrides to typed IR operations;
- attach source maps to every lowered object;
- parse the result through `parseFlowIr` before returning it.

Expose:

```ts
export function compileSource(request: CompileRequest): Result<FlowIr>;
```

- [ ] **Step 5: Run compiler tests**

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd apps/aiperf-flow
npm test -w @aiperf/flow-compiler
npm run build -w @aiperf/flow-compiler
```

Expected: PASS.

---

### Task 5: Pack deterministic runtime chunks

**Files:**
- Create: `apps/aiperf-flow/packages/compiler/src/pack.ts`
- Modify: `apps/aiperf-flow/packages/compiler/src/index.ts`
- Test: `apps/aiperf-flow/packages/compiler/test/pack.test.ts`

**Interfaces:**
- Consumes validated `FlowIr`.
- Produces `PackedFlow`, `PackManifest`, and `packFlow(ir)`.

- [ ] **Step 1: Write failing deterministic pack tests**

Define:

```ts
export type PackedFile = Readonly<{
  path: string;
  content: Uint8Array;
  mediaType: string;
  hash: string;
}>;

export type PackedFlow = Readonly<{
  manifest: PackManifest;
  files: readonly PackedFile[];
}>;
```

Compile and pack the same source twice. Assert byte-identical file contents,
identical SHA-256 hashes, lexicographically sorted file paths, one manifest, and
one scene chunk. Change a scene label and assert only the scene chunk and
manifest hashes change.

- [ ] **Step 2: Implement canonical JSON serialization**

Implement a recursive serializer that:

- sorts object keys lexicographically;
- preserves array order;
- rejects `undefined`, non-finite numbers, functions, symbols, and bigint;
- emits UTF-8 without insignificant whitespace;
- appends one newline.

Export `canonicalJson(value): Uint8Array`.

- [ ] **Step 3: Implement scene chunking and manifest creation**

`packFlow(ir)` writes:

- `flow.manifest.json`;
- `chunks/scene-<scene-id>.<hash-prefix>.json`;
- `transcript.txt`;

The manifest includes format version, Flow document metadata, required
capabilities, ordered scene descriptors, source filename, content hashes, and
relative chunk paths. Hash canonical bytes with Node `crypto.createHash`.

Validate all generated relative paths against traversal and duplicate-path
attacks.

- [ ] **Step 4: Verify packing**

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd apps/aiperf-flow
npm test -w @aiperf/flow-compiler -- pack.test.ts
npm run build -w @aiperf/flow-compiler
```

Expected: PASS.

---

### Task 6: Interpret normal and packed IR through the foundation fallback

**Files:**
- Create: `apps/aiperf-flow/packages/runtime/src/registry.ts`
- Create: `apps/aiperf-flow/packages/runtime/src/store.ts`
- Create: `apps/aiperf-flow/packages/runtime/src/renderer.tsx`
- Create: `apps/aiperf-flow/packages/runtime/src/player.ts`
- Create: `apps/aiperf-flow/packages/runtime/src/app.tsx`
- Create: `apps/aiperf-flow/packages/runtime/src/site.tsx`
- Create: `apps/aiperf-flow/packages/runtime/src/theme.css`
- Create: `apps/aiperf-flow/packages/runtime/src/index.ts`
- Create: `apps/aiperf-flow/packages/runtime/scripts/build-site.mjs`
- Test: `apps/aiperf-flow/packages/runtime/test/registry.test.ts`
- Test: `apps/aiperf-flow/packages/runtime/test/renderer.test.tsx`
- Test: `apps/aiperf-flow/packages/runtime/test/player.test.ts`

**Interfaces:**
- Consumes validated normal or packed manifest and scene IR only.
- Produces `CapabilityRegistry`, `SceneRenderer`, `TimelinePlayer`,
  `FlowApp`, and `dist/site.js`.

- [ ] **Step 1: Write registry and renderer tests**

Assert:

- duplicate runtime capability IDs throw `DuplicateCapabilityError`;
- missing required capabilities render the authored scene fallback;
- equivalent normal and packed IR render the same semantic scene;
- `rect`, `text`, and `connector` IR render semantic SVG/HTML;
- renderer state and timeline evaluation remain separate from SVG element
  creation so Plan 2 can add Canvas and semantic-twin backends without changing
  Flow IR;
- SVG elements expose accessible names and descriptions;
- selecting `runtime` executes the declared `inspect` action and opens a
  keyboard-focusable inspector;
- reduced-motion mode applies the authored final timeline state without
  animation;
- an invalid scene chunk renders scene summary, fallback, transcript access,
  and working navigation.

- [ ] **Step 2: Implement the closed capability registry**

Define:

```ts
export type RuntimeCapability<TNode extends RenderNodeIr = RenderNodeIr> =
  Readonly<{
    descriptor: CapabilityDescriptor;
    render(node: TNode, context: RenderContext): ReactNode;
  }>;

export class CapabilityRegistry {
  register(capability: RuntimeCapability): void;
  require(id: string): RuntimeCapability;
  manifest(): CapabilityRegistryManifest;
}
```

Register group, rect, text, connector, camera, timeline, and inspect
capabilities. Runtime code may invoke only registered capability functions.

- [ ] **Step 3: Implement immutable scene state and renderer**

The store owns:

- current scene ID;
- selected node ID;
- inspector state;
- playback time and status;
- active responsive variant;
- temporary camera takeover.

Use a reducer with serializable actions. `SceneRenderer` traverses render-tree
IR, delegates each node to the registry, and draws one responsive semantic SVG
fallback stage with HTML overlay slots. Connector geometry uses authored node
centers in this foundation slice. Keep scene/timeline evaluation independent
from SVG creation; the next runtime plan replaces the preferred visual path
with Canvas while retaining SVG as fallback.

The fallback visual baseline uses dark navy depth planes, blue control accents,
green execution accents, quiet technical typography, visible focus, and
token-driven styles. It establishes token names and semantic contrast, not the
final high-fidelity material, lighting, motion, or compositing system. All CSS
selectors are package-scoped.

- [ ] **Step 4: Implement deterministic timeline playback**

Define a `Clock` interface:

```ts
export interface Clock {
  nowNs(): bigint;
  requestFrame(callback: () => void): number;
  cancelFrame(handle: number): void;
}
```

`TimelinePlayer` supports play, pause, seek, reset, current time, and final
state. It computes scene state only from timeline IR and current virtual time.
Tests use a `ManualClock`; production uses `performance.now()`.

Wall-clock time advances integer virtual time but never becomes authored scene
state. Add a test proving that seeking directly to a foundation beat produces
the same semantic state as continuous playback to that beat.

- [ ] **Step 5: Implement site loading and resilience**

`site.tsx` loads `./flow.manifest.json`, verifies manifest version, checks
required capabilities, lazy-loads the first scene chunk, and mounts
`FlowApp`.

The app provides:

- previous/next scene navigation;
- play/pause/restart;
- narration transcript and synchronized highlight region;
- scene title and progress;
- responsive stage;
- inspector;
- semantic outline synchronized with the fallback stage;
- skip-to-transcript link;
- error boundary with fallback summary and transcript.

- [ ] **Step 6: Bundle the generic runtime site**

`build-site.mjs` invokes esbuild with:

```js
await build({
  entryPoints: ["src/site.tsx"],
  bundle: true,
  minify: true,
  format: "esm",
  platform: "browser",
  target: ["es2022"],
  outfile: "dist/site.js",
  sourcemap: true,
});
```

Update runtime `build` to run TypeScript declaration compilation followed by
the site bundler.

- [ ] **Step 7: Verify runtime behavior**

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd apps/aiperf-flow
npm test -w @aiperf/flow-runtime
npm run build -w @aiperf/flow-runtime
```

Expected: PASS and `packages/runtime/dist/site.js` exists.

---

### Task 7: Build the CLI and standalone static site

**Files:**
- Create: `apps/aiperf-flow/packages/cli/src/commands.ts`
- Create: `apps/aiperf-flow/packages/cli/src/site.ts`
- Create: `apps/aiperf-flow/packages/cli/src/main.ts`
- Create: `apps/aiperf-flow/packages/cli/src/index.ts`
- Test: `apps/aiperf-flow/packages/cli/test/commands.test.ts`
- Test: `apps/aiperf-flow/packages/cli/test/site.test.ts`

**Interfaces:**
- Consumes parser, formatter, compiler, packer, runtime bundle, and foundation
  capabilities.
- Produces CLI commands `format`, `check`, `build`, `inspect`, and
  `capabilities`.

- [ ] **Step 1: Write command tests using temporary directories**

Test:

- `format --check` exits 1 for noncanonical source and 0 after formatting;
- `check --json` emits stable JSON diagnostics and exits 1 for invalid source;
- `inspect --ir` emits Flow IR JSON;
- `capabilities --json` emits sorted descriptors;
- `build --out <dir>` writes a complete relative-path static site;
- two builds produce byte-identical output trees;
- build refuses a nonempty output directory unless `--clean` is supplied;
- build never writes outside the requested output directory.

- [ ] **Step 2: Implement command services independently from Commander**

Export testable functions:

```ts
export function formatCommand(request: FormatRequest): Promise<CommandResult>;
export function checkCommand(request: CheckRequest): Promise<CommandResult>;
export function inspectCommand(request: InspectRequest): Promise<CommandResult>;
export function capabilitiesCommand(
  request: CapabilitiesRequest,
): Promise<CommandResult>;
export function buildCommand(request: BuildRequest): Promise<CommandResult>;
```

`CommandResult` contains exit code, stdout, and stderr. File writes use explicit
UTF-8 and atomic rename. JSON diagnostics include code, severity, message,
source, start/end positions, and repair.

- [ ] **Step 3: Implement static-site emission**

`writeStaticSite(packed, outDir)` writes:

- all packed files;
- copied runtime `site.js` and source map;
- copied `theme.css`;
- an `index.html` with relative asset URLs, viewport metadata, title,
  transcript fallback, and `<script type="module" src="./site.js">`;
- `404.html` equal to `index.html` for static hosts.

Sort all writes by relative path. Reject symlinks and traversal. Write into a
temporary sibling directory and atomically rename only after every file
succeeds.

- [ ] **Step 4: Wire Commander without embedding business logic**

`main.ts` defines the public command surface from the spec and delegates to
command services. It prints diagnostics to stderr, content to stdout, and sets
`process.exitCode` without calling `process.exit()`.

Foundation command support:

```text
aiperf-flow format <sources...> [--check]
aiperf-flow check <sources...> [--strict] [--json]
aiperf-flow build <source> --out <directory> [--strict] [--clean]
aiperf-flow inspect <source> --ast|--ir|--manifest
aiperf-flow capabilities [--json]
```

- [ ] **Step 5: Verify CLI and site output**

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd apps/aiperf-flow
npm test -w @aiperf/flow-cli
npm run build -w @aiperf/flow-cli
node packages/cli/dist/main.js --help
```

Expected: tests and build PASS; help lists all five foundation commands.

---

### Task 8: Add the reference source and end-to-end proof

**Files:**
- Create: `apps/aiperf-flow/examples/foundation/request-flow.flow`
- Create: `apps/aiperf-flow/examples/foundation/request-flow.expected.json`
- Create: `apps/aiperf-flow/packages/cli/test/e2e-foundation.test.ts`
- Modify: `apps/aiperf-flow/README.md`

**Interfaces:**
- Consumes the complete foundation pipeline.
- Produces one checked-in explicit `.flow` exemplar and a reproducible
  standalone site.

- [ ] **Step 1: Add the full foundation source**

Use the Task 3 source as the base, then add a second scene that reverses the
responsive layout and continues stable IDs through qualified references. The
fixture must exercise:

- every foundation capability;
- literal and token-based style values;
- exact geometry;
- camera and timeline tracks;
- interaction;
- compact responsive variant;
- narration;
- reading order;
- node and scene fallbacks.

Store expected normalized IR in `request-flow.expected.json`. Generate it only
through `aiperf-flow inspect --ir`, then review it before checking it in.

- [ ] **Step 2: Add an end-to-end build test**

The test runs the command service against the reference source, reads every
output file, and asserts:

- manifest and scene chunks parse through strict schemas;
- all required capabilities exist in runtime registry manifest;
- `index.html`, `404.html`, `site.js`, `theme.css`, transcript, manifest, and
  scene chunks exist;
- every URL in `index.html` is relative;
- transcript contains both scene narrations;
- rebuilt output hashes match;
- rendering the first loaded scene in jsdom exposes title, SVG accessible name,
  nodes, connector label, playback controls, transcript link, inspector, and a
  semantic outline with the same meaningful entities and reading order.

- [ ] **Step 3: Document the supported workflow without example scripts**

Update `apps/aiperf-flow/README.md` with:

- Flow is a separate app from the legacy `apps/explainers` SPA;
- engine package architecture and dependency direction;
- exact `format`, `check`, `inspect`, `build`, and preview commands;
- statement that the foundation grammar is a vertical slice, not the final
  capability catalog;
- statement that the foundation React/SVG renderer is the simplified fallback,
  not the final cinematic Canvas renderer;
- pointer to the approved design and delivery roadmap;
- static deployment contract;
- no visual editor, runtime AI, or generated TypeScript scenes.

- [ ] **Step 4: Run complete verification**

Run:

```bash
source /home/anthony/nvidia/projects/aiperf/ajc/rust/.venv/bin/activate
cd apps/aiperf-flow
npm run flow:check
node packages/cli/dist/main.js format examples/foundation/request-flow.flow --check
node packages/cli/dist/main.js check examples/foundation/request-flow.flow --strict
rm -rf /tmp/aiperf-flow-foundation
node packages/cli/dist/main.js build examples/foundation/request-flow.flow --out /tmp/aiperf-flow-foundation
test -f /tmp/aiperf-flow-foundation/index.html
test -f /tmp/aiperf-flow-foundation/flow.manifest.json
cd ../..
/usr/bin/python3 tools/check_docs_current.py
```

Expected: every command exits 0.

- [ ] **Step 5: Review the vertical-slice completion gate**

Confirm all statements are true:

- `.flow` is parsed by the real block-language parser.
- AST, linked model, Flow IR, packed chunks, and runtime are separate
  contracts.
- Normal and packed IR produce equivalent runtime behavior.
- No source or generated TypeScript scene is shipped to the browser.
- Capability availability is checked both at compile and runtime boundaries.
- Output is deterministic and static-host safe.
- The reference scene remains navigable through fallback and transcript paths
  when a visual capability fails.
- The legacy `apps/explainers` SPA remains untouched.
- Plan 2 can add the evaluated-scene, Canvas, semantic-twin, and fallback
  backends without changing Flow IR meaning or replacing package boundaries.
- Plan 3 can expand syntax and modules without replacing any foundation package
  or public interface.

## Self-review results

- **Spec coverage:** This plan proves every architectural boundary and provides
  representative schema fields for document metadata, visual primitives,
  geometry, style, camera, motion, interaction, responsiveness, narration,
  accessibility, fallbacks, capabilities, packing, and static output. The
  roadmap assigns the final rendering substrate and full vocabulary coverage to
  Plans 2–9.
- **Scope:** The plan is a vertical slice, not an attempt to implement the
  complete production schema in one change. It leaves independently useful
  software: one explicit source can be checked, packed, rendered, and deployed.
- **Type consistency:** Package names, `Result<T>`, `SourceRange`,
  `CapabilityRegistryManifest`, `FlowIr`, `compileSource`,
  `packFlow`, and command-service signatures are defined before use.
- **Placeholder scan:** The plan contains no deferred implementation markers.
  Later capabilities are explicitly assigned to named roadmap plans.

